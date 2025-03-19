import numpy as np
import torch
import torch.nn as nn

class DynamicSparseScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, temperature, window_size=2, gating_threshold=0.5, attn_dropout=0.1):
        """
        :param d_k:          Query/Key 
        :param temperature:  sqrt(d_k)
        :param window_size:  局部窗口
        :param gating_threshold: 门控激活阈值
        :param attn_dropout: 注意力层dropout率
        """
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.gating_threshold = gating_threshold

        # 用于计算门控分数的可学习网络（简单起见，这里只用一个线性层）
        self.gating_linear = nn.Linear(d_k, 1)

        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        :param q:    (B*heads) x L_q x d_k
        :param k:    (B*heads) x L_k x d_k
        :param v:    (B*heads) x L_v x d_v
        :param mask: (B*heads) x L_q x L_k, 如果需要额外的mask可传入
        """
        B_heads, L_q, _ = q.shape
        _, L_k, _ = k.shape


        attn = torch.bmm(q, k.transpose(1, 2))  # (B*heads) x L_q x L_k
        attn = attn / self.temperature


        gating_score = self.gating_linear(q)   
        gating_prob = torch.sigmoid(gating_score) 

        gating_mask = (gating_prob > self.gating_threshold)

        window = self.window_size


        for b in range(B_heads):
            for i in range(L_q):
                if not gating_mask[b, i, 0]:  

                    left = max(0, i - window)
                    right = min(L_k, i + window + 1)

                    if left > 0:
                        attn[b, i, 0:left] = float('-inf')
                    if right < L_k:
                        attn[b, i, right:] = float('-inf')

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), float('-inf'))


        attn = self.softmax(attn)              # (B*heads, L_q, L_k)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)           # (B*heads, L_q, d_v)

        return output, attn


class DynamicSparseMultiHeadAttention(nn.Module):
    ''' 
    将上述动态稀疏注意力嵌入到多头注意力模块中，
    基本结构和原来的 MultiHeadAttention 相似。
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, 
                 window_size=2, gating_threshold=0.5):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # 这里替换成 DynamicSparseScaledDotProductAttention
        self.attention = DynamicSparseScaledDotProductAttention(
            d_k=d_k, 
            temperature=np.power(d_k, 0.5),
            window_size=window_size,
            gating_threshold=gating_threshold,
            attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        :param q:  [batch_size, len_q, d_model]
        :param k:  [batch_size, len_k, d_model]
        :param v:  [batch_size, len_v, d_model]
        :param mask: [batch_size, len_q, len_k] (可选)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # 1)Q, K, V
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # (B*n_head, L, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)


        if mask is not None:
            mask = mask.unsqueeze(1)                   # [B, 1, L_q, L_k]
            mask = mask.repeat(n_head, 1, 1, 1)        # [B*n_head, 1, L_q, L_k]
            mask = mask.squeeze(1)                     # [B*n_head, L_q, L_k]


        output, attn = self.attention(q, k, v, mask=mask)


        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)


        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
