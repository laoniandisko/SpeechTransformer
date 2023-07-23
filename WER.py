class WERCalculator:
    def compute_wer(self, reference, hypothesis):
        # 将参考答案和识别结果划分为单词列表
        ref_words = [_ for _ in reference]
        hyp_words = [_ for _ in hypothesis]

        # 初始化编辑距离矩阵
        dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            dp[i][0] = i

        for j in range(len(hyp_words) + 1):
            dp[0][j] = j

        # 动态规划计算编辑距离
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1

        # 计算WER
        wer = dp[len(ref_words)][len(hyp_words)] / len(ref_words) * 100
        return wer

wer_calculator = WERCalculator()
reference = "这是一个测试"
hypothesis = "这是一个实例"
wer = wer_calculator.compute_wer(reference, hypothesis)
# print(f"WER: {wer:.2f}%")