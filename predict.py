import argparse
import pickle
import random
from shutil import copyfile
from WER import WERCalculator
wer_calculator = WERCalculator()
import torch
import time
from tqdm import tqdm

from config import pickle_file, device, input_dim, LFR_m, LFR_n
from data_gen import build_LFR_features
from transformer.transformer import Transformer
from utils import extract_feature, ensure_folder


def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=5, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=5, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with open('data/char_list.pkl', 'rb') as file:
        char_list = pickle.load(file)
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list1 = data["IVOCAB"]
    samples = data['test']

    filename = 'speech-transformer-cn.pt'
    print('loading model: {}...'.format(filename))
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    # samples = random.sample(samples, 10)
    samples = random.sample(samples, 100)
    ensure_folder('audios')
    results = []
    WER = 0
    word_sum = 0

    for i, sample in enumerate(tqdm(samples)):
        wave = sample['wave']
        trn = sample['trn']

        copyfile(wave, 'audios/audio_{}.wav'.format(i))

        feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = torch.from_numpy(feature).to(device)
        input_length = [input[0].shape[0]]
        input_length = torch.LongTensor(input_length).to(device)
        nbest_hyps = model.recognize(input, input_length, char_list, args)
        out_list = []
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out]
            out = ''.join(out)
            out_list.append(out)
        print('OUT_LIST: {}'.format(out_list))
        del input
        del input_length
        del nbest_hyps
        torch.cuda.empty_cache()
        time.sleep(0.07)

        gt = [char_list1[idx] for idx in trn]
        gt = ''.join(gt)
        # print('GT: {}\n'.format(gt))


        # for sentence in out_list:
        WER_tmp = wer_calculator.compute_wer(reference = out_list[0][5:-5],hypothesis = gt[:-5])
        WER+=WER_tmp*len(gt[:-5])
        word_sum+=len(gt[:-5])



        results.append({'out_list_{}'.format(i): out_list, 'gt_{}'.format(i): gt})

    WER = WER/word_sum
    print(f"WER: {WER:.2f}%")

    import json

    with open('results.json', 'w',encoding = "utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
