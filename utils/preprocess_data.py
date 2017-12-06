import os
import sys
sys.path.append('utils')

from utils import *
import pandas as pd
import numpy as np


def build_caption_vector(enc_map, df):
    img_ids, cap_vecs = [], []
    for idx, row in df.iterrows():
        cap_vec = [ enc_map[x] for x in row['caption'].split() ]
        cap_vec = [ enc_map['<BEG>'] ] + cap_vec + [ enc_map['<END>'] ]
        img_ids.append(row['img_id'])
        cap_vecs.append(cap_vec)

    return pd.DataFrame({'img_id': img_ids, 'caption': cap_vecs}).set_index(['img_id'])

def build_vocab(df, vocab):
    threshold = 20
    # Compite occurance of each vocab
    voc_count = {v: 0 for v in vocab}
    for img_id, row in df.iterrows():
        for w in row['caption'].split():
            voc_count[w] += 1

    encode_map = {'<BEG>': 0, '<END>': 1, '<RARE>': 2}
    decode_map = {0: '<BEG>', 1: '<END>', 2: '<RARE>'}
    idx = 3
    for word, count in voc_count.items():
        if count < threshold:
            encode_map[word] = encode_map['<RARE>']
        else:
            encode_map[word] = idx
            decode_map[idx] = word
            idx += 1

    return encode_map, decode_map

def main():
    vocab = load_pickle('./dataset/vocab.pkl')
    df_train = pd.read_csv(os.path.join('./dataset', 'train.csv'))

    encode_map, decode_map = build_vocab(df_train, vocab)
    save_pickle(encode_map, './dataset/enc_map.pkl')
    save_pickle(decode_map, './dataset/dec_map.pkl')

    df_caption_vector = build_caption_vector(encode_map, df_train)
    df_caption_vector.to_csv('./dataset/train_cap_vec.csv')

if __name__ == '__main__':
    main()
