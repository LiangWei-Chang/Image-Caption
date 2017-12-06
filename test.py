import tensorflow as tf
import _pickle as cPickle
import pandas as pd
import os
import sys
sys.path.append('model')
sys.path.append('utils')
from tqdm import tqdm

def generate_captions(model, enc_map, dec_map, img_test, max_len=15):
    img_ids, caps = [], []
    pbar = tqdm(total=len(img_test.items()))
    for img_id, img in img_test.items():
        img_ids.append(img_id)
        img = np.expand_dims(img, axis=0)
        caps.append(model.inference(img, enc_map, dec_map))
        pbar.update(1)
    return pd.DataFrame({'img_id': img_ids, 'caption': caps}).set_index(['img_id'])

enc_map = cPickle.load(open('dataset/enc_map.pkl', 'rb'))  # token => id
dec_map = cPickle.load(open('dataset/dec_map.pkl', 'rb'))  # id => token

# create model
tf.reset_default_graph()
model = ImageCaptionModel(hparams, mode='inference')
model.build()

# load test image  size=20548
img_test = cPickle.load(open('dataset/test_img256.pkl', 'rb'))

# generate caption to csv file
df_predict = generate_captions(model, enc_map, dec_map, img_test)
df_predict.to_csv('generated/demo.csv')

os.system('cd CIDErD && ./gen_score -i ../generated/demo.csv -r ../generated/score.csv')
