import tensorflow as tf
import _pickle as cPickle
import pandas as pd
import os
import sys
sys.path.append('model')
sys.path.append('utils')
from tqdm import tqdm
from model import *
from utils import *
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_captions(model, enc_map, dec_map, img_test, max_len=15):
    img_ids, caps = [], []

    with tf.Session() as sess:
        saver = tf.train.Saver()
        # restore variables from disk.
        ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, tf.train.latest_checkpoint(hparams.ckpt_dir))

            pbar = tqdm(total=len(img_test.items()))
            for img_id, img in img_test.items():
                img_ids.append(img_id)
                img = np.expand_dims(img, axis=0)
                caps.append(model._beam_search(sess, img, enc_map, dec_map, beam_size=5))
                pbar.update(1)
        else:
            print("No checkpoint found.")
    return pd.DataFrame({'img_id': img_ids, 'caption': caps}).set_index(['img_id'])
# def generate_captions(model, enc_map, dec_map, img_test, max_len=15):
#     img_ids, caps = [], []
#
#     pbar = tqdm(total=len(img_test.items()))
#     df_test = pd.DataFrame(list(img_test.items()), columns=['img_id', 'img'])
#     batch_num = df_test.shape[0] // batch_size
#     with tf.Session() as sess:
#         saver = tf.train.Saver()
#         # restore variables from disk.
#         ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, tf.train.latest_checkpoint(hparams.ckpt_dir))
#
#             for i in tqdm(range(batch_num)):
#                 st = i * batch_size
#                 ed = (i + 1) * batch_size if not i == batch_num - 1 else df_test.shape[0]
#                 img = np.zeros((ed-st, 256))
#                 for j in range(img.shape[0]):
#                     img[j] = df_test['img'][j]
#                 img_ids += list(df_test['img_id'][st:ed])
#                 caps += model.inference(img, enc_map, dec_map)
#         else:
#             print("No checkpoint found.")
#     return pd.DataFrame({'img_id': img_ids, 'caption': caps}).set_index(['img_id'])

enc_map = cPickle.load(open('dataset/enc_map.pkl', 'rb'))  # token => id
dec_map = cPickle.load(open('dataset/dec_map.pkl', 'rb'))  # id => token
hparams = get_hparams()

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
