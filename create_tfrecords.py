import os
import pandas as pd
import tensorflow as tf
import _pickle as cPickle
from tqdm import tqdm
import numpy as np

def create_tfrecords(df_cap, img_df, filename, num_files=5):

    ''' create tfrecords for dataset '''
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    num_records_per_file = df_cap.shape[0] // num_files

    total_count = 0

    length = []
    for cap in df_cap['caption']:
        length.append(len(cap.split(',')))
    df_cap['len'] = np.array(length)
    df_cap = df_cap.sort_values(by=['len', 'img_id'])
    df_cap = df_cap.reset_index(drop=True)

    print("Create training dataset....")
    for i in range(num_files):
        # tfrecord writer: write record into files
        count = 0
        writer = tf.python_io.TFRecordWriter(filename + '-' + str(i + 1) + '.tfrecord')

        # put remaining records in last file
        st = i * num_records_per_file  # start point (inclusive)
        ed = (i + 1) * num_records_per_file if i != num_files - 1 else img_df.shape[0]  # end point (exclusive)
        pbar = tqdm(total=ed - st)
        for idx, row in img_df.iloc[st:ed].iterrows():

            img_representation = row['img']  # img representation in 256-d array format

            # each image has some captions describing it.
            for _, inner_row in df_cap[df_cap['img_id'] == row['img_id']].iterrows():
                caption = eval(inner_row['caption'])  # caption in different sequence length list format

                # construct 'example' object containing 'img', 'caption'
                example = tf.train.Example(features=tf.train.Features(feature={'img': _float_feature(img_representation), 'caption': _int64_feature(caption)}))

                count += 1
                writer.write(example.SerializeToString())
            pbar.update(1)

        print("Create {}-{}.tfrecord -- contains {} records".format(
        filename, str(i + 1), count))
        total_count += count
        writer.close()
    print("Total records: {}".format(total_count))

def main():
    df_cap = pd.read_csv('./dataset/train_cap_vec.csv') # a dataframe - 'img_id', 'caption'
    img_train = cPickle.load(open('./dataset/train_img256.pkl', 'rb')) # a dictionary - keys: 'img_id', values: '256-d array
    # transform img_dict to dataframe
    img_train_df = pd.DataFrame(list(img_train.items()), columns=['img_id', 'img'])
    create_tfrecords(df_cap, img_train_df, './dataset/tfrecord/train', 10)

if __name__ == '__main__':
    main()
