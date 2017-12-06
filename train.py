import sys
sys.path.append('model')
sys.path.append('utils')
from model import *
from utils import *

training_filenames = [
    "dataset/tfrecord/train-1.tfrecord", "dataset/tfrecord/train-2.tfrecord",
    "dataset/tfrecord/train-3.tfrecord", "dataset/tfrecord/train-4.tfrecord",
    "dataset/tfrecord/train-5.tfrecord", "dataset/tfrecord/train-6.tfrecord",
    "dataset/tfrecord/train-7.tfrecord", "dataset/tfrecord/train-8.tfrecord",
    "dataset/tfrecord/train-9.tfrecord", "dataset/tfrecord/train-10.tfrecord"]

# get the number of records in training files
def get_num_records(files):
    count = 0
    for fn in files:
        for record in tf.python_io.tf_record_iterator(fn):
            count += 1
    return count

def main():
    hparams = get_hparams()
    # rnn_units should be the same with image_embedding_size in our model
    assert (hparams.rnn_units == hparams.image_embedding_size)

    # create model
    model = ImageCaptionModel(hparams, mode='train')
    model.build()

    num_train_records = get_num_records(training_filenames)
    # start training
    model.train(training_filenames, num_train_records)

if __name__ == '__main__':
    main()
