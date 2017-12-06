import sys
sys.path.append('model')
sys.path.append('utils')
from model import *
from utils import *

def main():
    hparams = get_hparams()
    # rnn_units should be the same with image_embedding_size in our model
    assert (hparams.rnn_units == hparams.image_embedding_size)

    # create model
    model = ImageCaptionModel(hparams, mode='train')
    model.build()

    # start training
    model.train(training_filenames, num_train_records)

if __name__ == '__main__':
    main()
