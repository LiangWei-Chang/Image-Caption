import _pickle as cPickle
import tensorflow as tf

def load_pickle(data_path):
    data = cPickle.load(open(data_path, 'rb'))
    print('Load %s' % data_path)
    return data

def save_pickle(data, data_path):
    cPickle.dump(data, open(data_path, 'wb'))
    print('Save %s' % data_path)

def record_parser(record):
    ''' parse record from .tfrecord file and create training record

    :args
      record - each record extracted from .tfrecord

    :return
      a dictionary contains {
          'img': image array extracted from vgg16 (256-dim) (Tensor),
          'input_seq': a list of word id
                    which describes input caption sequence (Tensor),
          'output_seq': a list of word id
                    which describes output caption sequence (Tensor),
          'mask': a list of one which describe
                    the length of input caption sequence (Tensor)
      }
    '''

    keys_to_features = {
      "img": tf.FixedLenFeature([256], dtype=tf.float32),
      "caption": tf.VarLenFeature(dtype=tf.int64)
    }

    # features contains - 'img', 'caption'
    features = tf.parse_single_example(record, features=keys_to_features)

    img = features['img']  # tensor
    caption = features['caption'].values  # tensor (features['caption'] - sparse_tensor)
    caption = tf.cast(caption, tf.int32)

    # create input and output sequence for each training example
    # e.g. caption :   [0 2 5 7 9 1]
    #      input_seq:  [0 2 5 7 9]
    #      output_seq: [2 5 7 9 1]
    #      mask:       [1 1 1 1 1]
    caption_len = tf.shape(caption)[0]
    input_len = tf.expand_dims(tf.subtract(caption_len, 1), 0)

    # from [0] fetch input_len of element
    input_seq = tf.slice(caption, [0], input_len)
    # from [1] fetch input_len of element
    output_seq = tf.slice(caption, [1], input_len)
    mask = tf.ones(input_len, dtype=tf.int32)

    records = {
      'img': img,
      'input_seq': input_seq,
      'output_seq': output_seq,
      'mask': mask
    }

    return records

def tfrecord_iterator(filenames, batch_size, record_parser):
    ''' create iterator to eat tfrecord dataset

    :args
        filenames     - a list of filenames (string)
        batch_size    - batch size (positive int)
        record_parser - a parser that read tfrecord
                        and create example record (function)

    :return
        iterator      - an Iterator providing a way
                        to extract elements from the created dataset.
        output_types  - the output types of the created dataset.
        output_shapes - the output shapes of the created dataset.
    '''
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parser, num_parallel_calls=16)

    # padded into equal length in each batch
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={'img': [None],
                                                                         'input_seq': [None],
                                                                         'output_seq': [None],
                                                                         'mask': [None]},
                                                          padding_values={'img': 1.0,       # needless, for completeness
                                                                          'input_seq': 1,   # padding input sequence in this batch
                                                                          'output_seq': 1,  # padding output sequence in this batch
                                                                          'mask': 0})       # padding 0 means no words in this position

    dataset = dataset.repeat()             # repeat dataset infinitely
    dataset = dataset.shuffle(batch_size)  # shuffle the dataset

    iterator = dataset.make_initializable_iterator()
    output_types = dataset.output_types
    output_shapes = dataset.output_shapes

    return iterator, output_types, output_shapes

def get_seq_embeddings(input_seq, vocab_size, word_embedding_size):
    with tf.variable_scope('seq_embedding'), tf.device("/cpu:0"):
        embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[vocab_size, word_embedding_size], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

        # [batch_size, padded_length, embedding_size]
        seq_embeddings = tf.nn.embedding_lookup(embedding_matrix, input_seq)
    return seq_embeddings
