import tensorflow as tf
import numpy as np
import sys
sys.path.append('utils')
from utils import *
from attention import *
from tqdm import tqdm
import _pickle as cPickle
import os
import heapq

def get_hparams():
    dec_map = cPickle.load(open('./dataset/dec_map.pkl', 'rb'))  # id => token
    vocab_size = len(dec_map)
    hparams = tf.contrib.training.HParams(
        vocab_size=vocab_size,
        batch_size=64,
        rnn_units=256,
        image_embedding_size=256,
        word_embedding_size=256,
        drop_keep_prob=0.7,
        lr=1e-3,
        training_epochs=100,
        max_caption_len=15,
        ckpt_dir='model_ckpt/')
    return hparams

class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []

class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.
        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score

class ImageCaptionModel(object):

    def __init__(self, hparams, mode):
        self.hps = hparams
        self.mode = mode

    def _build_inputs(self):
        if self.mode == 'train':
            self.filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
            self.training_iterator, types, shapes = tfrecord_iterator(self.filenames, self.hps.batch_size, record_parser)

            self.handle = tf.placeholder(tf.string, shape=[], name='handle')
            iterator = tf.data.Iterator.from_string_handle(self.handle, types, shapes)
            records = iterator.get_next()

            image_embed = records['img']
            image_embed.set_shape([None, self.hps.image_embedding_size])
            input_seq = records['input_seq']
            target_seq = records['output_seq']
            input_mask = records['mask']

        else:
            image_embed = tf.placeholder(tf.float32, shape=[None, self.hps.image_embedding_size], name='image_embed')
            input_feed = tf.placeholder(tf.int32, shape=[None], name='input_feed')

            input_seq = tf.expand_dims(input_feed, axis=1)
            # in inference step, only use image_embed
            # and input_seq (the first start word)
            target_seq = None
            input_mask = None

        self.image_embed = image_embed
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.input_mask = input_mask

    def _build_seq_embeddings(self):
        with tf.variable_scope('seq_embedding'), tf.device('/cpu:0'):
            embedding_matrix = tf.get_variable(name='embedding_matrix',
                                               shape=[self.hps.vocab_size, self.hps.word_embedding_size],
                                               initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            # [batch_size, padded_length, embedding_size]
            seq_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.input_seq)
        self.seq_embeddings = seq_embeddings

    def _build_model(self):
        # create rnn cell, you can choose different cell,
        # even stack into multi-layer rnn
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hps.rnn_units, state_is_tuple=True)

        # when training, add dropout to regularize.
        if self.mode == 'train':
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=self.hps.drop_keep_prob, output_keep_prob=self.hps.drop_keep_prob)
            # Todo: maybe attention

        # run rnn
        with tf.variable_scope('rnn_scope', initializer=tf.random_uniform_initializer(minval=-1, maxval=1)) as rnn_scope:

            # feed the image embeddings to set the initial rnn state.
            zero_state = rnn_cell.zero_state(batch_size=tf.shape(self.image_embed)[0], dtype=tf.float32)
            _, initial_state = rnn_cell(self.image_embed, zero_state)

            rnn_scope.reuse_variables()

            if self.mode == 'train':
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                               inputs=self.seq_embeddings,
                                               sequence_length=sequence_length,
                                               initial_state=initial_state,
                                               dtype=tf.float32,
                                               scope=rnn_scope)
            else:
                # in inference mode,
                #  use concatenated states for convenient feeding and fetching.
                initial_state = tf.concat(values=initial_state, axis=1, name='initial_state')
                state_feed = tf.placeholder(tf.float32, shape=[None, sum(rnn_cell.state_size)], name='state_feed')
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                # run a single rnn step
                outputs, state = rnn_cell(inputs=tf.squeeze(self.seq_embeddings, axis=[1]), state=state_tuple)

                # concatenate the resulting state.
                final_state = tf.concat(values=state, axis=1, name='final_state')

        # stack rnn output vertically
        # [sequence_len * batch_size, rnn_output_size]
        rnn_outputs = tf.reshape(outputs , [-1, rnn_cell.output_size])

        # get logits after transforming from dense layer
        with tf.variable_scope("logits") as logits_scope:
            rnn_out = {'weights': tf.Variable(tf.random_normal(shape=[self.hps.rnn_units, self.hps.vocab_size],
                                                               mean=0.0,
                                                               stddev=0.1,
                                                               dtype=tf.float32)),
                       'bias':tf.Variable(tf.zeros(shape=[self.hps.vocab_size]))}

            # logits [batch_size*seq_len, vocab_size]
            logits = tf.add(tf.matmul(rnn_outputs, rnn_out['weights']), rnn_out['bias'])

        with tf.name_scope('optimize') as optimize_scope:
            if self.mode == 'train':
                targets = tf.reshape(self.target_seq, [-1])  # flatten to 1-d tensor
                indicator = tf.cast(tf.reshape(self.input_mask, [-1]), tf.float32)

                # loss function
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, indicator)), tf.reduce_sum(indicator), name='batch_loss')

                # add some regularizer or tricks to train well
                self.total_loss = batch_loss

                # save checkpoint
                self.global_step = tf.train.get_or_create_global_step()

                # create optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)

                # compute the gradients of a list of variables
                grads_and_vars = optimizer.compute_gradients(self.total_loss, tf.trainable_variables())
                # grads_and_vars is a list of tuple (gradient, variable)
                # do whatever you need to the 'gradients' part
                clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], 1.0), gv[1]) for gv in grads_and_vars]
                # apply gradient and variables to optimizer
                self.train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

            else:
                pred_softmax = tf.nn.softmax(logits, name='softmax')
                prediction = tf.argmax(pred_softmax, axis=1, name='prediction')

    def build(self):
        self._build_inputs()
        self._build_seq_embeddings()
        self._build_model()

    def train(self, training_filenames, num_train_records):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.hps.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # if checkpoint exists
                saver.restore(sess, ckpt.model_checkpoint_path)
                # assume the name of checkpoint is like '.../model.ckpt-1000'
                gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                sess.run(tf.assign(self.global_step, gs))
            else:
                # no checkpoint
                sess.run(tf.global_variables_initializer())

            training_handle = sess.run(self.training_iterator.string_handle())
            sess.run(self.training_iterator.initializer, feed_dict={self.filenames: training_filenames})

            num_batch_per_epoch_train = num_train_records // self.hps.batch_size

            loss = []
            for epoch in range(self.hps.training_epochs):
                _loss = []
                for i in tqdm(range(num_batch_per_epoch_train)):
                    train_loss_batch, _ = sess.run([self.total_loss, self.train_op], feed_dict={self.handle: training_handle})
                    _loss.append(train_loss_batch)
                    if (i % 1000 == 0):
                        print("minibatch training loss: {:.4f}".format(train_loss_batch))

                loss_this_epoch = np.sum(_loss)
                gs = self.global_step.eval()
                print('Epoch {:2d} - train loss: {:.4f}'.format(int(gs / num_batch_per_epoch_train), loss_this_epoch))
                loss.append(loss_this_epoch)
                if not os.path.exists(self.hps.ckpt_dir):
                    os.makedirs(self.hps.ckpt_dir)
                saver.save(sess, self.hps.ckpt_dir + 'model.ckpt', global_step=gs)
                print("save checkpoint in {}".format(self.hps.ckpt_dir + 'model.ckpt-' + str(gs)))

            print('Done')

    def inference(self, sess, img_embed, enc_map, dec_map):

        # get <start> and <end> word id
        st, ed = enc_map['<BEG>'], enc_map['<END>']

        caption_id = []
        # feed into input_feed
        start_word_feed = [st]

        # feed image_embed into initial state
        initial_state = sess.run(fetches='rnn_scope/initial_state:0', feed_dict={'image_embed:0': img_embed})

        # get the first word and its state
        nxt_word, this_state = sess.run(fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
                                        feed_dict={'input_feed:0': start_word_feed,
                                                   'rnn_scope/state_feed:0': initial_state})

        caption_id.append(int(nxt_word))

        for i in range(self.hps.max_caption_len - 1):
            nxt_word, this_state = sess.run(fetches=['optimize/prediction:0', 'rnn_scope/final_state:0'],
                                            feed_dict={'input_feed:0': nxt_word,
                                                       'rnn_scope/state_feed:0': this_state})
            caption_id.append(int(nxt_word))

        caption = [ dec_map[x] for x in caption_id[:None if ed not in caption_id else caption_id.index(ed)] ]

        return ' '.join(caption)

    def _beam_search(self , sess , img_embed , enc_map , dec_map , beam_size = 3 ):

        # get <start> and <end> word id
        st, ed = enc_map['<BEG>'], enc_map['<END>']

        # feed into input_feed
        start_word_feed = [st]

        # feed image_embed into initial state
        initial_state = sess.run(fetches='rnn_scope/initial_state:0', feed_dict={'image_embed:0': img_embed})


        initial_beam = Caption(sentence=[st], state=initial_state[0], logprob=0.0, score=0.0)

        partial_captions = TopN(beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(beam_size)

        for _ in range(self.hps.max_caption_len - 1):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

#             print("input = ",input_feed.shape)
#             print("state = ",state_feed.shape)
#             print("shape = ",input_feed.shape[0])
#             print(" i = ",input_feed)

            new_state_list , softmax_list = sess.run(fetches=['rnn_scope/final_state:0' , 'optimize/softmax:0'],
                                                     feed_dict={'input_feed:0': input_feed, 'rnn_scope/state_feed:0': state_feed})

            for i , tmp_partial_captions in enumerate(partial_captions_list):
                word_probility = softmax_list[i]

                state = new_state_list[i]
                words_and_probs = list(enumerate(word_probility))
                words_and_probs.sort(key=lambda x: -x[1])
                top_n_words = words_and_probs[0:beam_size]

                for w,p in top_n_words:
                    if p < 1e-12:
                        continue

                    sentence = tmp_partial_captions.sentence + [w]
                    logprob  = tmp_partial_captions.logprob + math.log(p)
                    score = logprob

#                     if metadata:
#                         metadata_list = tmp_partial_captions.metadata + [metadata[i]]
#                     else:
#                         metadata_list = None
                    if w == ed:
                        beam = Caption(sentence, state, logprob, score)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score)
                        partial_captions.push(beam)
            if partial_captions.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        if not complete_captions.size():
            complete_captions = partial_captions


        for i, caption_id in enumerate(complete_captions.extract(sort=True)):
            caption = [ dec_map[x] for x in caption_id.sentence[1:None if ed not in caption_id.sentence else caption_id.sentence.index(ed)] ]
            break

        return ' '.join(caption)
