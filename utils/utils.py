import _pickle as cPickle

def load_pickle(data_path):
    data = cPickle.load(open(data_path, 'rb'))
    print('Load %s' % data_path)
    return data

def save_pickle(data, data_path):
    cPickle.dump(data, open(data_path, 'wb'))
    print('Save %s' % data_path)
