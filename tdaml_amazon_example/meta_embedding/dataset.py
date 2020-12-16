import _pickle as pickle
from keras.preprocessing.sequence \
    import pad_sequences


class Loader(object):
    def __init__(self, path=None, padding_dict=None):

        if path is not None:
            self.data = Loader.read_pkl(path)

            '''
            !!!! only for tencent cvr!!!
            '''
            # self.data['label'] = (self.data['label'] + 1) / 2
        # '''
        # padding is not needed...
        # '''
        if padding_dict is not None:
            self.data = Loader.padding(
                self.data,
                padding_dict)

    @staticmethod
    def read_pkl(path):
        with open(path, "rb") as f:
            t = pickle.load(f)
        return t

    @staticmethod
    def padding(data, pad_dict):
        for col in pad_dict:
            data[col] = pad_sequences(
                data[col],
                maxlen=pad_dict[col]
            ).tolist()
        return data

    def load_all(self, cols, target_col='y'):
        data = self.data[cols]
        label = \
            self.data[target_col] \
                .map(float).values
        return data, label


class BatchLoader(Loader):
    def __init__(self, path: str,
                 padding_dict: dict,
                 batch_size):
        super(BatchLoader, self) \
            .__init__(path, padding_dict)
        self.n_samples = len(self.data)
        self.batch_size = batch_size
        self.n_batch = \
            self.n_samples // self.batch_size
        print('data loaded.')

    def load(self, i, cols, target_col='y'):
        start = i * self.batch_size
        end = (i + 1) * self.batch_size
        end = min([end, len(self.data)])
        data = self.data[start: end][cols]
        label = self.data[start: end][target_col].map(float).values
        return data, label

# test = Loader('../data/test_test.pkl')