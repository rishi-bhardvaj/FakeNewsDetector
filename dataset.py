import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import os


class DatasetBuilder:
    def __init__(self, data_dir, data):
        self.truth = pd.read_csv(os.path.join(data_dir, data[0]), delimiter=',', error_bad_lines=False)
        self.false = pd.read_csv(os.path.join(data_dir, data[1]), delimiter=',', error_bad_lines=False)
        self.__blend()
        self.__write()

    def __blend(self):
        self.truth['Label'] = 1
        self.false['Label'] = 0
        truth_train = self.truth[['text', 'Label']].sample(frac=0.8, random_state=1)
        false_train = self.false[['text', 'Label']].sample(frac=0.8, random_state=1)
        truth_test = self.truth.drop(truth_train.index)
        false_test = self.false.drop(false_train.index)
        self.train = truth_train.append(false_train, ignore_index=True).sample(frac=1, random_state=1).reset_index(
            drop=True)
        self.test = truth_test.append(false_test, ignore_index=True).sample(frac=1, random_state=1).reset_index(
            drop=True)
        del self.truth
        del self.false
        return

    def __write(self):
        f = open('data/Train.csv','w+')
        self.train.to_csv(f, sep=',', header=True, index=False)
        f.close()
        f = open('data/Test.csv', 'w+')
        self.test.to_csv(f, sep=',', header=True, index=False)


class NewsProducer(Dataset):
    def __init__(self, path, filename):
        self.data = pd.read_csv(os.path.join(path, filename), delimiter=',', error_bad_lines=False)

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        news = self.data.loc[item, 'text']
        label = self.data.loc[item, 'Label']
        return (news, label)


class IterableNewsProducer(IterableDataset):
    def __init__(self, path, filename):
        super(IterableNewsProducer).__init__()
        self.data = pd.read_csv(os.path.join(path, filename), delimiter=',', error_bad_lines=False)

    def datastream(self):
        for i in self.data.index:
            news = self.data.loc[i,'text']
            label = self.data.loc[i, 'Label']
            yield from (news, label)
    def __iter__(self):
        return self.datastream()
