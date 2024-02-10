import unittest
from datastream import NewsProducer,IterableNewsProducer
from torch.utils.data import DataLoader
import torch
import numpy

#class MyTestCase(unittest.TestCase):
    #def test_something(self):
     #   self.assertEqual(True, False)


if __name__ == '__main__':
    #unittest.main()
    #training_dataset = NewsProducer('/FakeNewsDetector/data, 'Train.csv')

    #torch.manual_seed(1)
    #train_loader = DataLoader(training_dataset, batch_size=10, shuffle=True, num_workers=0)

    training_dataset = IterableNewsProducer('~/FakeNewsDetector/data', 'Train.csv')

    torch.manual_seed(1)
    train_loader = DataLoader(training_dataset, batch_size=6, num_workers=0)
    print(type(train_loader))

    i = 1
    for batch in train_loader:
        if i == 3:
            print(type(batch))
            for news in batch:
                if isinstance(news, numpy.int64):
                    print(news)
                else:
                    print(news[:30])
            break
        i += 1
'''
'''