import torch
from transformers import AutoTokenizer
import config
from dataset import IterableNewsProducer, NewsProducer
from torch.utils.data import DataLoader


class Tokenizer:
    def __init__(self):
        super(Tokenizer).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)


if __name__ == "__main__":
    torch.manual_seed(1)
    tk_zer = Tokenizer()
    if config.ITERABLE:
        training_dataset = IterableNewsProducer('~/FakeNewsDetector/data', 'Train.csv')
        train_loader = DataLoader(training_dataset, batch_size=10, num_workers=0)
    else:
        training_dataset = NewsProducer('~/FakeNewsDetector/data', 'Train.csv')
        train_loader = DataLoader(training_dataset, batch_size=10, shuffle=True, num_workers=0)

    if config.ITERABLE:
        for batch in train_loader:
            news = batch[::2]
            labels = batch[1::2]
            assert len(news) == len(labels)
            tokens_batch = tk_zer.tokenizer(news, padding=True, truncation=True, return_tensors="pt")
            tokens_batch["labels"] = torch.tensor(labels)
            print(tokens_batch['input_ids'])
    else:
        for batch in train_loader:
            news = list(batch[0])
            labels = list(batch[1])
            assert len(news) == len(labels)
            tokens_batch = tk_zer.tokenizer(news, padding=True, truncation=True, return_tensors="pt")
            tokens_batch["labels"] = torch.tensor(labels)
            print(type(tokens_batch))
