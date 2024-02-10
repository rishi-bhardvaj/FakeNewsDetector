import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer
import config
from dataset import NewsProducer
from torch.utils.data import DataLoader
from tokenizer import Tokenizer



class SequenceModel(nn.Module):
    def __init__(self):
        super(SequenceModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)
        self.model.classifier = nn.Linear(self.model.bert.pooler.dense.out_features, 1, True)
        self.model.config.num_labels = 1
        self.optimizer = AdamW(self.model.parameters())
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_statistic = None

    def forward(self,input, train=True):
        if not train:
            self.model.eval()
        else:
            self.model.train()

        return self.model.forward(input)

    def backward(self):
        self.loss_statistic.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()

    def loss(self, X, Y):
        y_pred = self.forward(X)
        self.loss_statistic = self.loss_fn(y_pred.logits.view(-1, self.model.config.num_labels), Y.float().view(-1, self.model.config.num_labels))
        return self.loss_statistic



training_dataset = NewsProducer('~/FakeNewsDetector/data', 'Train.csv')
train_loader = DataLoader(training_dataset, batch_size=10, shuffle=True, num_workers=0)
tk_zer = Tokenizer()
seq_model = SequenceModel()
#print((seq_model.model.bert.pooler.dense.out_features))
#print((seq_model.model))
#print(seq_model.model.config.num_labels)

for training_iter in range(config.TRAINING_EPOCHS):
    for batch in train_loader:
        news = list(batch[0])
        labels = list(batch[1])
        assert len(news) == len(labels)
        tokens_batch = tk_zer.tokenizer(news, padding=True, truncation=True, return_tensors="pt")
        tokens_batch["labels"] = torch.tensor(labels)        
        loss = seq_model.loss(tokens_batch['input_ids'], tokens_batch['labels'])
        print(loss.item())
        seq_model.backward()

