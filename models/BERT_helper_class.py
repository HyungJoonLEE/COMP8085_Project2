import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import BertModel

bert_model_name = 'bert-base-uncased'


class YelpReview(Dataset):
    def __init__(self, review, target, tokenizer, max_len):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, index):
        review = self.review[index]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=160,
            truncation=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')

        return {'review': review,
                'input_id': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'target': torch.tensor(self.target[index], dtype=torch.long)
                }


# Add one more linear layer on top of the BERT-base model which takes inputs
# from a pooled layer of BERT base and generates 3D output for each input.
# These three outputs will indicate the chance of a review to its respective
# sentiments where First output represents Negative and second and third output
# represents Neutral and positive respectively.
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        temp = self.bert(input_ids,
                         attention_mask)  # Here we have added one linear layer on top of
        pooled_output = temp[1]  # BERT-base with number of output = 3
        out = self.dropout(pooled_output)  #
        out = self.linear(out)
        return out
