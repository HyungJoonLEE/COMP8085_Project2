import torch
import torch.nn.functional as F
from yelp_transformer import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import transformers
import json
import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams
from collections import defaultdict
from matplotlib import rc
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, BertTokenizer, BertModel, \
    get_linear_schedule_with_warmup
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
sns.set_palette(sns.color_palette("Paired"))

rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

sentiment_score_map = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}


def read_file(file_name):
    df = pd.read_csv(file_name)
    df['stars'] = df['stars'].astype(int)
    return df


def show_plot(data_field):
    ds_labels = data_field["stars"].value_counts(normalize=True)
    ds_labels.sort_index(inplace=True)
    plt.figure(figsize=(4, 3))
    ax = ds_labels.plot(kind="bar")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Ratio")
    plt.show()


# Make the customer stars starting from 0 instead of 1
def map_sentiment_scores(star_number):
    return sentiment_score_map[int(star_number)]


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.3,
                                         stratify=df['stars'],
                                         random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(test_df, test_size=0.5,
                                       random_state=RANDOM_SEED)
    print(train_df.shape, val_df.shape, test_df.shape)
    return train_df, val_df, test_df


def set_dataset(train_df, val_df, test_df):
    MAX_LEN = 128
    training_data = YelpReview(review=train_df['text'].to_numpy(),
                               target=train_df['stars'].to_numpy(),
                               tokenizer=tokenizer,
                               max_len=MAX_LEN)

    validation_data = YelpReview(review=val_df['text'].to_numpy(),
                                 target=val_df['stars'].to_numpy(),
                                 tokenizer=tokenizer,
                                 max_len=MAX_LEN)

    test_data = YelpReview(review=test_df['text'].to_numpy(),
                           target=test_df['stars'].to_numpy(),
                           tokenizer=tokenizer,
                           max_len=MAX_LEN)
    return training_data, validation_data, test_data


def set_dataloader(training_data, test_data, validation_data):
    BATCH_SIZE = 50
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(validation_data, batch_size=BATCH_SIZE,
                            shuffle=False)
    return train_loader, test_loader, val_loader


def get_sample_batch(data_loader):
    sample_batch = next(iter(data_loader))
    return sample_batch


"""
 Model takes a batch as input and linear layer as (size = 5) as output, 
 to get the final output we must apply softmax at the end.
"""
def set_sentiment_classifier(sample_batch):
    class_names = ['1-Star', '2-Star', '3-Star', '4-Star', '5-Star']
    model = SentimentClassifier(len(class_names))
    model = model.to(device)

    # An evaluation run of the model
    input_ids = sample_batch['input_id'].to(device)
    attention_mask = sample_batch['attention_mask'].to(device)
    F.softmax(model(input_ids, attention_mask), dim=1)
    return model


def set_optimizer(model, data_loader):
    EPOCHS = 10
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    total_steps = len(data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # For multi-class classification usually just use nn.CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss().to(device)

    return EPOCHS, optimizer, total_steps, scheduler, loss_fn


def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """
    Trains the given model using the provided data loader and optimizer.

    Args:
        model (torch.nn.Module): Model to train.
        data_loader (DataLoader): DataLoader providing the training data.
        loss_fn: Loss function to optimize.
        optimizer: Optimizer for updating the model's parameters.
        device: Device to use for training.
        scheduler: Learning rate scheduler.
        n_examples (int): Total number of training examples.

    Returns:
        float: Accuracy of the model on the training data.
        float: Average training loss.

    """
    model=model.train()
    losses = []
    correct_predictions = 0

    """  
    iterate over the batches provided by the data loader. 
    Within each iteration, the batch tensors are moved to the appropriate device. 
    The model performs a forward pass on the input tensors, and the predicted 
    labels are obtained by taking the maximum value along the appropriate dimension. 
    """
    for d in data_loader:
        input_ids = d["input_id"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["target"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets).cpu()

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions/n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_id"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _,preds = torch.max(outputs, dim = 1)

            loss = loss_fn(outputs, targets.detach())
            correct_predictions += torch.sum(preds == targets).cpu()
            losses.append(loss.item())
    return correct_predictions/n_examples, np.mean(losses)


# Model Evaluation on test_data_loader
def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_id"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def process_bert(train_df, val_df, EPOCHS, model,
                 train_loader, val_loader,
                 loss_fn, optimizer, device, scheduler):
    history = defaultdict(list)
    best_accuracy = 0

    df_train_size = len(train_df)
    df_val_size = len(val_df)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/ {EPOCHS}')
        print('-' * 15)
        train_acc, train_loss = train_model(model, train_loader, loss_fn,
                                            optimizer, device, scheduler,
                                            df_train_size)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_loader, loss_fn, device,
                                       df_val_size)
        print(f'Val loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            torch.save(model.state_dict(), 'best_model_state.pkt')
            best_accuracy = val_acc


def get_predictions(model, data_loader):

    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            texts = d["review"]
            input_ids = d["input_id"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["target"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Sentiment')
    plt.xlabel('Predicted Sentiment')


def initialize_bert_model(file_name):
    df = read_file(file_name)
    df['stars'].apply(map_sentiment_scores)
    train_df, val_df, test_df = split_data(df)
    training_data, validation_data, test_data = set_dataset(train_df,
                                                            val_df,
                                                            test_df)
    train_loader, test_loader, val_loader = set_dataloader(training_data,
                                                           test_data,
                                                           validation_data)
    sample_batch = get_sample_batch(train_loader)
    model = set_sentiment_classifier(sample_batch)
    EPOCHS, optimizer, total_steps, scheduler, loss_fn = set_optimizer(model,
                                                                     train_loader)
    process_bert(train_df, val_df, EPOCHS, model,
                 train_loader, val_loader,
                 loss_fn, optimizer, device, scheduler)


def calculate_star_prediction(trained_model_name, file_name):
    df = read_file(file_name)
    df['stars'].apply(map_sentiment_scores)
    test_loader = YelpReview(review=df['text'].to_numpy(),
                             target=df['stars'].to_numpy(),
                             tokenizer=tokenizer,
                             max_len=128)

    class_names = ['1-Star', '2-Star', '3-Star', '4-Star', '5-Star']
    model = SentimentClassifier(len(class_names))
    model.load_state_dict(
        torch.load(trained_model_name, map_location=torch.device('cpu')))
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model.to(device),test_loader)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)
