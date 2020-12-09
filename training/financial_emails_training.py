# python training/financial_emails_training.py --input data/train_data/type --model_weights weights/level1/model.pt --tokenizer weights/level1/tokenizer.pkl --label_dict weights/level1/label_dict.pkl --plot plots/plot.png

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fnmatch, os, argparse, pickle, csv
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix,classification_report
import torch
from torchtext import data
from torchtext.data.utils import get_tokenizer
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import spacy
from utils_train import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--input", required=True,
	help="path to input dataset of emails")

ap.add_argument("-m", "--model_weights", required=True,
	help="path to output trained model weights")

ap.add_argument("-t", "--tokenizer", required=True,
	help="path to output tokenizer")

ap.add_argument("-l", "--label_dict", required=True,
  help="path to output label dictionary")

ap.add_argument("-p", "--plot", required=True,
  help="path to output model history plot")

args = vars(ap.parse_args())

# Provide the directory/folder which contains all your excel (.xlsx) files
# Get all the excel files path in a list
# Concat all dataframes to single dataframe and convert them to list of dataframes with new column Subs
# Remove emails with just urls and extract docs from html files
# Clean messages to get a single line message body

dir = args["input"]
email_data = pd.concat([pd.read_excel(f'{dir}/{f}', usecols = ['Date', 'From Address', 'Subject', 'Message', 'Type']) \
						for f in os.listdir(dir) if f.endswith('.xlsx')], ignore_index=True)

email_data.dropna(inplace=True)
email_data.reset_index(drop=True, inplace=True)
email_data['Date']= pd.to_datetime(email_data.Date, dayfirst = True)
email_data['Subs'] = email_data['From Address'].map(str) + ' ' + email_data['Subject'].map(str)
email_data['Message'] = email_data['Message'].str.replace('http\S+|www.\S+', '', case=False)
email_data['Message'] = email_data['Message'].apply(lambda x: BeautifulSoup(x, "lxml").text)
email_data.at[:,'Message'] = email_data['Message'].str.replace('\r\n', ' ')
email_data.at[:,'Message'] = email_data['Message'].str.replace('\n', ' ')
email_data.at[:,'Message'] = email_data['Message'].str.replace('\t', ' ')
email_data.dropna(inplace=True)
email_data.reset_index(drop=True, inplace=True)

print("Printing the head of the data...")
# Print email head
print(email_data.head())

# Define the field variables
TEXT = data.Field(tokenize="spacy", batch_first=True, include_lengths=True)
LABEL = data.Field(tokenize="spacy", sequential=False, batch_first=True, is_target=True, unk_token=None)

# Define a list to Read the columns as fields
fields = [("Subs", TEXT), ("Type", LABEL)]

# Custom class to read the dataset from a pandas dataframe
class DataFrameDataset(data.Dataset):
    def __init__(self, df, fields, **kwargs):
        examples = []
        # Read each row in the dataframe
        for i, row in df.iterrows():
            label = row.Type
            text = row.Subs
            # Apply field variables and append each example
            examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

# Convert the input email data to pytorch Dataset
training_data = DataFrameDataset(email_data, fields)

# Build a vocabulary for the entire dataset
TEXT.build_vocab(training_data)
LABEL.build_vocab(training_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:",len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:",len(LABEL.vocab))

# print label dictionary
print(LABEL.vocab.stoi)

# Commonly used words
print(TEXT.vocab.freqs.most_common(10))

# Word dictionary
# print(TEXT.vocab.stoi)

# Save tokenizer dictionary in pickle format
tokenizer = open(args["tokenizer"], 'wb')
pickle.dump(TEXT.vocab.stoi, tokenizer, protocol=pickle.HIGHEST_PROTOCOL)

# Save labels dictionary in pickle format
labels_dict = open(args["label_dict"], 'wb')
pickle.dump(LABEL.vocab.stoi, labels_dict, protocol=pickle.HIGHEST_PROTOCOL)

#Reproducing same results
SEED = 2020

#Torch
torch.manual_seed(SEED)

# Split the data into train and test set
train_data, valid_data = training_data.split(split_ratio=0.2, random_state = random.seed(SEED))

# Set the devide to cpu
device = torch.device('cpu')
# Set batch size
BATCH_SIZE = 64

# Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.Subs),
    sort_within_batch=True,
    device = device)

# Define hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = 1024
num_hidden_nodes = 512
num_output_nodes = 3
num_layers = 1
bidirection = False
dropout = 0.2

# Instantiate the model
model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers,
                   bidirectional = False, dropout = dropout)

print(model)

print(f'The model has {count_parameters(model):,} trainable parameters')

# Define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Push to device (Can be used to push to cuda if available)
model = model.to(device)
criterion = criterion.to(device)

# Train and Evalutate the model
N_EPOCHS = 5
best_valid_loss = float('inf')

start = time.time()
model_weights = args["model_weights"]

columns = ["train_loss", "train_acc", "valid_loss", "valid_acc"]
history = pd.DataFrame(columns= columns)

for epoch in range(N_EPOCHS):

    # Train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # Evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    history.loc[epoch] = train_loss, train_acc, valid_loss, valid_acc

    # Save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_weights)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

end = time.time()

print("Model Training Complete!")
print("Model took %0.2f seconds to train"%(end - start))

# # Evaluate the network
# print("evaluating type classifier network...")
# score, acc = model.evaluate(X_valid, Y_valid, batch_size=BATCH_SIZE, verbose=0)
# Y_pred = model.predict_classes(X_valid, batch_size = BATCH_SIZE)
# df_valid = pd.DataFrame({'true': Y_valid.tolist(), 'pred':Y_pred})
# df_valid['true'] = df_valid['true'].apply(lambda x: np.argmax(x))
# print("confusion matrix \n", confusion_matrix(df_valid.true, df_valid.pred))
# print(classification_report(df_valid.true, df_valid.pred))

# Evaluation metrics
preds, y = evaluation_metrics(model, valid_iterator)
print("\nConfusion Matrix \n", confusion_matrix(y, preds))
print("\nClassification Report \n")
print(classification_report(y, preds))

# plot model history
print("Saving model plots...")
plt = plot_model_history(history)
plt.savefig(args["plot"])
