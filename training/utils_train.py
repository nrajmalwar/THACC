import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Define the Network
class classifier(nn.Module):

    # Define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):

        #Constructor
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # Dense layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):

        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        #concat the final forward and backward hidden state in case of bidirectional LSTM
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        # Hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function softmax
        # output = F.softmax(dense_outputs[0])

        return dense_outputs

# No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define metric
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    _, predictions = torch.max(preds, 1)

    correct = (predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc

# Define Training loop
def train(model, iterator, optimizer, criterion):

    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # Set the model in training phase
    model.train()

    for batch in iterator:

        # Resets the gradients after every batch
        optimizer.zero_grad()

        # Retrieve text and no. of words
        text, text_length = batch.Subs

        # Convert to 1D tensor
        predictions = model(text, text_length).squeeze()

        # Compute the loss
        loss = criterion(predictions, batch.Type)

        # Compute the binary accuracy
        acc = binary_accuracy(predictions, batch.Type)

        # Backpropage the loss and compute the gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        # Loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Define Evaluating Loop
def evaluate(model, iterator, criterion):

    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # Deactivating dropout layers
    model.eval()

    # Deactivates autograd
    with torch.no_grad():

        for batch in iterator:

            # Retrieve text and no. of words
            text, text_length = batch.Subs

            # Convert to 1d tensor
            predictions = model(text, text_length).squeeze()

            # Compute loss and accuracy
            loss = criterion(predictions, batch.Type)
            acc = binary_accuracy(predictions, batch.Type)

            # Keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# function to output list of predictions and truth labels
def evaluation_metrics(model, iterator):

  preds = []
  y = []

  # model is not training
  model.eval()

  # Deactivates autograd
  with torch.no_grad():
    for batch in iterator:
      # Retrieve text and no. of words
      text, text_length = batch.Subs

      # Convert to 1d tensor
      predictions = model(text, text_length).squeeze()
      _, predictions = torch.max(predictions, 1)

      preds.append(predictions.numpy())
      y.append(batch.Type.numpy())

  # Flatten the lists
  preds = [item for sublist in preds for item in sublist]
  y = [item for sublist in y for item in sublist]

  return preds, y
# function to plot accuracy and loss graphs
def plot_model_history(history):
    # plot the training loss and accuracy
    # print("plotting training accuracy and loss...")
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy
    axs[0].plot(range(1,len(history["train_acc"])+1), history["train_acc"])
    axs[0].plot(range(1,len(history["valid_acc"])+1), history["valid_acc"])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    # axs[0].set_xticks(np.arange(1,len(history["train_acc"])+1), len(history["train_acc"])/10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1,len(history['train_loss'])+1), history['train_loss'])
    axs[1].plot(range(1,len(history['valid_loss'])+1), history['valid_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    # axs[1].set_xticks(np.arange(1,len(history['train_loss'])+1), len(history['train_loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    # plt.show()

    return plt
