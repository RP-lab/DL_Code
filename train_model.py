import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from sklearn.metrics import accuracy_score
import logging
from model.LSTMModel import LSTMModel

# Ensure the logs directory exists
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(filename=os.path.join(log_dir, 'train_log.txt'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Load IMDb dataset
train_iter, test_iter = IMDB(split=('train', 'test'))

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define a text preprocessing pipeline
def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return 1 if label == "pos" else 0

# Collate function for DataLoader
def collate_batch(batch):
    text_list, label_list, lengths = [], [], []
    for _label, _text in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(label_pipeline(_label))
        lengths.append(len(processed_text))
    # Padding
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list.to(device), label_list.to(device), lengths.to(device)

# Create DataLoader
batch_size = 64
train_iter, test_iter = IMDB(split=('train', 'test'))
train_dataloader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Initialize model
input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1  # For binary classification (positive/negative)
model = LSTMModel(input_dim, embedding_dim, hidden_dim, output_dim).to(device)

# Set up optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Training loop
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total_batches = 0  # Track the number of batches

    for batch in dataloader:
        text, labels, lengths = batch  # Correct unpacking
        optimizer.zero_grad()

        # Forward pass
        predictions = model(text, lengths).squeeze(1)

        # Compute loss and accuracy
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        total_batches += 1

    # Return average metrics
    return epoch_loss / total_batches, epoch_acc / total_batches

# Binary accuracy function
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

# Training loop
for epoch in range(10):
    try:
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    except Exception as e:
        logging.error(f"Error during training at epoch {epoch+1}: {e}")
        print(f"Error during training at epoch {epoch+1}: {e}")
        break

# Save the trained model
if not os.path.exists('model'):
    os.makedirs('model')
torch.save(model.state_dict(), 'model/sentiment_lstm.pth')
