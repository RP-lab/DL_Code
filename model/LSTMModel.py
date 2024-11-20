import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, text_lengths):
        # Embed the input text
        embedded = self.embedding(text)
        # Pack the sequence, move lengths to CPU as required by pack_padded_sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # Use the last hidden state
        hidden = self.dropout(hidden[-1])
        # Pass through fully connected layer
        return self.fc(hidden)
