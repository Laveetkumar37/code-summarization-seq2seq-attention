import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder network: converts input code tokens into hidden representations
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=128, hid_dim=256):
        super().__init__()

        # Embedding layer to convert token IDs into dense vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # LSTM to process embedded input sequence
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        # Embed the input source sequence
        embedded = self.embedding(src)
        # Pass embeddings through LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # Return all encoder outputs and final hidden and cell states
        return outputs, hidden, cell

# Attention mechanism to focus on relevant encoder outputs
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # Linear layer to compute attention energy
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        # Linear layer to produce attention scores
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # Length of the source sequence
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state across source length
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy scores for attention
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Compute attention weights
        attention = self.v(energy).squeeze(2)

        # Normalize attention weights using softmax
        return F.softmax(attention, dim=1)

# Decoder network with attention
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=128, hid_dim=256):
        super().__init__()

        # Embedding layer for target tokens
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # LSTM that takes embedding + context vector as input
        self.lstm = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        # Fully connected layer to predict next token
        self.fc = nn.Linear(hid_dim * 2, output_dim)
        # Attention module
        self.attention = Attention(hid_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        # Add time dimension to input token
        input = input.unsqueeze(1)
        # Embed the input token
        embedded = self.embedding(input)
        # Compute attention weights over encoder outputs
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)

        # Concatenate embedding and context vector
        lstm_input = torch.cat((embedded, context), dim=2)
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # Generate prediction for next token
        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden, cell

# Seq2Seq model combining Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Encoder and decoder networks
        self.encoder = encoder
        self.decoder = decoder
        # Device (CPU or GPU)
        self.device = device

    def forward(self, src, trg):
        # Get batch size and target sequence length
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.fc.out_features
       
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

         # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to decoder is <sos> token
        input = trg[:, 0]
        
        # Decode token by token
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            # Use predicted token as next input (greedy decoding)
            input = output.argmax(1)

        return outputs
