import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import CodeDataset   # Dataset class that loads and preprocesses code and docstrings
from model import Encoder, Decoder, Seq2Seq  # Encoder, Decoder, and Seq2Seq model definitions
import os

# Select GPU (CUDA) if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset from CSV file
dataset = CodeDataset("./data/sample_data2.csv")
# DataLoader to load data in batches and shuffle it
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the encoder with code vocabulary size
encoder = Encoder(len(dataset.code_vocab))

# Initialize the decoder with summary/docstring vocabulary size
decoder = Decoder(len(dataset.sum_vocab))

# Create Seq2Seq model using encoder and decoder
model = Seq2Seq(encoder, decoder, device).to(device)

# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer to update model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop for 45 epochs
for epoch in range(45):
    total_loss = 0
    print("epoch start..............!")
    # Iterate over each batch
    for code, docstring in loader:
        # Move code tensor to selected device (CPU/GPU)
        code = code.to(device)
        print("Batch running.....!")
        # Move docstring tensor to selected device
        docstring = docstring.to(device)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Forward pass
        # docstring[:, :-1] is used as input (last token excluded)
        output = model(code, docstring[:, :-1])

        # Reshape output for loss computation
        output = output.reshape(-1, output.shape[-1])

        # Target is shifted docstring (next token prediction)
        target = docstring[:, 1:].reshape(-1)

        # Compute loss
        loss = criterion(output, target)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate batch loss
        total_loss += loss.item()

    # Print loss after each epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Create models directory if it does not exist
os.makedirs("models", exist_ok=True)

# Save trained model weights
torch.save(model.state_dict(), "./models/model.pth")
print("Model Saved")
