import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizer import tokenize  # Tokenization function

# Function to build vocabulary from a list of sentences
def build_vocab(sentences):
    # Initialize vocabulary with special tokens
    vocab = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
    idx = 4

    # Iterate through all sentences
    for sent in sentences:
        # Tokenize each sentence and add new words to vocabulary
        for word in tokenize(str(sent)):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# Function to encode a sentence into a fixed-length tensor
def encode(sentence, vocab, max_len=20):
    # Add start-of-sequence and end-of-sequence tokens
    tokens = ["<sos>"] + tokenize(str(sentence)) + ["<eos>"]

    # Convert tokens to their corresponding vocabulary indices
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]

    # Truncate sequence if it exceeds maximum length
    ids = ids[:max_len]

    # Pad sequence with <pad> tokens to reach max_len
    ids += [vocab["<pad>"]] * (max_len - len(ids))

    # Return encoded sequence as a PyTorch tensor
    return torch.tensor(ids)

# Custom Dataset class for code–summary pairs
class CodeDataset(Dataset):
    def __init__(self, csv_path):
        # Load data from CSV file
        self.data = pd.read_csv(csv_path)

        # Build vocabulary for code and docstrings
        self.code_vocab = build_vocab(self.data["code"])
        self.sum_vocab = build_vocab(self.data["docstring"])

    def __len__(self):
        # Return total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get code and docstring for a given index
        code = self.data.iloc[idx]["code"]
        docstring = self.data.iloc[idx]["docstring"]
        
        # Encode code and docstring into tensors
        return encode(code, self.code_vocab), encode(docstring, self.sum_vocab)
