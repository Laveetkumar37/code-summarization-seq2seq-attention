import torch
from preprocess import CodeDataset, encode   # Dataset class and encoding function
from model import Encoder, Decoder, Seq2Seq  # Encoder, Decoder, and Seq2Seq model

# Select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (used to get vocabularies)
dataset = CodeDataset("./data/sample_data2.csv")

# Initialize encoder using code vocabulary size
encoder = Encoder(len(dataset.code_vocab))

# Initialize decoder using summary/docstring vocabulary size
decoder = Decoder(len(dataset.sum_vocab))

# Create Seq2Seq model and move it to the selected device
model = Seq2Seq(encoder, decoder, device).to(device)

# Load trained model weights
model.load_state_dict(torch.load("./models/model.pth", map_location=device))

# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Create inverse vocabulary to map token IDs back to words
inv_vocab = {v:k for k,v in dataset.sum_vocab.items()}

# Function to generate summary for a given code snippet
def summarize(code_text):
    # Encode input code and add batch dimension
    src = encode(code_text, dataset.code_vocab).unsqueeze(0).to(device)

    # Initialize target sequence (start with empty input)
    trg = encode("", dataset.sum_vocab).unsqueeze(0).to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        output = model(src, trg)
    
    # Get predicted token IDs by selecting max probability
    tokens = output.argmax(-1).squeeze().tolist()

    # Convert token IDs back to words
    words = [inv_vocab.get(t,"") for t in tokens]

    # Remove special tokens from the output
    clean = [w for w in words if w not in ["<pad>","<sos>","<eos>"]]

    # Print the final generated summary
    print("Summary:", " ".join(clean))

# Example: summarize a simple Python function
summarize("def sum_two(a, b): return a + b")


