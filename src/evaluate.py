import torch
from preprocess import CodeDataset, encode  # Dataset and encoding utilities
from model import Encoder, Decoder, Seq2Seq  # Model components
from nltk.translate.bleu_score import sentence_bleu  # BLEU score for evaluation
from rouge_score import rouge_scorer   # ROUGE score for evaluation

# Select GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (used for vocabularies and references)
dataset = CodeDataset("./data/sample_data2.csv")

# Initialize encoder and decoder using dataset vocab sizes
encoder = Encoder(len(dataset.code_vocab))
decoder = Decoder(len(dataset.sum_vocab))

# Create Seq2Seq model and move it to the selected device
model = Seq2Seq(encoder, decoder, device).to(device)

# Load trained model weights
model.load_state_dict(torch.load("./models/model.pth", map_location=device))
# Set model to evaluation mode
model.eval()

# Inverse vocabulary to convert token IDs back to words
inv_vocab = {v: k for k, v in dataset.sum_vocab.items()}

# Function to generate summary using greedy decoding
def generate_summary(code_text, max_len=20):
    # Encode input code and add batch dimension
    src = encode(code_text, dataset.code_vocab).unsqueeze(0).to(device)
    
    # Run encoder to get encoder outputs and initial hidden states
    encoder_outputs, hidden, cell = model.encoder(src)
    # Start decoding with <sos> token
    input_token = torch.tensor([dataset.sum_vocab["<sos>"]]).to(device)

    generated = []
    
    # Generate tokens up to max_len
    for _ in range(max_len):
        # Decode one token at a time
        output, hidden, cell = model.decoder(
            input_token, hidden, cell, encoder_outputs
        )
        # Select token with highest probability
        top1 = output.argmax(1).item()
        # Stop if end-of-sequence token is generated
        if top1 == dataset.sum_vocab["<eos>"]:
            break

        # Store generated token
        generated.append(top1)
        # Use predicted token as next input
        input_token = torch.tensor([top1]).to(device)
    # Convert generated token IDs to words and remove special tokens
    words = [inv_vocab.get(t, "") for t in generated if inv_vocab.get(t, "") not in ["<pad>","<sos>","<eos>"]]
    return " ".join(words)

# List to store BLEU scores for all samples
bleu_scores = []

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# Evaluate on up to 50 samples from the dataset
for i in range(min(50, len(dataset))):
    # Get code and reference docstring
    code = dataset.data.iloc[i]["code"]
    reference = dataset.data.iloc[i]["docstring"]
    
    # Generate predicted summary
    prediction = generate_summary(code)
    
    # Compute BLEU score
    bleu = sentence_bleu([reference.split()], prediction.split())
    bleu_scores.append(bleu)
    
    # Compute ROUGE scores
    rouge_scores = rouge.score(reference, prediction)
   
   # Print evaluation results
    print("Prediction:", prediction)
    print("Reference :", reference)
    print("BLEU:", bleu)
    print("ROUGE:", rouge_scores)
    print("-" * 40)

# Print evaluation results
print("Average BLEU:", sum(bleu_scores) / len(bleu_scores))
