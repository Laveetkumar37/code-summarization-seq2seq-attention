def tokenize(text):
    # Convert text to lowercase to ensure consistency
    text = text.lower()

    # Replace brackets with spaces to separate tokens
    text = text.replace("(", " ").replace(")", " ")

    # Replace punctuation with spaces
    text = text.replace(":", " ").replace(",", " ")

    # Split the text into tokens based on whitespace
    return text.split()
