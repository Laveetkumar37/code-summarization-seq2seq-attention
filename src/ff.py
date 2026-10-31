import pandas as pd

# JSON / JSONL file load
df = pd.read_json("./data/sample_data.json", lines=True)

# CSV me save
df.to_csv("./data/sample_data2.csv", index=False)

print("Converted to CSV")
