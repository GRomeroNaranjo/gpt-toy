from datasets import load_dataset
import torch
import tiktoken
from tqdm import tqdm
import os

print("Loading dataset (going to be 6B tokens)...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:40%]", streaming=True)

tokenizer = tiktoken.get_encoding("gpt2")
eot_token = tokenizer._special_tokens['<|endoftext|>']

def tokenize(example):
    tokens = [eot_token] + tokenizer.encode_ordinary(example['text'])
    return {"tokens": tokens}

print("Tokenizing...")
tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

print("Flattening tokens...")
all_tokens = []
for example in tqdm(tokenized):
    all_tokens.extend(example["tokens"])

tokenized_data = torch.tensor(all_tokens, dtype=torch.long)
print(f"Len of the tokens: {len(tokenized_data):,}")


output_dir = ""
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "fineweb_sample_6B_tokens.pt")
torch.save(tokenized_data, output_file)

print(f"Tokenized data saved to: {output_file}")
