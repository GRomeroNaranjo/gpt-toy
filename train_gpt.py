import torch 
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import random
import pickle as pkl
import tiktoken
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.head_size * config.n_heads, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embed // config.n_heads
        self.queries = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.keys = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.values = nn.Linear(config.n_embed, self.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape 

        q = self.queries(x) 
        k = self.keys(x)
        v = self.values(x)

        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) 

        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v  

        return out
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lnf1 = nn.LayerNorm(config.n_embed)
        self.attention = MultiHeadAttention(config)
        self.lnf2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.lnf1(x))
        x = x + self.mlp(self.lnf2(x))
        return x

@dataclass
class Config():
    n_layers = 12
    n_heads = 12
    n_embed = 768

    vocab_size = 50257
    block_size = 1024
    dropout = 0.2
    head_size = n_embed // n_heads


class GPT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.wpe = nn.Embedding(config.block_size, config.n_embed)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.lnf = nn.LayerNorm(config.n_embed)
        self.projection = nn.Linear(config.n_embed, config.vocab_size)

    def forward(self, x, targets):
        B, T = x.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        wpe = self.wpe(pos)
        wte = self.wte(x)
        x = wpe + wte

        print(f"type(self.h): {type(self.h)}") 

        for block in self.h:
            x = block(x)

        x = self.lnf(x)
        logits = self.projection(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            cropped = input_tokens[:, -self.config.block_size:]
            output, _ = self.forward(cropped, None)
            logits = output[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k)
                min_top_value = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_top_value, torch.full_like(logits, float('-inf')), logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_token), dim=1)

        return input_tokens
    
# -----------------------------------------------------------------------------

def get_lr(min_lr, max_lr, max_step, current_step):
    lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos((current_step / max_step) * math.pi)))
    return lr

def validation_calculation(model, val_x, val_y):
    model.eval()
    with torch.no_grad():
        idx = random.randint(0, len(val_x) - 1)
        x = val_x[idx].to(device)
        y = val_y[idx].to(device)
        _, loss = model(x, y)
    return loss.item()

def hella_swag_eval(model, tokenizer, dataset, max_samples=100):
    model.eval()
    correct = 0

    for example in tqdm(dataset.select(range(max_samples))):
        context = example["ctx_a"] + " " + example["ctx_b"]
        choices = example["endings"]
        label = example["label"]

        losses = []
        for choice in choices:
            full_input = context + " " + choice
            tokens = tokenizer.encode(full_input)

            if len(tokens) > model.config.block_size:
                tokens = tokens[:model.config.block_size]

            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            input_ids = tokens_tensor[:, :-1]
            target_ids = tokens_tensor[:, 1:]

            with torch.no_grad():
                _, loss = model(input_ids, target_ids)
            losses.append(loss.item())

        prediction = torch.tensor(losses).argmin().item()
        if prediction == label:
            correct += 1

    accuracy = correct / max_samples
    return accuracy

class CustomLoader():
    def __init__(self, data, B, T):
        self.data = data
        self.B = B
        self.T = T

    def load(self):
        number = len(self.data) - 1
        total = (number // (self.B * self.T)) * (self.B * self.T)

        x = self.data[:total]
        y = self.data[1:total + 1]

        num_batches = total // (self.B * self.T)
        x = x.view(num_batches, self.B, self.T)
        y = y.view(num_batches, self.B, self.T)

        return x, y

class FullLoader():
    def __init__(self, dataset, train_test_split):
        split = int(len(dataset) * train_test_split)
        self.train_data = dataset[:split]
        self.val_data = dataset[split:]

    def load(self, B, T):
        train_x, train_y = CustomLoader(self.train_data, B, T).load()
        val_x, val_y = CustomLoader(self.val_data, B, T).load()

        return train_x, train_y, val_x, val_y
    
data = "datadir"
loader = FullLoader(data, 0.9)
train_x, train_y, val_x, val_y = loader.load(64, 1024)
hellaswag = load_dataset("hellaswag", split="validation")

# -----------------------------------------------------------------------------

train_loss = []
val_loss = []
hella_swag_accuracy = []

check_point_dir = ""
gradient_accumulation_steps = 8

full_iterations = 1
val_data_calculation = 100
hella_swag_calculation = 200
text_sample = 1000
checkpoint = 200

max_lr = 6e-4
min_lr = 6e-4 * 0.1
total_steps = len(train_x) // gradient_accumulation_steps

model = GPT_Model(Config()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()
tokenizer = tiktoken.get_encoding("gpt2")

gradient_accumulation_steps = 8
optimizer.zero_grad()

global_step = 0
for iteration in range(full_iterations):
    for step, (x_batch, y_batch) in enumerate(zip(train_x, train_y)):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        is_last_step = (step == len(train_x) - 1)

        with autocast():
            logits, loss = model(x_batch, y_batch)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0 or is_last_step:
            lr = get_lr(min_lr, max_lr, total_steps, global_step)
            for param_group in optimizer.param_groups():
                param_group["lr"] = lr

            global_step += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"Epoch {iteration+1}/{full_iterations} | Step {global_step}/{total_steps} | Loss {loss.item() * gradient_accumulation_steps:.4f} | lr {lr:.6f}")
            train_loss.append(loss.item() * gradient_accumulation_steps)

        if step % val_data_calculation == 0 and step != 0:
            val = validation_calculation(model, val_x, val_y)
            print(f"Validation Loss {val}")
            val_loss.append(val)

        if step % hella_swag_calculation == 0 and step != 0:
            accuracy = hella_swag_eval(model, tokenizer, hellaswag, max_samples=100)
            print(f"HellaSwag Accuracy: {accuracy:.2%}")
            hella_swag_accuracy.append(accuracy)

        if step % text_sample == 0 and step != 0:
            model.eval()
            input = "Hello I am a language model, "
            encoded_input = tokenizer.encode(input)
            encoded_input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)
            output = model.generate(encoded_input_tensor, 100, temperature=1.0, top_k=50)
            decoded = tokenizer.decode(output[0])
            print(f"[Text Sample @ Step {step}]\n{decoded}")

        if step % checkpoint == 0 and step != 0:
            torch.save(model.state_dict(), f"checkpoint_model/checkpoint_step_{step}.pt")
            print(f"Checkpoint saved at step {step}")
            metric_checkpoint_path = ""
            with open(f"{metric_checkpoint_path}/checkpoint_step_{step}", "wb") as f:
                pkl.dump({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "hella_swag_accuracy": hella_swag_accuracy
                }, f)

        if is_last_step:
            val = validation_calculation(model, val_x, val_y)
            print(f"Validation Loss: {val}")
            val_loss.append(val)

            accuracy = hella_swag_eval(model, tokenizer, hellaswag, max_samples=100)
            print(f"HellaSwag Accuracy: {accuracy:.2%}")
            hella_swag_accuracy.append(accuracy)

            model.eval()
            input = "Hello I am a language model, "
            encoded_input = tokenizer.encode(input)
            encoded_input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)
            output = model.generate(encoded_input_tensor, 100, temperature=1.0, top_k=50)
            decoded = tokenizer.decode(output[0])
            print(f"[Final Text Sample]\n{decoded}")

            torch.save(model.state_dict(), f"checkpoint_dir/checkpoint_step_{step}.pt")

metric_path = ""
with open(metric_path, "wb") as f:
    pkl.dump({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "hella_swag_accuracy": hella_swag_accuracy
    }, f)