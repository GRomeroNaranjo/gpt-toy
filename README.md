# gpt-toy
This repository aims at implementing a GPT2 style architecture, then trained with a fraction of the data (6B tokens). Particularly, using a batch size of 64, 1024 tokens, and updating the model every 8 accumulated gradient steps. Employing the use of HellaSwag as the primary accuracy metric, and using PyTorch's cross-entropy loss function. Furthermore, this model uses a custom DataLoader, and learning rate schedueling system, where this ReadME file will provide information regarading the implementation and technical details, explaining how it can be implemented into a local machine.

# technical-details
The config file of this model as are follows, closely aligning with a GPT-2 small style implementation:
```
@dataclass
class Config():
    n_layers = 12
    n_heads = 12
    n_embed = 768

    vocab_size = 50257
    block_size = 1024
    dropout = 0.2
    head_size = n_embed // n_heads
```

Additionally, using a from-scratch highly-mathematical lr schedule systsem, where the function is as follows:
```
def get_lr(min_lr, max_lr, max_step, current_step):
    lr = min_lr + (0.5 * (max_lr - min_lr) * (1 + math.cos((current_step / max_step) * math.pi)))
    return lr
```

Then this implementation uses the following hyperparameters for the training loop, where gradient accumulation is integrate effectively too (where when it says that val_data calculation is 100, it means 100 steps or updates, not necessarily batches):
```
full_iterations = 1
val_data_calculation = 100
hella_swag_calculation = 200
text_sample = 1000
checkpoint = 200

max_lr = 6e-4
min_lr = 6e-4 * 0.1
total_steps = len(train_x) // gradient_accumulation_steps
```
