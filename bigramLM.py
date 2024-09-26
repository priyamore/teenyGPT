import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------ hyperparameters ------------
batch_size = 32 # number of sequences for parallel processing
block_size = 8 # context length
max_iters = 3000 # number of steps
l_rate = 1e-2 
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------------------------

torch.manual_seed(1337)

# download the tiny shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Input
# Understand and analyse the input
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# unique chars in the datset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# build lookup tables and tokenizer
stoi_lookup = {char: i for i, char in enumerate(chars)}  # . already in the dataset so no need to add it
itos_lookup = {i: char for i,char in enumerate(chars)}

#tokenizer - convert raw text(strings) to seq of integers
# In practice, sub_words are used for tokenization, so for a sentence we will get only a few tokens
encode = lambda s: [stoi_lookup[char] for char in s]
decode = lambda ix: "".join(itos_lookup[i] for i in ix)


# tokenise the dataset and split it into train and val datasets
tokenised_out = torch.tensor(encode(text), dtype=torch.long)
train_w_end = int(0.9 * len(tokenised_out))
x_tr = tokenised_out[:train_w_end]
x_val = tokenised_out[train_w_end:]

# ------------ data loader ------------
def get_batch(split):
    # minibatch construct
    data = x_tr if split == 'train' else x_val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]) # stack rows
    y = torch.stack([data[i+1: i + (block_size+1)] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
# ---------------------------------------

@torch.no_grad() # dont calculate grads while calulating the loss
def estimate_loss():
    f_out = {}
    model.eval() # set the model mode to eval
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # calculate the train and val losses for the eval iters
        for i in range(eval_iters):
            ix, iy = get_batch(split) # load the batch
            _, loss = model(ix, iy) # calculate the loss
            losses[i] = loss.item()
        f_out[split] = losses.mean() # avg loss for the eval iters
    
    model.train() # set the model mode back to train
    return f_out


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # store the embeddings
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size) # dict of len vocab_size with each value for vector with dim vocab_size 
    
    def forward(self, ix, target=None):
        # forward a mini_batch of size batch_size with each tensor having context as the input 
        # # 1. caculcate the logits
        logits = self.token_embeddings(ix)              # B, T, C - batch, time, channel
    
        if target is not None:
            B, T, C = logits.shape
            # 2. calculate the loss - nll/cross_entropy
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))            # cross_entropy only accepts ((C), (N,C) (N,C)), N - batch_size 
        else:
            loss = None
        return logits, loss

    def generate(self, ix, max_new_tokens):
        # ix is (B, T) i.e (batch, time) => (32, 8)
        for _ in range(max_new_tokens):
            logits, _ = self(ix) # get the targets
            logits = logits[:, -1, :] # what comes next in the sequence? the last char in the time dimension i.e context
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1) # sampling for the next char from the dist
            ix = torch.cat((ix, next_ix), dim=1) # running sampled indices for generating the next chars in the sequence
        
        return ix

bi_model = BigramLM(vocab_size)
model = bi_model.to(device)

# ------------ training ------------
# optimiser for training the BigramML
optimizer = torch.optim.AdamW(bi_model.parameters(), lr=l_rate) # Adam instead of SGD, learning rate = 0.001

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    # forward pass
    logits, loss = model(xb, yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True) # set grad= 0 from the prev steps as usual
    loss.backward()
    # update the model params
    optimizer.step()
# ------------------------------------

# sample/generate from the model
ix = torch.zeros((1, 1), dtype=torch.long, device = device) # start from a newline char(0)
generated_ix = model.generate(ix, max_new_tokens=500)
print('generated text: ', decode(generated_ix[0].tolist()))