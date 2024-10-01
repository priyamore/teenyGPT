import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------ hyperparameters ------------
batch_size = 64 # number of sequences for parallel processing
block_size = 256 # context length
emb_dim = 384 # size of embedding vector, say 32 for a vocabulary of 65
max_iters = 5000 # number of steps
l_rate = 3e-4
eval_interval = 500
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_layers = 6
num_heads = 6
dropout = 0.2
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
    ix = torch.randint(len(data) - block_size, (batch_size,)) # high is set to len(data) - block_size so that we stay within the bounds
    x = torch.stack([data[i: i + block_size] for i in ix]) # stack and sliced rows, shape: (32, 8)
    y = torch.stack([data[i+1: i + (block_size+1)] for i in ix]) # stack for the next token
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

class Head(nn.Module):
    """self-attention with a single head(communication channel)"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias= False)
        self.value = nn.Linear(emb_dim, head_size, bias= False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        # calculate the relevance scores i.e weights
        weights = Q @ K.transpose(-1, -2)  * (C ** -0.5) # scaling factor - square root of the head_size 
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # weighted agg of V
        f_out = weights @ V
        return f_out

class MultiHead(nn.Module):
    """Multiple Self Attentions in Parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        # register Heads as modules
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # concatentae the outputs from multiple attention heads over C
        f_out = torch.cat([head(x) for head in self.heads], dim=-1)
        f_out = self.projection(f_out)
        f_out = self.dropout(f_out)
        return f_out

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
        )
        self.projection = nn.Linear(4 * emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        f_out = self.network(x)
        f_out = self.projection(f_out)
        f_out = self.dropout(f_out)
        return f_out

class Block(nn.Module):
    """ Transformer Block - Multiheaded Attention followed by feed forward"""

    def __init__(self, emb_dim, num_heads):
        super().__init__()
        head_size = emb_dim // num_heads
        # layer norm is applied before the transformation takes place(Unlike the Attention paper)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        # communication
        self.multiheads = MultiHead(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(emb_dim=emb_dim)
        
    def forward(self, x):
        # add skip connections by adding input back to each layer
        x = x + self.multiheads(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x
    

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        # emebedding table mapping each token to a vector
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim) 
        # positional embedding table to capture the postion of each time step
        self.pos_embeddings = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.ModuleList([Block(emb_dim, num_heads) for _ in range(num_layers)])
        self.blocks = nn.Sequential(*self.blocks) # make sure to unpack the list
        self.layer_norm = nn.LayerNorm(emb_dim) # final layer norm
        self.bi_lm_head = nn.Linear(emb_dim, vocab_size) #language model head

    def forward(self, ix, target=None):
        B, T = ix.shape
        token_emb = self.token_embeddings(ix)                           # (B, T, C)
        pos_emb = self.pos_embeddings(torch.arange(T, device=device))   # idx of the each time step results to (T, C) 
        x = token_emb + pos_emb                                         # go to the position by adding pos_emb, results to (B, T, C)
        x = self.blocks(x) # apply self attention
        x = self.layer_norm(x)
        logits = self.bi_lm_head(x) # (B, T, vocab_size)

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
            ix_cond = ix[:, -block_size:] # the ix has to be within the bounds of the time step
            logits, _ = self(ix_cond) # get the targets
            logits = logits[:, -1, :] # what comes next in the sequence? the last char in the time dimension i.e context, (B, C)
            probs = F.softmax(logits, dim=-1) #apply softmax on the C dimension to get the probs for dim embeddings
            next_ix = torch.multinomial(probs, num_samples=1) # sampling for the next char from the dist
            ix = torch.cat((ix, next_ix), dim=1) # add the next token to  input i.e running generatio
            
        return ix

bi_model = BigramLM()
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
    optimizer.zero_grad(set_to_none=True) # set grad= 0 to clear the grads from the prev steps as usual
    loss.backward()
    # update the model params
    optimizer.step()
# ------------------------------------

# sample/generate from the model
ix = torch.zeros((1, 1), dtype=torch.long, device = device) # start from a newline char(0)
generated_ix = model.generate(ix, max_new_tokens=500)
print('generated text: ', decode(generated_ix[0].tolist()))