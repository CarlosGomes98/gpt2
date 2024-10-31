from collections import namedtuple
from dataclasses import dataclass
import math
import time
import torch
from torch.utils.data import IterableDataset
import tiktoken
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os

class ShakespeareDataset(IterableDataset):
    def __init__(self, path, *, batch_size, seq_len, world_size, rank):
        with open(path, "r") as f:
            data = f.read()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)

        self.position = batch_size * seq_len * rank
        self.increment = batch_size * seq_len * world_size
    
    def __iter__(self):
        position = self.position
        while position + self.seq_len * self.batch_size + 1 <= len(self.tokens):
            # yield
            tokens = self.tokens[position:position + self.seq_len * self.batch_size + 1]
            x = tokens[:-1].view(self.batch_size, self.seq_len)
            y = tokens[1:].view(self.batch_size, self.seq_len)
            yield x, y
            position += self.increment

##########################
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Attention(nn.Module):
    def __init__(self, *, embed_dim, n_heads, seq_len) -> None:
        super().__init__()
        # self.q = nn.Linear(embed_dim,  embed_dim)
        # self.k = nn.Linear(embed_dim,  embed_dim)
        # self.v = nn.Linear(embed_dim,  embed_dim)
        self.n_embed = embed_dim
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.n_heads = n_heads
        self.internal_dim = embed_dim / self.n_heads
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
    
    def forward(self, x: torch.Tensor):
        B, T, E = x.size()

        # q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        # organize into different heads
        q = rearrange(q, "B T (heads e) -> B heads T e", B=B, T=T, heads=self.n_heads)
        k = rearrange(k, "B T (heads e) -> B heads T e", B=B, T=T, heads=self.n_heads)
        v = rearrange(v, "B T (heads e) -> B heads T e", B=B, T=T, heads=self.n_heads)
        # do attention
        # logits = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # logits = logits.masked_fill(self.bias[..., :T, :T] == 0, -torch.inf)
        # weights = F.softmax(logits, dim=-1)
        # v = weights @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        # B heads T e -> B T heads*e
        v = rearrange(v, "B heads T e -> B T (heads e)", B=B, T=T, heads=self.n_heads)
        return self.c_proj(v)
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = Attention(embed_dim=config.n_embed, n_heads=config.n_head, seq_len=config.block_size)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int
    n_embed: int
    block_size: int
    n_layer: int
    n_head: int

Output = namedtuple("Output", ["logits", "loss"])

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPTModel(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self, idx, labels=None):
        B, T = idx.size()

        tokens = self.transformer.wte(idx)
        pos = self.transformer.wpe(torch.arange(T, device=idx.device, dtype=torch.long))
        x = tokens + pos

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if labels is None:
            return Output(logits, None)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return Output(logits, loss)

GPT_NANO_CONFIG = GPTConfig(vocab_size=50304, n_embed=768, block_size=1024, n_layer=4, n_head=4)


#######

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    ddp = "RANK" in os.environ

    if ddp:
        torch.distributed.init_process_group()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        device = int(rank)
        master_process = rank == 0
    else:
        rank = 0
        world_size = 1
        master_process = True
        device = rank

    torch.manual_seed(42)
    # model = GPTModel.from_pretrained("gpt2")
    cfg = GPT_NANO_CONFIG
    model = GPTModel(cfg).to(device)
    model = torch.compile(model)
    batch_size = 16
    grad_acc_steps = 16
    warmup_steps = 32
    epochs = 10
    max_lr = 3e-4
    min_lr = 0.1*max_lr

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    raw_model = model if not ddp else model.module
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=3e-4, fused=True)
    dataset = ShakespeareDataset("input.txt", batch_size=batch_size, seq_len=cfg.block_size, world_size=world_size, rank=rank)

    max_steps = epochs * (len(dataset.tokens) // (cfg.block_size * batch_size * grad_acc_steps * world_size))
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset: ShakespeareDataset = worker_info.dataset  # the dataset copy in this worker process
        dataset.position = dataset.batch_size * dataset.seq_len * worker_info.num_workers * dataset.rank + dataset.batch_size * dataset.seq_len * worker_info.id
        dataset.increment = worker_info.num_workers * dataset.increment

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, batch_sampler=None, num_workers=4, worker_init_fn=worker_init_fn)
    scaler = torch.cuda.amp.GradScaler()

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    step = 0
    for epoch in range(epochs):
        print(f"==== Epoch {epoch} ====")
        total_loss = 0.0
        for index, (x, y) in enumerate(dataloader):
            if index % grad_acc_steps == 0: # first micro batch
                t0 = time.time()   
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast():
                output, loss = model(x, labels=y)
                loss = loss / grad_acc_steps
                total_loss += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (index + 1) % grad_acc_steps == 0
            scaler.scale(loss).backward()
            
            if (index + 1) % grad_acc_steps == 0: # final micro batch
                if ddp:
                    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # determine and set the learning rate for this iteration
                lr = get_lr(step)
                step += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                scaler.step(optimizer)
                scaler.update()

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.time()
                dt = t1 - t0 # time difference in seconds
                tokens_processed = dataset.batch_size * dataset.seq_len * world_size * grad_acc_steps
                tokens_per_sec = tokens_processed / dt

                print(f"Batch {index // grad_acc_steps} ({grad_acc_steps}) | Loss {total_loss:.4f} | lr {lr:.4e} | norm: {norm:.4f} | dt {dt*1000:.2f} ms | Rate {tokens_per_sec:.2f} tok/sec")
                total_loss = 0.0
                optimizer.zero_grad()
    if ddp:
        torch.distributed.destroy_process_group()
    import sys; sys.exit(0)

    encoder = tiktoken.get_encoding("gpt2")
    # prefix tokens
    model.eval()
    num_return_sequences = 5
    max_length = 30
    tokens = encoder.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # generate! right now x is (B, T) where B = 5, T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x).logits # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = encoder.decode(tokens)
        print(">", decoded)