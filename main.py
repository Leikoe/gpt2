import torch
from torch import nn
import torch.nn.functional as F
import math

torch.set_float32_matmul_precision('high')

class MultiHeadAttention(nn.Module):
    def __init__(self, max_seq_len: int, n_heads: int, n_embed: int) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads

        self.c_attn = nn.Linear(n_embed, 3 * n_embed) # q, k, v
        self.c_proj = nn.Linear(n_embed, n_embed) # output proj ?

        # 1 0 0
        # 1 1 0
        # 1 1 1
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).reshape(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3 and x.shape[-1] == self.n_embed
        B, L, D = x.shape
        qkv = self.c_attn(x) # (B, L, D*3)
        q, k, v = qkv.split(self.n_embed, 2) # each is (B, L, D)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, L, head_dim)
        k = k.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, L, head_dim)
        v = v.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, L, head_dim)

        scores = q @ k.transpose(-2, -1) # (B, n_heads, L, head_dim) @ (B, n_heads, head_dim, L) = (B, n_heads, L, L)
        scores /= math.sqrt(self.head_dim) # sqrt(d_k) in attention is all you need
        scores = scores.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        scores = scores.to(torch.float32) # softmax at float32
        scores = F.softmax(scores, -1).to(x.dtype) # (B, n_heads, L, L)
        o = scores @ v # (B, n_heads, L, L) @ (B, n_heads, L, head_dim) = (B, n_heads, L, head_dim)
        o = o.transpose(1, 2).reshape(B, L, D) # (B, L, D)
        return self.c_proj(o) # (B, L, D)


class Mlp(nn.Module):
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embed, n_embed * 4)
        self.c_proj = nn.Linear(n_embed * 4, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.c_fc(x), approximate="tanh") # (B, L, n_embed * 4)
        return self.c_proj(h) # (B, L, n_embed)

class Layer(nn.Module):
    def __init__(self, max_seq_len: int, n_heads: int, n_embed: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(max_seq_len, n_heads, n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = Mlp(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x)) # (B, L, D)
        x = x + self.mlp(self.ln_2(x)) # (B, L, D)
        return x

class Gpt2(nn.Module):
    def __init__(self, n_vocab: int, max_seq_len: int, n_layers: int, n_heads: int, n_embed: int):
        super().__init__()

        self.n_vocab = n_vocab
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embed = n_embed

        self.wte = nn.Embedding(n_vocab, n_embed)
        self.wpe = nn.Embedding(max_seq_len, n_embed)
        self.h = nn.ModuleList([Layer(max_seq_len, n_heads, n_embed) for i in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, n_vocab, bias=False) # should end up being tied to wte iirc (but transposed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2 and x.shape[1] <= self.max_seq_len
        B, L = x.shape
        positions = torch.arange(0, L, dtype=torch.long, device=x.device)
        pos_embeddings = self.wpe(positions) # (L, D)
        token_embeddings = self.wte(x) # (L, D)
        h = token_embeddings + pos_embeddings # (L, D)
        for layer in self.h:
            h = layer(h)
        h = self.ln_f(h)
        return self.lm_head(h)


if __name__ == "__main__":
    model = Gpt2(50257, 1024, 12, 12, 768)
    # model.compile()
    sd = model.state_dict()

    # load weights from hf
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    hf_key = lambda k: f"transformer.{k}" if not "lm_head" in k else k
    for k in sd.keys():
        if k.endswith(".mask"): continue
        hf_k = hf_key(k)
        if any([k.endswith(w) for w in transposed]):
            with torch.no_grad():
                hf_sd[hf_k].transpose_(-1, -2)
        assert sd[k].shape == hf_sd[hf_k].shape, f"sd[{k}] = {sd[k].shape} didn't match hf_sd[{hf_k}] = {hf_sd[hf_k].shape}"
        with torch.no_grad():
            sd[k].copy_(hf_sd[hf_k])

    model.to(torch.bfloat16) # comment for fp32
    model.cuda()

    # inference:
    n_streams = 1
    n_new_tokens = 128
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = "Hello, I'm a language model,"
    tokens = tokenizer(prompt, return_tensors='pt')["input_ids"].cuda().repeat(n_streams, 1)
    print("prompt:", tokens)

    import time
    with torch.inference_mode():
        start = time.monotonic()
        for _ in range(n_new_tokens):
            logits = model(tokens) # (B, L, n_vocab)
            logits = logits[:, -1, :] # (B, n_vocab)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            topk_sampled = torch.multinomial(topk_probs, num_samples=1)
            sampled = torch.gather(topk_indices, -1, topk_sampled)
            # print("sampled:", sampled.shape)
            tokens = torch.cat([tokens, sampled], dim=1)
        end = time.monotonic()
    print(f"generated @ {(n_new_tokens * n_streams)/(end-start)}tok/s")

    # print("tokens:", tokens[0].tolist())
    print("-" * 10)
    for b in range(n_streams):
        print(">", tokenizer.decode(tokens[b].tolist()))
