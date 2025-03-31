import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from main import Gpt2
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

class DataloaderLite():
    def __init__(self, file_path: str, batch_size: int, seq_len: int):
        self.file_path = file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        with open(file_path, "rb") as f:
            file_as_str = f.read().decode()
            self.tokens = tokenizer(file_as_str, return_tensors='pt')["input_ids"][0]
        self.len = len(self.tokens)
        self.pos = 0

    def next(self):
        if (self.pos + self.batch_size * self.seq_len + 1) > self.len:
            self.pos = 0
        batch = self.tokens[self.pos: self.pos + self.batch_size * self.seq_len + 1]
        x = batch[:-1].reshape(self.batch_size, self.seq_len)
        y = batch[1:].reshape(self.batch_size, self.seq_len)
        self.pos += self.batch_size * self.seq_len
        return x, y

if __name__ == "__main__":
    device = "cuda"

    B, T = 64, 256
    train_loader = DataloaderLite("input.txt", B, T)


    model = Gpt2(50257, 256, 12, 12, 768)
    model.to(torch.bfloat16)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(10):
        for batch in range(train_loader.len // (B * T)):
            x, y = train_loader.next()
            x, y = x.to(device), y.to(device)
            # print("-" * 10)
            # print(f"'{tokenizer.decode(x[0].tolist())}'")
            # print(f"'{tokenizer.decode(y[0].tolist())}'")
            preds = model(x)
            loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]), y.reshape(-1))
            print(f"loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # inference:
    n_streams = 1
    n_new_tokens = 128
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = "I shall"
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
