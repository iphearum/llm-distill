"""Train embeddings for the added tokens with the base model frozen, then merge.

Fixes relative to the original:

1. Tied embeddings (critical). Qwen3-0.6B/1.7B/4B set tie_word_embeddings=True:
   the input embedding and LM head are one tensor. The original trained two
   separate row sets and wrote both into that tensor, so the output rows
   overwrote the input rows. The saved model used, as input embeddings,
   vectors that were trained only as output vectors. Training and inference
   disagreed. Here a tied model trains ONE shared row set.
2. Measurement. The original printed a noisy batch-of-1 training loss and
   never checked whether the model got worse. Here the base model's
   bits-per-character on held-out text is the target, and the expanded model
   is evaluated on the same text periodically. Per-token loss can't be
   compared across tokenizers; bits per character can.
3. Training budget. 200 steps x 1 sequence x <=192 tokens is about 38k
   tokens, so most new rows got a handful of updates. Here: packed batches,
   warmup + cosine schedule, and the best checkpoint by held-out BPC is kept.
4. No weight decay on the new rows: decay pulls them toward zero, away from
   the scale of the existing embeddings.

What this still cannot do: the transformer layers never saw these tokens. If
the BPC gap to the base model stays large, the next step is continued
pretraining with the layers (or LoRA adapters) unfrozen, not more steps here.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import bits_per_char, pick_device, pick_dtype

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=2000)
p.add_argument('--batch-size', type=int, default=4)
p.add_argument('--grad-accum', type=int, default=2)
p.add_argument('--seq-len', type=int, default=512)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--eval-every', type=int, default=250)
p.add_argument('--eval-rows', type=int, default=200)
p.add_argument('--out-init', choices=['mean', 'first'], default='mean',
               help='Untied models only: init LM-head rows from the mean of the '
                    'sub-token rows, or from the first sub-token row')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', default='artifacts/qwen3-khmer-expanded')
p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
p.add_argument('--cpu-threads', type=int, default=0)
a = p.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

meta = json.loads(Path('artifacts/vocab_meta.json').read_text())
tok = AutoTokenizer.from_pretrained('artifacts/tokenizer', use_fast=True)
original = AutoTokenizer.from_pretrained(meta['base'], use_fast=True)
old_n, new_n = meta['old_vocab_size'], len(tok)
assert new_n > old_n, 'No added tokens; lower the discovery thresholds.'

device = pick_device(a.device)
dtype = pick_dtype(device)
if device == 'cpu' and a.cpu_threads > 0:
    torch.set_num_threads(a.cpu_threads)
print(f"Loading {meta['base']} as {dtype} on {device}.")
model = AutoModelForCausalLM.from_pretrained(
    meta['base'], dtype=dtype, attn_implementation='sdpa', low_cpu_mem_usage=True).to(device)
model.eval()
model.config.use_cache = False
for prm in model.parameters():
    prm.requires_grad_(False)

emb = model.get_input_embeddings()
head = model.get_output_embeddings()
tied = head.weight.data_ptr() == emb.weight.data_ptr()
print(f'Embedding rows: {emb.num_embeddings}, tokenizer: {old_n} -> {new_n}, tied: {tied}')
if old_n > emb.num_embeddings:
    raise SystemExit('Tokenizer and model disagree; rebuild artifacts for the same base.')
backbone = model.model

# ---- initialisation from the old tokenizer's decomposition -----------------
in_init, out_init = [], []
with torch.no_grad():
    for idx in range(old_n, new_n):
        word = tok.convert_ids_to_tokens(idx)
        pieces = original.encode(word, add_special_tokens=False) or [original.eos_token_id]
        ids = torch.tensor(pieces, device=device)
        in_init.append(emb.weight[ids].float().mean(0))
        if not tied:
            src = head.weight[ids[:1]] if a.out_init == 'first' else head.weight[ids]
            out_init.append(src.float().mean(0))
new_in = torch.nn.Parameter(torch.stack(in_init))
new_out = new_in if tied else torch.nn.Parameter(torch.stack(out_init))
params = [new_in] if tied else [new_in, new_out]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)


def lr_lambda(step):
    if step < a.warmup:
        return (step + 1) / a.warmup
    progress = (step - a.warmup) / max(1, a.steps - a.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def embed(x):
    mask = x >= old_n
    e = emb(x.masked_fill(mask, 0))
    if mask.any():
        extra = F.embedding((x - old_n).clamp(min=0), new_in).to(e.dtype)
        e = torch.where(mask.unsqueeze(-1), extra, e)
    return e


def expanded_logits(x):
    h = backbone(inputs_embeds=embed(x), use_cache=False, return_dict=True).last_hidden_state
    # Only the first old_n head rows exist in the merged model (see merge below),
    # so the training softmax matches the inference softmax exactly.
    return torch.cat([F.linear(h, head.weight[:old_n]),
                      F.linear(h, new_out.to(h.dtype))], dim=-1)


# ---- data: one packed token stream -----------------------------------------
def read_texts(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line)['text'] for line in f]


train_texts = read_texts('data/train.jsonl')
valid_texts = read_texts('data/valid.jsonl')[:a.eval_rows]
sep = tok.convert_tokens_to_ids('<|endoftext|>')
if sep is None or sep == tok.unk_token_id:
    sep = tok.eos_token_id
stream = []
for i in tqdm(range(0, len(train_texts), 1000), desc='Tokenizing'):
    for ids in tok(train_texts[i:i + 1000], add_special_tokens=False)['input_ids']:
        stream.extend(ids)
        stream.append(sep)
stream = torch.tensor(stream, dtype=torch.long)
assert int(stream.max()) < new_n
if len(stream) <= a.seq_len + 1:
    raise SystemExit('Not enough training text for one sequence.')
share = (stream >= old_n).float().mean().item()
print(f'Training stream: {len(stream):,} tokens; added tokens are {100 * share:.1f}% of them.')
seen_tokens = a.steps * a.grad_accum * a.batch_size * a.seq_len
print(f'Planned: {seen_tokens:,} tokens ({seen_tokens / len(stream):.2f} epochs).')


def batch():
    starts = torch.randint(0, len(stream) - a.seq_len - 1, (a.batch_size,))
    return torch.stack([stream[s:s + a.seq_len + 1] for s in starts.tolist()]).to(device)


# ---- evaluation ----------------------------------------------------------
history = []
base_bpc = bits_per_char(lambda x: model(input_ids=x, use_cache=False).logits,
                         original, valid_texts, device)
print(f'Base model, base tokenizer: {base_bpc:.4f} bits/char (target)')


def evaluate(step):
    bpc = bits_per_char(expanded_logits, tok, valid_texts, device)
    history.append({'step': step, 'bpc': bpc, 'gap_to_base': bpc - base_bpc})
    tqdm.write(f'step {step}: {bpc:.4f} bits/char (gap {bpc - base_bpc:+.4f})')
    return bpc


best_bpc = evaluate(0)
best = [t.detach().clone().cpu() for t in params]

# ---- training -------------------------------------------------------------
progress = tqdm(range(1, a.steps + 1), desc='Training', unit='step')
for step in progress:
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(a.grad_accum):
        xy = batch()
        logits = expanded_logits(xy[:, :-1]).float()
        loss = F.cross_entropy(logits.reshape(-1, new_n), xy[:, 1:].reshape(-1)) / a.grad_accum
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    post = {'loss': f'{total:.3f}', 'lr': f'{sched.get_last_lr()[0]:.1e}'}
    if device == 'cuda':
        post['gpu_GB'] = f'{torch.cuda.max_memory_allocated() / 1e9:.1f}'
    progress.set_postfix(post)
    if step % a.eval_every == 0 or step == a.steps:
        bpc = evaluate(step)
        if bpc < best_bpc:
            best_bpc = bpc
            best = [t.detach().clone().cpu() for t in params]

print(f'Best expanded: {best_bpc:.4f} bits/char vs base {base_bpc:.4f} '
      f'({100 * (best_bpc / base_bpc - 1):+.2f}%).')
best_in = best[0]
best_out = best[0] if tied else best[1]

# ---- merge into a standard checkpoint -----------------------------------
# resize_token_embeddings(new_n) shrinks or grows the matrix to exactly new_n
# rows. For Qwen3 with 256 added tokens that is a shrink (151,936 -> 151,925),
# dropping the unused padding rows so no stray logits remain at inference.
print('Merging new rows and saving.')
model = model.cpu()
if device == 'cuda':
    torch.cuda.empty_cache()
model.resize_token_embeddings(new_n, mean_resizing=False)
in_w = model.get_input_embeddings().weight
out_w = model.get_output_embeddings().weight
with torch.no_grad():
    in_w[old_n:new_n].copy_(best_in.to(in_w.dtype))
    if tied:
        assert out_w.data_ptr() == in_w.data_ptr(), 'Tie was broken by resizing.'
    else:
        out_w[old_n:new_n].copy_(best_out.to(out_w.dtype))
assert torch.equal(in_w[old_n:new_n], best_in.to(in_w.dtype))
model.config.use_cache = True
model.save_pretrained(a.output, safe_serialization=True, max_shard_size='4GB')
tok.save_pretrained(a.output)
Path(a.output, 'train_report.json').write_text(json.dumps({
    'base_bpc': base_bpc, 'best_bpc': best_bpc, 'tied': tied,
    'added_token_share_of_train_stream': share, 'history': history, 'args': vars(a),
}, indent=2))
print('Saved:', a.output)