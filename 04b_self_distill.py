"""Step 2 of self-distillation: teach the new word tokens from the model itself.

The teacher and the student are the SAME frozen model; no second copy is
loaded. The teacher reads text with the original tokenizer (a word is split
into several pieces). The student reads the same text with the expanded
tokenizer (the word is one new token). Only the new token rows train.

Two losses:

1. Next-token loss on the base model's own text (data/self_train.jsonl),
   tokenized the new way. Because the text was sampled from the teacher,
   minimising this pushes p(new token) toward the teacher's probability of the
   whole word. This teaches the OUTPUT side: when to emit the word.
2. Alignment loss. After the student reads the new token, its hidden state
   should equal the teacher's hidden state after reading the last piece of
   the same word, in the same sentence. This teaches the INPUT side directly:
   "this one token means exactly what those pieces mean."

Tied models (Qwen3-0.6B/1.7B/4B) share one row set for both sides, so the two
losses pull on the same vectors; --align-weight balances them.
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
p.add_argument('--steps', type=int, default=1500)
p.add_argument('--batch-size', type=int, default=8, help='Texts per optimizer step')
p.add_argument('--max-chars', type=int, default=1200)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--warmup', type=int, default=100)
p.add_argument('--align-weight', type=float, default=1.0)
p.add_argument('--eval-every', type=int, default=250)
p.add_argument('--eval-rows', type=int, default=150)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', default='artifacts/qwen3-khmer-expanded')
p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
a = p.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

meta = json.loads(Path('artifacts/vocab_meta.json').read_text())
tok = AutoTokenizer.from_pretrained('artifacts/tokenizer', use_fast=True)
original = AutoTokenizer.from_pretrained(meta['base'], use_fast=True)
old_n, new_n = meta['old_vocab_size'], len(tok)
assert new_n > old_n, 'No added tokens.'

device = pick_device(a.device)
model = AutoModelForCausalLM.from_pretrained(
    meta['base'], dtype=pick_dtype(device), attn_implementation='sdpa',
    low_cpu_mem_usage=True).to(device).eval()
model.config.use_cache = False
for prm in model.parameters():
    prm.requires_grad_(False)
emb, head, backbone = model.get_input_embeddings(), model.get_output_embeddings(), model.model
tied = head.weight.data_ptr() == emb.weight.data_ptr()
print(f'Tokenizer {old_n} -> {new_n}, embedding rows {emb.num_embeddings}, tied: {tied}')

# Initialise each new row from the mean of its old pieces.
with torch.no_grad():
    pieces = [original.encode(tok.convert_ids_to_tokens(i), add_special_tokens=False)
              for i in range(old_n, new_n)]
    new_in = torch.nn.Parameter(torch.stack(
        [emb.weight[torch.tensor(ps, device=device)].float().mean(0) for ps in pieces]))
    new_out = new_in if tied else torch.nn.Parameter(torch.stack(
        [head.weight[torch.tensor(ps, device=device)].float().mean(0) for ps in pieces]))
params = [new_in] if tied else [new_in, new_out]
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / a.warmup if s < a.warmup else
                                          0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup)))))


def embed(x):
    mask = x >= old_n
    e = emb(x.masked_fill(mask, 0))
    if mask.any():
        e = torch.where(mask.unsqueeze(-1),
                        F.embedding((x - old_n).clamp(min=0), new_in).to(e.dtype), e)
    return e


def student_hidden(x):
    return backbone(inputs_embeds=embed(x), use_cache=False, return_dict=True).last_hidden_state


def student_logits(h):
    return torch.cat([F.linear(h, head.weight[:old_n]),
                      F.linear(h, new_out.to(h.dtype))], dim=-1)


PREFIX = original.encode('\n\n', add_special_tokens=False)   # same ids in both tokenizers


def aligned_pairs(text):
    """Student index of each new token, and teacher index of that word's last piece."""
    s = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    t = original(text, add_special_tokens=False, return_offsets_mapping=True)
    t_off = t['offset_mapping']
    last_piece_ending_at = {}
    for i, (_, end) in enumerate(t_off):
        nxt_start = t_off[i + 1][0] if i + 1 < len(t_off) else len(text)
        if nxt_start >= end:            # a clean boundary in the teacher's split
            last_piece_ending_at[end] = i
    pairs = [(j, last_piece_ending_at[end])
             for j, (i, (_, end)) in enumerate(zip(s['input_ids'], s['offset_mapping']))
             if i >= old_n and end in last_piece_ending_at]
    return s['input_ids'], t['input_ids'], pairs


def sample_loss(text):
    s_ids, t_ids, pairs = aligned_pairs(text[:a.max_chars])
    P = len(PREFIX)
    xs = torch.tensor(PREFIX + s_ids, device=device)[None]
    h_s = student_hidden(xs)
    lm = F.cross_entropy(student_logits(h_s[0, :-1]).float(), xs[0, 1:])
    if not pairs:
        return lm, None
    with torch.no_grad():
        xt = torch.tensor(PREFIX + t_ids, device=device)[None]
        h_t = backbone(input_ids=xt, use_cache=False, return_dict=True).last_hidden_state
    si = torch.tensor([P + j for j, _ in pairs], device=device)
    ti = torch.tensor([P + i for _, i in pairs], device=device)
    align = 1 - F.cosine_similarity(h_s[0, si].float(), h_t[0, ti].float(), dim=-1).mean()
    return lm, align


def read(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line)['text'] for line in f]


train, valid = read('data/self_train.jsonl'), read('data/self_valid.jsonl')[:a.eval_rows]
print(f'{len(train)} self-generated training texts, {len(valid)} held out')

base_bpc = bits_per_char(lambda x: model(input_ids=x, use_cache=False).logits,
                         original, valid, device, max_chars=a.max_chars)
history = []


def evaluate(step):
    bpc = bits_per_char(lambda x: student_logits(student_hidden(x)), tok, valid, device,
                        max_chars=a.max_chars)
    with torch.no_grad():
        aligns = [al.item() for al in (sample_loss(t)[1] for t in valid[:50]) if al is not None]
    rec = {'step': step, 'bpc': bpc, 'gap_to_base_pct': 100 * (bpc / base_bpc - 1),
           'align_cos': 1 - sum(aligns) / max(len(aligns), 1)}
    history.append(rec)
    tqdm.write(f"step {step}: {bpc:.4f} bits/char ({rec['gap_to_base_pct']:+.2f}% vs base), "
               f"hidden-state cosine {rec['align_cos']:.3f}")
    return bpc


print(f'Teacher (base model): {base_bpc:.4f} bits/char')
best_bpc = evaluate(0)
best = [t.detach().clone().cpu() for t in params]
bar = tqdm(range(1, a.steps + 1), desc='Self-distilling')
for step in bar:
    opt.zero_grad(set_to_none=True)
    lm_sum = al_sum = 0.0
    for text in random.sample(train, min(a.batch_size, len(train))):
        lm, align = sample_loss(text)
        loss = lm + (a.align_weight * align if align is not None else 0)
        (loss / a.batch_size).backward()
        lm_sum += lm.item()
        al_sum += align.item() if align is not None else 0
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    bar.set_postfix(lm=f'{lm_sum / a.batch_size:.3f}', align=f'{al_sum / a.batch_size:.3f}')
    if step % a.eval_every == 0 or step == a.steps:
        bpc = evaluate(step)
        if bpc < best_bpc:
            best_bpc, best = bpc, [t.detach().clone().cpu() for t in params]

# Merge exactly as 04_train.py does: resize to new_n rows, write the rows once
# when tied, and check the tie survived.
model = model.cpu()
if device == 'cuda':
    torch.cuda.empty_cache()
model.resize_token_embeddings(new_n, mean_resizing=False)
in_w, out_w = model.get_input_embeddings().weight, model.get_output_embeddings().weight
with torch.no_grad():
    in_w[old_n:new_n].copy_(best[0].to(in_w.dtype))
    if tied:
        assert out_w.data_ptr() == in_w.data_ptr(), 'Tie was broken by resizing.'
    else:
        out_w[old_n:new_n].copy_(best[1].to(out_w.dtype))
model.config.use_cache = True
model.save_pretrained(a.output, safe_serialization=True)
tok.save_pretrained(a.output)
Path(a.output, 'distill_report.json').write_text(json.dumps(
    {'base_bpc': base_bpc, 'best_bpc': best_bpc, 'tied': tied,
     'history': history, 'args': vars(a)}, indent=2))
print(f'Saved {a.output}. Best {best_bpc:.4f} vs teacher {base_bpc:.4f} bits/char.')