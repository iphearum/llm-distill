"""Compare the expanded model against the base model on the same text.

The original reported tokens/second for the expanded model alone, which
cannot show a speedup: per-token cost is nearly unchanged (+256 output rows),
so any gain must come from fewer tokens per character, and only if the model
actually emits the new tokens. This version reports, for BOTH models:

- compression (chars per token) on held-out text,
- bits per character (quality; lower is better, comparable across tokenizers),
- characters per second when generating,
- how often the expanded model emits an added token, and whether its output
  is canonical (re-encoding the text gives the same number of tokens). A model
  that keeps spelling words with old sub-tokens gets no speedup.
"""
import argparse
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import bits_per_char, chat_input_ids, pick_device, pick_dtype

p = argparse.ArgumentParser()
p.add_argument('--model', default='artifacts/qwen3-khmer-expanded')
p.add_argument('--base', default='Qwen/Qwen3-0.6B')
p.add_argument('--valid', default='data/valid.jsonl',
               help='Held-out text; use data/self_valid.jsonl after self-distillation')
p.add_argument('--samples', type=int, default=500)
p.add_argument('--bpc-samples', type=int, default=200)
p.add_argument('--bpc', action='store_true')
p.add_argument('--generate', action='store_true')
p.add_argument('--max-new-tokens', type=int, default=256)
p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
a = p.parse_args()

with open(a.valid, encoding='utf-8') as f:
    texts = [json.loads(x)['text'] for x in f]
orig = AutoTokenizer.from_pretrained(a.base, use_fast=True)
new = AutoTokenizer.from_pretrained(a.model, use_fast=True)
old_n = len(orig)

sample = texts[:a.samples]
chars = sum(len(t) for t in sample)
old_tokens = sum(len(orig.encode(t, add_special_tokens=False)) for t in sample)
new_tokens = sum(len(new.encode(t, add_special_tokens=False)) for t in sample)
report = {'compression': {
    'samples': len(sample),
    'chars_per_token_base': round(chars / old_tokens, 3),
    'chars_per_token_expanded': round(chars / new_tokens, 3),
    'token_reduction_pct': round(100 * (1 - new_tokens / old_tokens), 2),
}}
print(json.dumps(report['compression'], indent=2))
if not (a.bpc or a.generate):
    raise SystemExit(0)

device = pick_device(a.device)
dtype = pick_dtype(device)
prompts = ['ពន្យល់អំពីប្រព័ន្ធមូលដ្ឋានទិន្នន័យ។',
           'តើការសិក្សាបច្ចេកវិទ្យាមានសារៈសំខាន់ដូចម្តេច?',
           'សរសេរអត្ថបទខ្លីអំពីប្រវត្តិសាស្ត្រនៃអង្គរវត្ត។']


def run(name, path, tok):
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, attn_implementation='sdpa').to(device).eval()
    out = {}
    if a.bpc:
        out['bits_per_char'] = round(bits_per_char(
            lambda x: model(input_ids=x, use_cache=False).logits,
            tok, texts[:a.bpc_samples], device), 4)
    if a.generate:
        gen_kwargs = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                          pad_token_id=tok.eos_token_id)
        rows = []
        for prompt in prompts:
            x = chat_input_ids(tok, [{'role': 'user', 'content': prompt}], device)
            with torch.inference_mode():
                model.generate(x, max_new_tokens=16, **gen_kwargs)  # warm-up
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                y = model.generate(x, max_new_tokens=a.max_new_tokens, **gen_kwargs)
            if device == 'cuda':
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            ids = y[0, x.shape[1]:].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            rows.append({
                'prompt': prompt,
                'generated_tokens': len(ids),
                'chars': len(text),
                'chars_per_second': round(len(text) / sec, 2),
                'tokens_per_second': round(len(ids) / sec, 2),
                'added_token_share_pct': round(100 * sum(i >= old_n for i in ids) / max(len(ids), 1), 1),
                'reencoded_tokens': len(tok.encode(text, add_special_tokens=False)),
                'answer': text,
            })
        out['generation'] = rows
        out['mean_chars_per_second'] = round(
            sum(r['chars_per_second'] for r in rows) / len(rows), 2)
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    report[name] = out
    print(name, json.dumps({k: v for k, v in out.items() if k != 'generation'}, indent=2))


run('base', a.base, orig)
run('expanded', a.model, new)
if a.bpc:
    b, e = report['base']['bits_per_char'], report['expanded']['bits_per_char']
    print(f'Quality: expanded is {100 * (e / b - 1):+.2f}% bits/char vs base (lower is better).')
if a.generate:
    b, e = report['base']['mean_chars_per_second'], report['expanded']['mean_chars_per_second']
    print(f'Speed: expanded generates {e / b:.2f}x the characters per second of base.')
    print('Note: greedy outputs differ between models, so compare answers too, not only speed.')
with open('artifacts/benchmark.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Wrote artifacts/benchmark.json')