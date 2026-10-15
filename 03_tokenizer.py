"""Add the discovered words to the base tokenizer and measure the real effect.

Changes from the original:
- Records the model's embedding row count and whether input/output
  embeddings are tied (Qwen3-0.6B/1.7B/4B tie them; this decides how 04 must
  train and merge).
- Records len(tokenizer) separately from the embedding row count. For Qwen3
  len(tok) is 151,669 but the checkpoint has 151,936 rows, so the first
  added IDs land on existing padding rows.
- Replaces the 7 hard-coded samples with a measurement on held-out data:
  token reduction and the share of added-token matches that cut a cluster.
"""
import argparse
import json
from itertools import islice
from pathlib import Path

from transformers import AddedToken, AutoConfig, AutoTokenizer

from common import splits_cluster

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='Qwen/Qwen3-0.6B')
ap.add_argument('--eval-rows', type=int, default=500)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
original = AutoTokenizer.from_pretrained(args.base, use_fast=True)
cfg = AutoConfig.from_pretrained(args.base)
old_n = len(tok)

items = json.loads(Path('artifacts/candidates.json').read_text(encoding='utf-8'))
new = [AddedToken(x['token'], normalized=False, single_word=False, special=False)
       for x in items]
added = tok.add_tokens(new)
if added != len(items):
    print(f'Warning: {len(items) - added} candidates were already in the vocabulary.')
new_n = len(tok)

Path('artifacts/tokenizer').mkdir(parents=True, exist_ok=True)
tok.save_pretrained('artifacts/tokenizer')

with open('data/valid.jsonl', encoding='utf-8') as f:
    texts = [json.loads(line)['text'] for line in islice(f, args.eval_rows)]
old_total = new_total = new_hits = new_bad = chars = 0
for text in texts:
    old_total += len(original.encode(text, add_special_tokens=False))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    new_total += len(enc['input_ids'])
    chars += len(text)
    for i, (s, e) in zip(enc['input_ids'], enc['offset_mapping']):
        if i >= old_n:
            new_hits += 1
            new_bad += splits_cluster(text, s, e)

meta = {
    'base': args.base,
    'old_vocab_size': old_n,
    'new_vocab_size': new_n,
    'added': added,
    'model_embedding_rows': cfg.vocab_size,
    'tie_word_embeddings': bool(getattr(cfg, 'tie_word_embeddings', False)),
    'valid_eval': {
        'rows': len(texts),
        'chars_per_token_old': round(chars / max(old_total, 1), 3),
        'chars_per_token_new': round(chars / max(new_total, 1), 3),
        'token_reduction_pct': round(100 * (1 - new_total / max(old_total, 1)), 2),
        'added_token_share_of_tokens_pct': round(100 * new_hits / max(new_total, 1), 2),
        'added_token_cluster_split_pct': round(100 * new_bad / max(new_hits, 1), 2),
    },
}
Path('artifacts/vocab_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(json.dumps(meta, indent=2))
if new_n > cfg.vocab_size:
    print(f'Note: {new_n} tokens exceed the {cfg.vocab_size} embedding rows; 04 will grow the matrix.')