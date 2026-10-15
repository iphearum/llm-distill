"""Step 1 of self-distillation: the base model writes its own training text.

No external corpus. For every word in artifacts/candidates.json the base
model is asked, in Khmer, to use that word, so every new token gets examples.
A set of general-topic prompts adds ordinary text. The output is the base
model's own Khmer, which is exactly what the new tokens must reproduce: the
goal is "same model, fewer tokens", not "better model".

Output: data/self_train.jsonl and data/self_valid.jsonl
"""
import argparse
import json
import random
import re
import zlib
from pathlib import Path

import regex
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import pick_device, pick_dtype

WORD_TEMPLATES = [
    'សរសេរកថាខណ្ឌខ្លីមួយដែលប្រើពាក្យ «{w}»។',
    'ពន្យល់អត្ថន័យនៃពាក្យ «{w}» ហើយផ្តល់ឧទាហរណ៍ប្រយោគបីប្រយោគ។',
    'សរសេររឿងខ្លីមួយដែលមានពាក្យ «{w}»។',
]
TOPICS = ['ការអប់រំ', 'សុខភាព', 'កសិកម្ម', 'ប្រវត្តិសាស្ត្រខ្មែរ', 'បច្ចេកវិទ្យា',
          'សេដ្ឋកិច្ច', 'វប្បធម៌ខ្មែរ', 'បរិស្ថាន', 'កីឡា', 'ម្ហូបខ្មែរ', 'ទេសចរណ៍',
          'ច្បាប់', 'វិទ្យាសាស្ត្រ', 'គ្រួសារ', 'ការងារ', 'ភាសាខ្មែរ']
TOPIC_TEMPLATES = ['សរសេរអត្ថបទខ្លីអំពី{t}។', 'ពន្យល់អំពីសារៈសំខាន់នៃ{t}។']

KHMER = regex.compile(r'\p{Script=Khmer}')
THINK = re.compile(r'<think>.*?</think>', re.S)

p = argparse.ArgumentParser()
p.add_argument('--base', default='Qwen/Qwen3-0.6B')
p.add_argument('--per-word', type=int, default=6, help='Generations per candidate word')
p.add_argument('--per-topic', type=int, default=20, help='Generations per topic prompt')
p.add_argument('--max-new-tokens', type=int, default=320)
p.add_argument('--batch-size', type=int, default=32)
p.add_argument('--temperature', type=float, default=0.8)
p.add_argument('--top-p', type=float, default=0.95)
p.add_argument('--min-khmer-ratio', type=float, default=0.6)
p.add_argument('--valid-ratio', type=float, default=0.05)
p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)

words = [x['token'] for x in json.loads(Path('artifacts/candidates.json').read_text(encoding='utf-8'))]
prompts = []
for w in words:
    for i in range(a.per_word):
        prompts.append(WORD_TEMPLATES[i % len(WORD_TEMPLATES)].format(w=w))
for t in TOPICS:
    for tpl in TOPIC_TEMPLATES:
        prompts += [tpl.format(t=t)] * a.per_topic
random.shuffle(prompts)
print(f'{len(prompts)} prompts ({len(words)} words x {a.per_word}, plus topics)')

device = pick_device(a.device)
tok = AutoTokenizer.from_pretrained(a.base, use_fast=True)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    a.base, dtype=pick_dtype(device), attn_implementation='sdpa').to(device).eval()


def good(text):
    if len(text) < 60:
        return False
    non_space = len(text) - text.count(' ') - text.count('\n')
    if len(KHMER.findall(text)) / max(non_space, 1) < a.min_khmer_ratio:
        return False
    raw = text.encode()
    return len(zlib.compress(raw)) / len(raw) > 0.15   # rejects looping output


rows, rejected = [], 0
for i in tqdm(range(0, len(prompts), a.batch_size), desc='Generating'):
    batch = prompts[i:i + a.batch_size]
    chats = [tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
             for q in batch]
    enc = tok(chats, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True,
                             temperature=a.temperature, top_p=a.top_p, top_k=None,
                             pad_token_id=tok.pad_token_id)
    for q, ids in zip(batch, out[:, enc['input_ids'].shape[1]:]):
        text = THINK.sub('', tok.decode(ids, skip_special_tokens=True)).strip()
        if good(text):
            rows.append({'prompt': q, 'text': text})
        else:
            rejected += 1

random.shuffle(rows)
cut = max(1, round(len(rows) * a.valid_ratio))
for name, subset in [('self_valid', rows[:cut]), ('self_train', rows[cut:])]:
    with open(f'data/{name}.jsonl', 'w', encoding='utf-8') as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
covered = sum(any(w in r['text'] for r in rows) for w in words)
print(f'Kept {len(rows)}, rejected {rejected}. '
      f'{covered}/{len(words)} words appear in the kept text.')