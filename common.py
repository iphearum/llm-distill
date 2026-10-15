"""Shared helpers for the Khmer vocabulary-expansion pipeline.

torch is imported lazily so steps 01-03 run without it.
"""
import math

import regex

COENG = '្'                      # Khmer subscript sign: joins the next consonant to this cluster
MARK = regex.compile(r'\p{M}')
KHMER_WORD = regex.compile(r'^[\p{Script=Khmer}\p{M}]+$')


def splits_cluster(text, start, end):
    """True if the span text[start:end] cuts through a Khmer orthographic cluster.

    A span is unsafe if the character after it is a combining mark or COENG
    (the cluster continues), or if the character before it is COENG (the span
    starts on a subscript consonant).
    """
    if end < len(text):
        nxt = text[end]
        if nxt == COENG or MARK.match(nxt):
            return True
    if start > 0 and text[start - 1] == COENG:
        return True
    return False


def pick_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise SystemExit(f'CUDA requested, but torch {torch.__version__} cannot see a GPU.')
    return requested


def pick_dtype(device):
    import torch
    if device == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def chat_input_ids(tok, messages, device, enable_thinking=False):
    """Chat-template encode that works on transformers 4.x and 5.x.

    In 5.x apply_chat_template returns a BatchEncoding by default, so code that
    treats the result as a tensor (inputs.shape[1], generate(inputs)) breaks.
    """
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  return_tensors='pt', return_dict=True,
                                  enable_thinking=enable_thinking)
    return enc['input_ids'].to(device)


def bits_per_char(logits_fn, tok, texts, device, max_chars=1500, prefix='\n\n'):
    """Bits per character of `texts` under a model.

    Comparing models with different tokenizers by per-token loss is invalid,
    because the token counts differ. Bits per character is the same quantity
    for both. A fixed ASCII prefix (tokenized identically by both tokenizers)
    gives every Khmer token a context, so no token goes unscored.

    logits_fn: LongTensor[1, T] -> FloatTensor[1, T, V]
    """
    import torch
    import torch.nn.functional as F
    prefix_ids = tok.encode(prefix, add_special_tokens=False)
    total_nats, total_chars = 0.0, 0
    with torch.no_grad():
        for text in texts:
            text = text[:max_chars]
            ids = prefix_ids + tok.encode(text, add_special_tokens=False)
            x = torch.tensor(ids, device=device)[None]
            logits = logits_fn(x).float()
            nll = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction='none')
            total_nats += nll[len(prefix_ids) - 1:].sum().item()
            total_chars += len(text)
    return total_nats / math.log(2) / max(total_chars, 1)