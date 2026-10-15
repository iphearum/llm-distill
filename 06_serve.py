"""Minimal OpenAI-style chat endpoint for local testing.

Fixes relative to the original:
- Works on transformers 5.x (apply_chat_template returns a dict there; the
  original passed it to generate() as if it were a tensor).
- FastAPI runs plain `def` endpoints in a thread pool, so two requests could
  call generate() on the same model at once. A lock serialises them.
- finish_reason is 'length' when max_tokens is hit, not always 'stop'.
- Optional bearer-token check via API_KEY.

For real traffic, serve the merged checkpoint with vLLM instead
(`vllm serve artifacts/qwen3-khmer-expanded`): it is a standard Qwen3
checkpoint after merging, and vLLM provides batching and streaming.
"""
import os
import threading
from typing import List, Optional

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import chat_input_ids, pick_device, pick_dtype

MODEL = os.getenv('MODEL', 'artifacts/qwen3-khmer-expanded')
API_KEY = os.getenv('API_KEY')
DEVICE = pick_device(os.getenv('DEVICE', 'auto'))
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=pick_dtype(DEVICE), attn_implementation='sdpa').to(DEVICE).eval()
_eos = model.generation_config.eos_token_id
STOP_IDS = {tok.eos_token_id, *(_eos if isinstance(_eos, list) else [_eos])} - {None}
lock = threading.Lock()
app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class Completion(BaseModel):
    model: str = MODEL
    messages: List[Message]
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)


def check_key(authorization: Optional[str]):
    if API_KEY and authorization != f'Bearer {API_KEY}':
        raise HTTPException(401, 'invalid API key')


@app.get('/v1/models')
def models(authorization: Optional[str] = Header(None)):
    check_key(authorization)
    return {'object': 'list', 'data': [{'id': MODEL, 'object': 'model'}]}


@app.post('/v1/chat/completions')
def chat(body: Completion, authorization: Optional[str] = Header(None)):
    check_key(authorization)
    if not body.messages:
        raise HTTPException(400, 'messages required')
    msgs = [m.model_dump() for m in body.messages]
    x = chat_input_ids(tok, msgs, DEVICE)
    sampling = body.temperature > 0
    kwargs = dict(do_sample=sampling,
                  temperature=body.temperature if sampling else None,
                  top_p=body.top_p if sampling else None,
                  top_k=None, pad_token_id=tok.eos_token_id)
    with lock, torch.inference_mode():
        y = model.generate(x, max_new_tokens=body.max_tokens, **kwargs)
    ids = y[0, x.shape[1]:].tolist()
    hit_eos = bool(ids) and ids[-1] in STOP_IDS
    return {
        'object': 'chat.completion',
        'model': MODEL,
        'choices': [{'index': 0,
                     'message': {'role': 'assistant',
                                 'content': tok.decode(ids, skip_special_tokens=True)},
                     'finish_reason': 'stop' if hit_eos else 'length'}],
        'usage': {'prompt_tokens': x.shape[1], 'completion_tokens': len(ids),
                  'total_tokens': x.shape[1] + len(ids)},
    }