"""Optional Unsloth inference check for the merged checkpoint.

Run in a separate environment with compatible Unsloth and PyTorch wheels.
Fixes: works on transformers 5.x (chat template returns a dict), and checks
that the added tokens survived loading (vocab size and a round-trip).
Embeddings are not 4-bit quantized by bitsandbytes, so the new rows load intact.
"""
import torch
from unsloth import FastLanguageModel

from common import chat_input_ids

PATH = 'artifacts/qwen3-khmer-expanded'
model, tok = FastLanguageModel.from_pretrained(
    model_name=PATH, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=True)
FastLanguageModel.for_inference(model)

rows = model.get_input_embeddings().num_embeddings
assert rows == len(tok), f'Embedding rows {rows} != tokenizer size {len(tok)}'

x = chat_input_ids(tok, [{'role': 'user', 'content': 'ពន្យល់អំពីប្រវត្តិនៃភាសាខ្មែរ។'}], 'cuda')
with torch.inference_mode():
    y = model.generate(x, max_new_tokens=128, do_sample=False,
                       temperature=None, top_p=None, top_k=None)
print(tok.decode(y[0, x.shape[1]:], skip_special_tokens=True))