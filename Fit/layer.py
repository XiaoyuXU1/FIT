"""FIT layer attribution and input-gradient importance."""
import math
import torch
from utils import decoder,encode

def select_layers(model, tokenizer, text, ratio):
    layers = decoder(model).layers
    inputs = encode(tokenizer, text, model.device, 128)
    model.eval()
    with torch.no_grad():
        original = model(**inputs, labels=inputs.input_ids).loss.item()
        scores = []
        for index, layer in enumerate(layers):
            saved = [p.detach().clone() for p in layer.parameters()]
            try:
                for p in layer.parameters():
                    p.zero_()
                masked = model(**inputs, labels=inputs.input_ids).loss.item()
                scores.append((index, abs(masked-original)))
            finally:
                for p, old in zip(layer.parameters(), saved):
                    p.copy_(old)
            del saved
    return [i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)
            [:max(1, math.ceil(len(layers)*ratio))]]


def importance(model, tokenizer, text):
    inputs = encode(tokenizer, text, model.device, 4096)
    embeddings = model.get_input_embeddings()(inputs.input_ids).detach().requires_grad_(True)
    model.eval()
    loss = model(inputs_embeds=embeddings, attention_mask=inputs.attention_mask,
                 labels=inputs.input_ids, use_cache=False).loss
    return torch.autograd.grad(loss, embeddings)[0].norm().item()
