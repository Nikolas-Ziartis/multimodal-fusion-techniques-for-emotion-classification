import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# fix the random seeds so runs are repeatable
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# keep full fp32 unless tf32 is explicitly allowed
def configure_tf32(allow_tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


# count the trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_total(model):
    return sum(p.numel() for p in model.parameters())


# rough estimate of the multiply add operations
def count_macs(model, inputs):
    total = [0]
    hooks = []

    def lin_hook(m, inp, out):
        out_rows = out.numel() // m.out_features
        total[0] += out_rows * m.in_features * m.out_features

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(*inputs)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return int(total[0])


# accuracy and f1 scores from logits
def classification_metrics(logits, true):
    if isinstance(logits, torch.Tensor):
        pred = logits.argmax(1).cpu().numpy()
    else:
        pred = np.asarray(logits).argmax(1)
    true = true.cpu().numpy() if isinstance(true, torch.Tensor) else np.asarray(true)
    true = true.reshape(-1)
    return {
        'Acc': round(float(accuracy_score(true, pred)), 5),
        'F1_macro': round(float(f1_score(true, pred, average='macro', zero_division=0)), 5),
        'F1_weighted': round(float(f1_score(true, pred, average='weighted', zero_division=0)), 5),
    }
