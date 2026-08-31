import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# fix the random seeds so runs are repeatable
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# count the trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
