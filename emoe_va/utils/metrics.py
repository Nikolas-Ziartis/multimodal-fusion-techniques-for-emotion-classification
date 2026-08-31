import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


# accuracy and f1 scores from predictions
def _classification_metrics(pred, true):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    pred_idx = pred.argmax(axis=1)
    acc = accuracy_score(true, pred_idx)
    f1m = f1_score(true, pred_idx, average='macro', zero_division=0)
    f1w = f1_score(true, pred_idx, average='weighted', zero_division=0)
    return {
        'Acc': float(acc),
        'F1_macro': float(f1m),
        'F1_weighted': float(f1w),


    }


# gives back the right metric function for a dataset
class MetricsTop:
    def __init__(self, train_mode):
        self.train_mode = train_mode

    def getMetics(self, dataset_name):

        return _classification_metrics
