import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LEN_A_CAP = 250
VIDEO_FRAMES = 30
FEATURE_DIM = 768


# load the cached features and labels for a dataset
def load_npz(cache_dir, dataset):
    path = os.path.join(cache_dir, f"{dataset}_pretrained.npz")
    z = np.load(path, allow_pickle=True)
    need = {'video_feats', 'audio_feats', 'labels'}
    missing = need - set(z.files)
    if missing:
        raise KeyError(
            f"{os.path.basename(path)} is missing {sorted(missing)}; it actually "
            f"contains {list(z.files)}. Edit load_npz to map your keys.")
    out = {'video_feats': z['video_feats'], 'audio_feats': z['audio_feats'],
           'labels': z['labels']}
    if 'audio_lens' in z.files:
        out['audio_lens'] = z['audio_lens']
    if 'actors' in z.files:
        out['actors'] = z['actors']
    return out


def load_label_classes(cache_dir, dataset):
    path = os.path.join(cache_dir, f"{dataset}_label_encoder.joblib")
    if not os.path.exists(path):
        return None
    import joblib
    le = joblib.load(path)
    classes = getattr(le, 'classes_', le)
    return [str(c) for c in classes]


def load_folds(cache_dir, dataset):
    with open(os.path.join(cache_dir, f"{dataset}_folds.pkl"), 'rb') as f:
        return pickle.load(f)


def encode_labels(labels):
    classes = np.unique(labels)
    ints = np.searchsorted(classes, labels)
    return ints.astype(np.int64), classes


class VADataset(Dataset):
    def __init__(self, data, indices, labels_int):
        self.video = data['video_feats']
        self.audio = data['audio_feats']
        self.audio_lens = data.get('audio_lens')
        self.labels = labels_int
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        v = torch.as_tensor(np.asarray(self.video[i]), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(self.audio[i]), dtype=torch.float32)
        if a.dim() == 1:
            a = a.unsqueeze(0)


        if self.audio_lens is not None and a.size(0) == LEN_A_CAP:
            a_len = int(self.audio_lens[i])
        else:
            a_len = a.size(0)
        a_len = int(np.clip(a_len, 1, LEN_A_CAP))
        return {'vision': v, 'audio': a, 'audio_len': a_len,
                'label': int(self.labels[i]), 'orig_index': i}


def va_collate(batch):
    B = len(batch)
    vision = torch.stack([b['vision'] for b in batch])
    audio = torch.zeros(B, LEN_A_CAP, FEATURE_DIM)
    audio_len = torch.empty(B, dtype=torch.long)
    for k, b in enumerate(batch):
        a = b['audio']
        T = min(a.size(0), LEN_A_CAP)
        audio[k, :T] = a[:T]
        audio_len[k] = min(b['audio_len'], T)
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    orig_index = torch.tensor([b['orig_index'] for b in batch], dtype=torch.long)
    return {'vision': vision, 'audio': audio, 'audio_len': audio_len,
            'label': labels, 'orig_index': orig_index}


def make_dataloaders(data, train_idx, val_idx, test_idx, labels_int,
                     batch_size=64, num_workers=2):
    N = len(labels_int)
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        idx = np.asarray(idx)
        assert idx.size == 0 or (idx.min() >= 0 and idx.max() < N), \
            f"{name} fold indices do not address this cache (n={N}, max idx={idx.max()})"

    def _dl(idx, train):
        ds = VADataset(data, idx, labels_int)
        return DataLoader(ds, batch_size=batch_size, shuffle=train,
                          drop_last=False, num_workers=num_workers,
                          collate_fn=va_collate)

    return {'train': _dl(train_idx, True),
            'valid': _dl(val_idx, False),
            'test': _dl(test_idx, False)}
