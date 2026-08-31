import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LEN_A_CAP = 250
VIDEO_FRAMES = 30
FEATURE_DIM = 768


# load the cached features and labels
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


# load the actor independent fold splits
def load_folds(cache_dir, dataset):
    with open(os.path.join(cache_dir, f"{dataset}_folds.pkl"), 'rb') as f:
        return pickle.load(f)


# turn label strings into integers
def encode_labels(labels):
    classes = np.unique(labels)
    ints = np.searchsorted(classes, labels)
    return ints.astype(np.int64), classes


# wraps the features so a dataloader can index them
class VADataset(Dataset):
    def __init__(self, data, indices, labels_int):
        self.video = data['video_feats']
        self.audio = data['audio_feats']
        self.labels = labels_int
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        v = torch.as_tensor(np.asarray(self.video[i]), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(self.audio[i]), dtype=torch.float32)
        return {'vision': v, 'audio': a,
                'labels': {'M': torch.tensor(int(self.labels[i]))},
                'orig_index': i}


# pad the audio in a batch to the same length
def va_collate(batch):
    B = len(batch)
    vision = torch.stack([b['vision'] for b in batch])
    audio = torch.zeros(B, LEN_A_CAP, FEATURE_DIM)
    lengths = []
    for k, b in enumerate(batch):
        a = b['audio']
        if a.dim() == 1:
            a = a.unsqueeze(0)
        T = min(a.size(0), LEN_A_CAP)
        audio[k, :T] = a[:T]
        lengths.append(T)
    labels = torch.stack([b['labels']['M'] for b in batch])
    text = torch.zeros(B, 1, 1)
    return {'vision': vision, 'audio': audio, 'text': text,
            'labels': {'M': labels},
            'audio_lengths': torch.tensor(lengths),
            'orig_index': torch.tensor([b['orig_index'] for b in batch])}


# build the train val and test loaders for a fold
def make_dataloaders(data, train_idx, val_idx, test_idx, labels_int,
                     batch_size=16, num_workers=2):
    N = len(labels_int)
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        idx = np.asarray(idx)
        assert idx.size == 0 or (idx.min() >= 0 and idx.max() < N), \
            f"{name} fold indices do not address this cache (n={N}, max idx={idx.max()})"

    def _dl(idx, train):
        ds = VADataset(data, idx, labels_int)
        return DataLoader(ds, batch_size=batch_size, shuffle=train,
                          drop_last=train, num_workers=num_workers,
                          collate_fn=va_collate)

    return {'train': _dl(train_idx, True),
            'valid': _dl(val_idx, False),
            'test': _dl(test_idx, False)}
