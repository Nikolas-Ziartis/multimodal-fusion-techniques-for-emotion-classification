import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# wraps the cached features for the dataloader
class PretrainedFeatureDataset(Dataset):
    def __init__(self, video_feats, audio_feats, labels_int):

        self.video = video_feats
        self.audio = audio_feats
        self.labels = labels_int.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'audio':  torch.from_numpy(self.audio[idx]).float(),
            'vision': torch.from_numpy(self.video[idx]).float(),
            'labels': {'M': torch.tensor(self.labels[idx], dtype=torch.long)},
        }


# load the cached features and labels
def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {
        'video_feats': d['video_feats'],
        'audio_feats': d['audio_feats'],
        'audio_lens':  d['audio_lens'],
        'labels':      d['labels'].astype(str),
        'actors':      d['actors'],
    }


# turn label strings into integers
def encode_labels(labels_str, classes=None):
    le = LabelEncoder()
    if classes is not None:
        le.fit(classes)
    else:
        le.fit(np.unique(labels_str))
    return le.transform(labels_str), le


# build the train val and test loaders for a fold
def make_dataloaders(data, train_idx, val_idx, test_idx, labels_int, batch_size=8, num_workers=2):
    train_ds = PretrainedFeatureDataset(data['video_feats'][train_idx],
                                        data['audio_feats'][train_idx],
                                        labels_int[train_idx])
    val_ds   = PretrainedFeatureDataset(data['video_feats'][val_idx],
                                        data['audio_feats'][val_idx],
                                        labels_int[val_idx])
    test_ds  = PretrainedFeatureDataset(data['video_feats'][test_idx],
                                        data['audio_feats'][test_idx],
                                        labels_int[test_idx])
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=False),
        'valid': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        'test':  DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
