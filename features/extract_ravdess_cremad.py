










































import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import cv2
import librosa
import subprocess
import tempfile
import shutil

import torch
import torch.nn.functional as F
from transformers import WavLMModel, AutoImageProcessor, AutoModel
from tqdm import tqdm



BASE_DIR    = os.environ.get('DISS_BASE', '.')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
CACHE_DIR   = os.path.join(BASE_DIR, 'cache')

os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



T_VIDEO          = 30
FACE_SIZE        = 224
SR_AUDIO         = 16000
MAX_AUDIO_FRAMES = 250

DINOV2_MODEL  = 'facebook/dinov2-base'
WAVLM_MODEL   = 'microsoft/wavlm-base-plus'

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised',
}
CREMAD_EMOTIONS = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad',
}
VIDEO_EXTS = ('.mp4', '.flv', '.avi', '.mov', '.mkv')
HAAR_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'



# locate the ffmpeg binary
def find_ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    return '/usr/bin/ffmpeg'

FFMPEG_EXE = find_ffmpeg()
print(f'ffmpeg: {FFMPEG_EXE}')





print('=' * 70)
print('Loading frozen pretrained backbones')
print('=' * 70)
print(f'  DINOv2: {DINOV2_MODEL}')
print(f'  WavLM:  {WAVLM_MODEL}')
print(f'  Device: {device}')


dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
dinov2 = AutoModel.from_pretrained(DINOV2_MODEL).to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False


wavlm = WavLMModel.from_pretrained(WAVLM_MODEL).to(device).eval()
for p in wavlm.parameters():
    p.requires_grad = False

dinov2_params = sum(p.numel() for p in dinov2.parameters())
wavlm_params  = sum(p.numel() for p in wavlm.parameters())
print(f'  DINOv2 params: {dinov2_params:,} (frozen)')
print(f'  WavLM  params: {wavlm_params:,} (frozen)')


with torch.no_grad():
    dummy_img = torch.randn(1, 3, 224, 224, device=device)
    dummy_aud = torch.randn(1, 16000, device=device)
    dinov2_out = dinov2(pixel_values=dummy_img).last_hidden_state
    wavlm_out  = wavlm(input_values=dummy_aud).last_hidden_state
    print(f'  DINOv2 forward: input (1, 3, 224, 224) -> last_hidden_state {tuple(dinov2_out.shape)}  (CLS = idx 0)')
    print(f'  WavLM forward:  input (1, 16000)       -> last_hidden_state {tuple(wavlm_out.shape)}  (~50 Hz frame rate)')





# finds and crops the largest face in a frame
class FaceCropper:
    def __init__(self, use_mtcnn=False):
        self.haar = cv2.CascadeClassifier(HAAR_PATH)
        self.use_mtcnn = use_mtcnn
        self.mtcnn = None
        if use_mtcnn:
            try:
                from facenet_pytorch import MTCNN as MTCNN_torch
                self.mtcnn = MTCNN_torch(keep_all=True, device=device, post_process=False)
            except Exception as e:
                print(f'  WARN: MTCNN unavailable ({e}), falling back to Haar only')
                self.mtcnn = None

    def crop_largest_face(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)

        box = None
        if self.mtcnn is not None:
            try:


                rgb_for_mtcnn = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb_for_mtcnn)
                if boxes is not None and len(boxes) > 0:
                    bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    box = (max(0, int(bx1)), max(0, int(by1)),
                           int(bx2 - bx1), int(by2 - by1))
            except Exception:
                box = None
        if box is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det  = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(det):
                box = tuple(max(det, key=lambda b: b[2] * b[3]))
        if box is None:
            return np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
        x, y, w, h = box
        y2 = min(frame_bgr.shape[0], y + h)
        x2 = min(frame_bgr.shape[1], x + w)
        face_bgr = frame_bgr[y:y2, x:x2]
        if face_bgr.size == 0:
            return np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
        face_bgr = cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE))
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return face_rgb


# read the video and crop a face from each frame
def extract_face_stack(video_path, cropper):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T_VIDEO
    idxs = np.linspace(0, max(0, total - 1), num=T_VIDEO, dtype=int)
    faces = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:
            faces.append(np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8))
            continue
        faces.append(cropper.crop_largest_face(frame))
    cap.release()
    return np.stack(faces, axis=0)





# pull the audio and resample to 16 khz
def extract_audio_16khz(video_path):
    tmp_wav = tempfile.mktemp(suffix='.wav')
    try:
        cmd = [FFMPEG_EXE, '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
               '-ar', str(SR_AUDIO), '-ac', '1', '-y', tmp_wav]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            return np.zeros(SR_AUDIO, dtype=np.float32)
        y, sr = librosa.load(tmp_wav, sr=SR_AUDIO, mono=True)
        if y.size == 0:
            return np.zeros(SR_AUDIO, dtype=np.float32)
        return y.astype(np.float32)
    except Exception as e:
        print(f'  audio load failed for {video_path}: {e}')
        return np.zeros(SR_AUDIO, dtype=np.float32)
    finally:
        if os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except Exception: pass





@torch.no_grad()
# get video features from the frozen dinov2
def dinov2_forward(face_stack_rgb_uint8):
    pil_imgs = [face_stack_rgb_uint8[i] for i in range(face_stack_rgb_uint8.shape[0])]
    inputs = dinov2_processor(images=pil_imgs, return_tensors='pt').to(device)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
        out = dinov2(**inputs).last_hidden_state
    cls = out[:, 0, :].float().cpu().numpy()
    return cls





@torch.no_grad()
# get audio features from the frozen wavlm
def wavlm_forward(audio_1d):
    audio_t = torch.from_numpy(audio_1d).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
        out = wavlm(input_values=audio_t).last_hidden_state
    return out.squeeze(0).float().cpu().numpy()





# pad or cut the audio features to a fixed length
def pad_audio_feats(audio_feats):
    T = audio_feats.shape[0]
    if T >= MAX_AUDIO_FRAMES:
        return audio_feats[:MAX_AUDIO_FRAMES], MAX_AUDIO_FRAMES
    out = np.zeros((MAX_AUDIO_FRAMES, audio_feats.shape[1]), dtype=audio_feats.dtype)
    out[:T] = audio_feats
    return out, T





# read emotion and actor from a ravdess filename
def parse_ravdess(filename):
    parts = os.path.splitext(filename)[0].split('-')
    if len(parts) < 7:
        return None
    if parts[0] != '01':
        return None
    emo = RAVDESS_EMOTIONS.get(parts[2][:2])
    if emo is None:
        return None
    return {'emotion': emo, 'actor': int(parts[6])}


# read emotion and actor from a crema d filename
def parse_cremad(filename):
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) < 3:
        return None
    emo_code = os.path.splitext(parts[2])[0]
    emo = CREMAD_EMOTIONS.get(emo_code)
    if emo is None:
        return None
    if not parts[0].isdigit():
        return None
    return {'emotion': emo, 'actor': int(parts[0])}


# list the clips and their labels
def collect_clips(root_dir, parser, exts=VIDEO_EXTS):
    records = []
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            if not f.lower().endswith(exts):
                continue
            meta = parser(f)
            if meta is None:
                continue
            records.append({**meta, 'video_path': os.path.join(r, f)})
    return records





# run extraction over one dataset and save the cache
def process_dataset(name, root_dir, parser, use_mtcnn):
    print('\n' + '=' * 70)
    print(f'Processing {name.upper()}')
    print('=' * 70)
    print(f'  Root: {root_dir}')
    print(f'  Face detector: {"MTCNN+Haar" if use_mtcnn else "Haar only"}')

    out_path = os.path.join(CACHE_DIR, f'{name}_pretrained.npz')
    if os.path.exists(out_path):
        print(f'  {out_path} already exists; skipping. Delete to re-extract.')
        return

    records = collect_clips(root_dir, parser)
    N = len(records)
    print(f'  Found {N} clips')
    if N == 0:
        print(f'  WARNING: no clips found at {root_dir}')
        return

    cropper = FaceCropper(use_mtcnn=use_mtcnn)

    video_feats = np.zeros((N, T_VIDEO, 768), dtype=np.float32)
    audio_feats = np.zeros((N, MAX_AUDIO_FRAMES, 768), dtype=np.float32)
    audio_lens  = np.zeros(N, dtype=np.int32)
    labels      = np.empty(N, dtype=object)
    actors      = np.zeros(N, dtype=np.int32)
    blank_face_counts = np.zeros(N, dtype=np.int32)

    t_start = time.time()
    for i, rec in enumerate(tqdm(records, desc=f'{name} clips')):

        face_stack = extract_face_stack(rec['video_path'], cropper)
        blank_face_counts[i] = int((face_stack.sum(axis=(1, 2, 3)) == 0).sum())
        video_feats[i] = dinov2_forward(face_stack)


        audio = extract_audio_16khz(rec['video_path'])
        audio_seq = wavlm_forward(audio)
        a_pad, a_len = pad_audio_feats(audio_seq)
        audio_feats[i] = a_pad
        audio_lens[i]  = a_len

        labels[i] = rec['emotion']
        actors[i] = rec['actor']

    elapsed = (time.time() - t_start) / 60.0
    print(f'  {name} extracted in {elapsed:.1f} min')
    print(f'  blank_face_avg/clip: {blank_face_counts.mean():.2f}/30')

    np.savez_compressed(out_path,
                        video_feats=video_feats,
                        audio_feats=audio_feats,
                        audio_lens=audio_lens,
                        labels=labels.astype(str),
                        actors=actors,
                        blank_face_counts=blank_face_counts)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  Saved {out_path} ({size_mb:.1f} MB)')


    d = np.load(out_path, allow_pickle=True)
    assert d['video_feats'].max() != 0, f'{name}: video_feats appear all-zero'
    assert d['audio_feats'].max() != 0, f'{name}: audio_feats appear all-zero'
    print(f'  Verification: video {d["video_feats"].shape} max {d["video_feats"].max():.3f}')
    print(f'                audio {d["audio_feats"].shape} max {d["audio_feats"].max():.3f} '
          f'lens range [{d["audio_lens"].min()}, {d["audio_lens"].max()}]')





if __name__ == '__main__':
    print('=' * 70)
    print('NB05a: Pretrained-Features Extraction (DINOv2-base + WavLM-base+)')
    print('=' * 70)

    process_dataset(name='ravdess',
                    root_dir=os.path.join(DATASET_DIR, 'RAVDESS'),
                    parser=parse_ravdess,
                    use_mtcnn=False)

    process_dataset(name='cremad',
                    root_dir=os.path.join(DATASET_DIR, 'CREMA-D', 'VideoFlash'),
                    parser=parse_cremad,
                    use_mtcnn=True)

    print('\nNB05a feature extraction fully complete.')
