

































import os
import re
import sys
import time
import glob
import argparse
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







BASE_DIR     = os.environ.get('DISS_BASE', '.')
MEAD_RAW_DIR = os.path.join(BASE_DIR, 'mead_data')
CACHE_DIR    = os.path.join(BASE_DIR, 'cache')

MEAD8_NPZ = os.path.join(CACHE_DIR, 'mead8_pretrained.npz')
MEAD6_NPZ = os.path.join(CACHE_DIR, 'mead6_pretrained.npz')

os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



T_VIDEO          = 30
FACE_SIZE        = 224
SR_AUDIO         = 16000
MAX_AUDIO_FRAMES = 250

DINOV2_MODEL  = 'facebook/dinov2-base'
WAVLM_MODEL   = 'microsoft/wavlm-base-plus'



MEAD_EMOTION_MAP = {
    'angry':     'angry',
    'disgusted':   'disgust',
    'contempt':  'contempt',
    'fear':      'fearful',
    'happy':     'happy',
    'sad':       'sad',
    'surprised': 'surprised',
    'neutral':   'neutral',
}
MEAD_ALL_EMOTIONS = tuple(MEAD_EMOTION_MAP.keys())



CREMA6_LABELS = ('angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad')


FRONTAL_VIEW_TOKENS = ('front',)

VIDEO_EXTS = ('.mp4', '.flv', '.avi', '.mov', '.mkv')
AUDIO_EXTS = ('.m4a', '.wav', '.aac', '.mp3', '.flac')
HAAR_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_LEVEL_RE  = re.compile(r'(level[_]?\d+)', re.IGNORECASE)


# locate the ffmpeg binary
def find_ffmpeg():
    return shutil.which('ffmpeg') or '/usr/bin/ffmpeg'

FFMPEG_EXE = find_ffmpeg()





def _components(path):
    return [c.lower() for c in os.path.normpath(path).split(os.sep) if c]

def _emotion_from_path(path):
    comps = set(_components(path))
    for emo in MEAD_ALL_EMOTIONS:
        if emo in comps:
            return emo
    return None

def _level_from_path(path):
    m = _LEVEL_RE.search(path)
    return (m.group(1).lower() if m else 'level_1')

def _is_frontal(path):
    return any(tok in set(_components(path)) for tok in FRONTAL_VIEW_TOKENS)

def _actor_token(path, root):
    return os.path.relpath(path, root).split(os.sep)[0]

def _utt_stem(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


# find the mead clips we want and read their labels
def collect_mead_clips(root_dir):
    audio_index = {}
    for path in glob.iglob(os.path.join(root_dir, '**', '*'), recursive=True):
        if not path.lower().endswith(AUDIO_EXTS):
            continue
        emo = _emotion_from_path(path)
        if emo is None:
            continue
        audio_index.setdefault(
            (_actor_token(path, root_dir), emo, _level_from_path(path), _utt_stem(path)), path)

    records = []
    n_video_total = n_frontal = n_audio_matched = 0
    views_seen, levels_seen = set(), set()

    for path in glob.iglob(os.path.join(root_dir, '**', '*'), recursive=True):
        if not path.lower().endswith(VIDEO_EXTS):
            continue
        emo = _emotion_from_path(path)
        if emo is None:
            continue
        n_video_total += 1
        for c in _components(path):
            if c.startswith(('front', 'left', 'right', 'top', 'down')):
                views_seen.add(c)
        levels_seen.add(_level_from_path(path))

        if not _is_frontal(path):
            continue
        n_frontal += 1

        actor = _actor_token(path, root_dir)
        level = _level_from_path(path)
        apath = audio_index.get((actor, emo, level, _utt_stem(path)))
        if apath is not None:
            n_audio_matched += 1

        records.append({
            'emotion':    MEAD_EMOTION_MAP[emo],
            'actor_tok':  actor,
            'intensity':  level,
            'video_path': path,
            'audio_path': apath,
        })

    actor_tokens = sorted({r['actor_tok'] for r in records})
    actor_to_id = {a: i for i, a in enumerate(actor_tokens)}
    for r in records:
        r['actor'] = actor_to_id[r['actor_tok']]

    diag = {
        'n_video_total': n_video_total, 'n_frontal': n_frontal,
        'n_records': len(records), 'n_audio_matched': n_audio_matched,
        'views_seen': sorted(views_seen), 'levels_seen': sorted(levels_seen),
        'actor_tokens': actor_tokens,
    }
    return records, diag


def _print_diag(diag):
    print(f'  Total video files found:        {diag["n_video_total"]:,}')
    print(f'  Frontal-view videos (all 8):    {diag["n_frontal"]:,}')
    print(f'  Final records (with actor id):  {diag["n_records"]:,}')
    print(f'  Audio matched to video:         {diag["n_audio_matched"]:,} / {diag["n_records"]:,}')
    print(f'  View tokens seen:               {diag["views_seen"] or "NONE -- check FRONTAL_VIEW_TOKENS"}')
    print(f'  Intensity levels seen:          {diag["levels_seen"]}')
    print(f'  Actors discovered:              {len(diag["actor_tokens"])}')
    if diag['n_frontal'] == 0:
        print('  *** No frontal videos matched. Your copy likely names the frontal view')
        print('      folder something other than "front" -- set FRONTAL_VIEW_TOKENS from')
        print('      the view tokens above, then re-run --dry-run.')
    if diag['n_records'] and diag['n_audio_matched'] / diag['n_records'] < 0.9:
        print('  *** Many clips lack matched audio. MEAD .mp4s are often silent, so those')
        print('      clips would extract as zero-audio. Inspect the audio/ tree naming.')





# load the frozen dinov2 and wavlm feature extractors
def load_backbones():
    print('=' * 70 + '\nLoading frozen pretrained backbones\n' + '=' * 70)
    print(f'  DINOv2: {DINOV2_MODEL}\n  WavLM:  {WAVLM_MODEL}\n  Device: {device}')
    dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    dinov2 = AutoModel.from_pretrained(DINOV2_MODEL).to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False
    wavlm = WavLMModel.from_pretrained(WAVLM_MODEL).to(device).eval()
    for p in wavlm.parameters():
        p.requires_grad = False
    print(f'  DINOv2 params: {sum(p.numel() for p in dinov2.parameters()):,} (frozen)')
    print(f'  WavLM  params: {sum(p.numel() for p in wavlm.parameters()):,} (frozen)')
    with torch.no_grad():
        do = dinov2(pixel_values=torch.randn(1, 3, 224, 224, device=device)).last_hidden_state
        wo = wavlm(input_values=torch.randn(1, 16000, device=device)).last_hidden_state
        print(f'  DINOv2 forward: (1,3,224,224) -> {tuple(do.shape)}  (CLS = idx 0)')
        print(f'  WavLM  forward: (1,16000)      -> {tuple(wo.shape)}  (~50 Hz)')
    return dinov2_processor, dinov2, wavlm





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
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb)
                if boxes is not None and len(boxes) > 0:
                    bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    box = (max(0, int(bx1)), max(0, int(by1)), int(bx2 - bx1), int(by2 - by1))
            except Exception:
                box = None
        if box is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
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
        return cv2.cvtColor(cv2.resize(face_bgr, (FACE_SIZE, FACE_SIZE)), cv2.COLOR_BGR2RGB)


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
def extract_audio_16khz(media_path):
    tmp_wav = tempfile.mktemp(suffix='.wav')
    try:
        cmd = [FFMPEG_EXE, '-i', media_path, '-vn', '-acodec', 'pcm_s16le',
               '-ar', str(SR_AUDIO), '-ac', '1', '-y', tmp_wav]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
            return np.zeros(SR_AUDIO, dtype=np.float32)
        y, sr = librosa.load(tmp_wav, sr=SR_AUDIO, mono=True)
        return y.astype(np.float32) if y.size else np.zeros(SR_AUDIO, dtype=np.float32)
    except Exception as e:
        print(f'  audio load failed for {media_path}: {e}')
        return np.zeros(SR_AUDIO, dtype=np.float32)
    finally:
        if os.path.exists(tmp_wav):
            try: os.remove(tmp_wav)
            except Exception: pass





@torch.no_grad()
# get video features from the frozen dinov2
def dinov2_forward(face_stack_rgb_uint8, dinov2_processor, dinov2):
    pil_imgs = [face_stack_rgb_uint8[i] for i in range(face_stack_rgb_uint8.shape[0])]
    inputs = dinov2_processor(images=pil_imgs, return_tensors='pt').to(device)
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
        out = dinov2(**inputs).last_hidden_state
    return out[:, 0, :].float().cpu().numpy()


@torch.no_grad()
# get audio features from the frozen wavlm
def wavlm_forward(audio_1d, wavlm):
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





# sanity check the saved cache
def verify_cache(npz_path, expected_labels):
    print('\n  --- verification ---')
    d = np.load(npz_path, allow_pickle=True)
    vf, af, al, lab, act = (d['video_feats'], d['audio_feats'], d['audio_lens'],
                            d['labels'], d['actors'])
    N = vf.shape[0]
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(f'    [{"OK " if cond else "FAIL"}] {msg}')
        ok = ok and cond
    check(vf.shape == (N, T_VIDEO, 768), f'video_feats shape {vf.shape}')
    check(af.shape == (N, MAX_AUDIO_FRAMES, 768), f'audio_feats shape {af.shape}')
    check(al.shape[0] == N and lab.shape[0] == N and act.shape[0] == N, 'all arrays length N')
    check(not np.isnan(vf).any() and not np.isinf(vf).any(), 'video_feats finite (no NaN/Inf)')
    check(not np.isnan(af).any() and not np.isinf(af).any(), 'audio_feats finite (no NaN/Inf)')
    check(float(np.abs(vf).max()) > 0, 'video_feats not all-zero')
    check(float(np.abs(af).max()) > 0, 'audio_feats not all-zero')
    check(int(al.min()) >= 1 and int(al.max()) <= MAX_AUDIO_FRAMES, 'audio_lens in [1, MAX]')
    vocab = set(map(str, np.unique(lab)))
    check(vocab == set(expected_labels), f'label vocab == expected ({sorted(vocab)})')
    silent = int((np.abs(af).reshape(N, -1).max(axis=1) == 0).sum())
    blankvid = int((np.abs(vf).reshape(N, -1).max(axis=1) == 0).sum())
    print(f'    [i ] silent-audio clips: {silent}/{N}   all-blank-video clips: {blankvid}/{N}')
    u, c = np.unique(lab, return_counts=True)
    print(f'    [i ] class balance: ' + ', '.join(f'{x}={n}' for x, n in zip(u, c)))
    print(f'    [i ] mean samples/class: {c.mean():.0f}  (CREMA-D ~1240, RAVDESS ~180)')
    print(f'    [i ] actors: {len(np.unique(act))}')
    print(f'  --- {"ALL CHECKS PASSED" if ok else "CHECKS FAILED -- see above"} ---')
    return ok





# run extraction over all clips and save the cache
def process_mead(dinov2_processor, dinov2, wavlm, dry_run=False):
    print('\n' + '=' * 70 + '\nProcessing MEAD (all 8 native classes)\n' + '=' * 70)
    print(f'  Raw root: {MEAD_RAW_DIR}\n  Cache:    {MEAD8_NPZ}')

    records, diag = collect_mead_clips(MEAD_RAW_DIR)
    _print_diag(diag)
    if dry_run:
        print('\n  --dry-run: discovery only, nothing extracted.')
        return
    if os.path.exists(MEAD8_NPZ):
        print(f'  {MEAD8_NPZ} exists; skipping. Delete to re-extract.')
        return
    N = len(records)
    if N == 0:
        print('  WARNING: no clips to extract. Fix discovery (see notes above).')
        return

    cropper = FaceCropper(use_mtcnn=True)
    video_feats = np.zeros((N, T_VIDEO, 768), dtype=np.float32)
    audio_feats = np.zeros((N, MAX_AUDIO_FRAMES, 768), dtype=np.float32)
    audio_lens  = np.zeros(N, dtype=np.int32)
    labels      = np.empty(N, dtype=object)
    actors      = np.zeros(N, dtype=np.int32)
    intensity   = np.empty(N, dtype=object)
    view        = np.empty(N, dtype=object)
    blank_face_counts = np.zeros(N, dtype=np.int32)

    t0 = time.time()
    for i, rec in enumerate(tqdm(records, desc='mead clips', mininterval=30)):
        if i % 100 == 0:
            print(f'[PROGRESS] {i}/{len(records)} clips done', flush=True)
        fs = extract_face_stack(rec['video_path'], cropper)
        blank_face_counts[i] = int((fs.sum(axis=(1, 2, 3)) == 0).sum())
        video_feats[i] = dinov2_forward(fs, dinov2_processor, dinov2)
        audio_src = rec['audio_path'] or rec['video_path']
        audio = extract_audio_16khz(audio_src)
        seq = wavlm_forward(audio, wavlm)
        a_pad, a_len = pad_audio_feats(seq)
        audio_feats[i] = a_pad
        audio_lens[i]  = a_len
        labels[i], actors[i] = rec['emotion'], rec['actor']
        intensity[i], view[i] = rec['intensity'], 'front'

    print(f'  extracted in {(time.time()-t0)/60.0:.1f} min; '
          f'blank_face_avg/clip {blank_face_counts.mean():.2f}/30')
    np.savez_compressed(MEAD8_NPZ,
                        video_feats=video_feats, audio_feats=audio_feats,
                        audio_lens=audio_lens, labels=labels.astype(str),
                        actors=actors, intensity=intensity.astype(str),
                        view=view.astype(str), blank_face_counts=blank_face_counts)
    print(f'  Saved {MEAD8_NPZ} ({os.path.getsize(MEAD8_NPZ)/1e6:.1f} MB)')
    verify_cache(MEAD8_NPZ, set(MEAD_EMOTION_MAP.values()))


# make the six class version that matches crema d
def filter_crema6():
    print('=' * 70 + '\nDeriving mead6 (CREMA-D-matched 6 classes) from mead8\n' + '=' * 70)
    if not os.path.exists(MEAD8_NPZ):
        print(f'ERROR: {MEAD8_NPZ} not found. Run the extraction first.')
        return
    d = np.load(MEAD8_NPZ, allow_pickle=True)
    keep = np.isin(d['labels'].astype(str), np.array(CREMA6_LABELS))
    print(f'  Keeping {int(keep.sum()):,} / {len(keep):,} clips '
          f'(classes: {", ".join(CREMA6_LABELS)})')
    out = {k: (d[k][keep] if d[k].shape[:1] == keep.shape else d[k]) for k in d.files}
    np.savez_compressed(MEAD6_NPZ, **out)
    print(f'  Saved {MEAD6_NPZ} ({os.path.getsize(MEAD6_NPZ)/1e6:.1f} MB)')
    verify_cache(MEAD6_NPZ, set(CREMA6_LABELS))





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Discover and report clip counts only; no backbones, no extraction.')
    ap.add_argument('--filter-crema6', action='store_true',
                    help='Derive mead6_pretrained.npz from mead8_pretrained.npz; no GPU.')
    args = ap.parse_args()

    print('=' * 70 + '\nNB05a (MEAD): DINOv2-base + WavLM-base+ feature extraction\n' + '=' * 70)
    print(f'ffmpeg: {FFMPEG_EXE}')

    if args.filter_crema6:
        filter_crema6(); sys.exit(0)
    if args.dry_run:
        process_mead(None, None, None, dry_run=True); sys.exit(0)

    proc, dinov2, wavlm = load_backbones()
    process_mead(proc, dinov2, wavlm, dry_run=False)
    print('\nDone. Next: python 05a_extract_pretrained_features_mead.py --filter-crema6')
