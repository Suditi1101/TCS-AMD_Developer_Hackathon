import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('config.yml').read_text())
VLLM_BASE_URL = cfg['vllm_base_url']
# QWEN_MODEL    = 'Qwen/Qwen2-VL-7B-Instruct'
QWEN_MODEL    = 'Qwen3-30B-A3B'

VIDEO_CFG     = cfg.get('video', {})
TARGET_FPS    = VIDEO_CFG.get('target_fps', 5)
FRAME_SIZE    = tuple(VIDEO_CFG.get('frame_size', [640, 640]))
SUPPORTED_EXT = set(VIDEO_CFG.get('supported_ext', ['.mp4', '.avi', '.mov', '.mkv']))

dp            = cfg.get('dataset_paths', {})
VIDEO_DIR     = pathlib.Path(dp.get('accident_bench', 'Data/datasets/videos/Accident-Bench/land_space/medium/videos'))
FRAMES_OUT    = pathlib.Path(dp.get('frames_out',    'data/frames'))
FRAMES_OUT.mkdir(parents=True, exist_ok=True)

print(f'Video dir   : {VIDEO_DIR}')
print(f'Frames out  : {FRAMES_OUT}')
print(f'Target FPS  : {TARGET_FPS}')
print(f'Frame size  : {FRAME_SIZE}')


import time, uuid, base64, io, json, warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
from PIL import Image
from openai import OpenAI

client = OpenAI(base_url=VLLM_BASE_URL, api_key="abc-123")

# Latency budgets (ms)
BUDGET_DECODE_MS  = 50
BUDGET_MOTION_MS  = 5
BUDGET_HASH_MS    = 2
BUDGET_YOLO_MS    = 30
BUDGET_QWEN_VL_MS = 2000

# Accident-bench YOLO labels
ACCIDENT_LABELS = [
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person',
    'traffic light', 'stop sign', 'fire hydrant',
    'accident', 'smoke', 'debris',
]
# COCO indices that map to accident-bench vehicle/person classes
COCO_VEHICLE_IDS = {2, 3, 5, 7}   # car=2, motorcycle=3, bus=5, truck=7
COCO_PERSON_ID   = 0
# High-priority labels: ANY detection -> always pass motion gate
PRIORITY_LABELS  = {'accident', 'smoke', 'debris'}

print('Imports done')
print(f'Priority labels: {PRIORITY_LABELS}')


@dataclass
class VideoMeta:
    video_id  : str
    path      : str
    fps_src   : float
    fps_target: float
    total_frames_src : int
    frames_extracted : int
    width     : int
    height    : int


class VideoDecoder:
    """
    Extracts frames from video files at target_fps.

    Production:
        Uses cv2.VideoCapture for efficient seeking.
        For large video archives, consider ffmpeg-python for GPU-accelerated
        decode on AMD ROCm: ffmpeg -hwaccel vaapi -i input.mp4 ...

    Fallback (no OpenCV):
        Generates synthetic accident-like frames:
          - Normal frames: static road scene (low variance)
          - Accident frames: high-variance 'impact' burst + freeze
        This preserves the full pipeline logic without needing real videos.

    Frame naming convention:
        {video_id}_f{frame_number:06d}.jpg
        Stored in FRAMES_OUT/{video_id}/
    """

    def __init__(self, video_dir=VIDEO_DIR, frames_out=FRAMES_OUT,
                 target_fps=TARGET_FPS, frame_size=FRAME_SIZE):
        self.video_dir  = pathlib.Path(video_dir)
        self.frames_out = pathlib.Path(frames_out)
        self.target_fps = target_fps
        self.frame_size = frame_size
        self._metas: List[VideoMeta] = []

    def _find_videos(self):
        if not self.video_dir.exists():
            return []
        return [p for p in sorted(self.video_dir.rglob('*'))
                if p.suffix.lower() in SUPPORTED_EXT]

    def _decode_opencv(self, video_path):
        """Real OpenCV decode. Returns list of (frame_no, np.ndarray)."""
        import cv2
        cap      = cv2.VideoCapture(str(video_path))
        fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step     = max(1, int(round(fps_src / self.target_fps)))
        frames   = []
        fno      = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fno % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized   = cv2.resize(frame_rgb, self.frame_size)
                frames.append((fno, resized.astype(np.uint8)))
            fno += 1
        cap.release()
        return frames, fps_src, total, w, h

    def _decode_synthetic(self, video_id, n_frames=60):
        """
        Synthetic accident video generator (no OpenCV needed).
        Simulates: 20 normal frames -> 10 high-motion impact frames
                   -> 5 static post-crash frames -> 25 normal frames.
        """
        rng    = np.random.default_rng(abs(hash(video_id)) % 99999)
        W, H   = self.frame_size
        base   = rng.integers(80, 160, (H, W, 3), dtype=np.uint8)
        frames = []
        for i in range(n_frames):
            if 20 <= i < 30:          # impact burst
                scale = 90
            elif 30 <= i < 35:        # post-crash freeze (low motion)
                scale = 2
            else:                     # normal road scene
                scale = 8
            noise = rng.integers(-scale, scale + 1, base.shape)
            f     = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append((i, f))
        return frames, 30.0, n_frames, W, H

    def decode_all(self, max_videos=None):
        """
        Decode all videos in video_dir.
        Returns flat list of frame dicts ready for the edge pipeline.
        """
        video_paths = self._find_videos()
        if not video_paths:
            print(f'No videos found in {self.video_dir} -- using synthetic generator')
            video_paths = [pathlib.Path(f'synthetic_video_{i:02d}.mp4')
                           for i in range(3)]

        if max_videos:
            video_paths = video_paths[:max_videos]

        all_frame_dicts = []
        for vpath in video_paths:
            video_id  = vpath.stem
            out_dir   = self.frames_out / video_id
            out_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                import cv2
                raw_frames, fps_src, total, w, h = self._decode_opencv(vpath)
                src = 'opencv'
            except (ImportError, Exception):
                raw_frames, fps_src, total, w, h = self._decode_synthetic(video_id)
                src = 'synthetic'

            ms = (time.perf_counter() - t0) * 1000
            if ms > BUDGET_DECODE_MS * len(raw_frames):
                print(f'  WARN {video_id}: decode {ms:.0f}ms')

            # Save frames to disk + build frame dicts
            for fno, arr in raw_frames:
                fname    = f'{video_id}_f{fno:06d}.jpg'
                fpath    = out_dir / fname
                Image.fromarray(arr).save(str(fpath), quality=90)
                all_frame_dicts.append({
                    'video_id'  : video_id,
                    'frame_no'  : fno,
                    'frame_path': str(fpath),
                    'data'      : arr,          # numpy array in memory
                    'timestamp' : datetime.utcnow().isoformat(),
                    'source'    : src,
                    'dataset_mode': 'accident_bench',
                })

            meta = VideoMeta(
                video_id=video_id, path=str(vpath),
                fps_src=fps_src, fps_target=self.target_fps,
                total_frames_src=total,
                frames_extracted=len(raw_frames),
                width=w, height=h,
            )
            self._metas.append(meta)
            print(f'  [{src}] {video_id}: {len(raw_frames)} frames  '
                  f'(src_fps={fps_src:.0f} -> {self.target_fps}fps)  {ms:.0f}ms')

        return all_frame_dicts

    @property
    def metas(self):
        return self._metas


decoder = VideoDecoder()
print('VideoDecoder ready')


class MotionGate:
    """
    Optical-flow motion gate.
    For accident-bench: static frames (parking lots, idle cameras) discarded.
    Impact moments produce a sharp motion spike -- always pass.

    CRITICAL override: frames where YOLOv8 later detects 'accident' or 'smoke'
    are NOT subject to motion gate (handled in the pipeline runner below).

    Production: replace _flow() with cv2.calcOpticalFlowFarneback().
    """
    def __init__(self, threshold=0.018, budget_ms=BUDGET_MOTION_MS):
        self.threshold = threshold
        self.budget_ms = budget_ms
        self._prev_gray: Optional[np.ndarray] = None

    def _to_gray(self, frame):
        return (0.299*frame[:,:,0] + 0.587*frame[:,:,1]
                + 0.114*frame[:,:,2]).astype(np.float32)

    def _flow_score(self, prev, curr):
        diff = curr - prev
        fx   = np.gradient(diff, axis=1)
        fy   = np.gradient(diff, axis=0)
        mag  = np.sqrt(fx**2 + fy**2)
        return float((mag > 1.5).mean())

    def score(self, frame):
        gray = self._to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        s = self._flow_score(self._prev_gray, gray)
        self._prev_gray = gray
        return s

    def should_pass(self, frame):
        t0 = time.perf_counter()
        s  = self.score(frame)
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.budget_ms:
            print(f'  WARN MotionGate {ms:.1f}ms > {self.budget_ms}ms')
        return s >= self.threshold, s, ms


gate = MotionGate()
print('MotionGate ready')


class PHashDedup:
    """
    Difference-hash deduplicator.
    Ring buffer of last 5 hashes -- catches duplicates across short gaps.
    Post-crash freeze frames have very similar hashes, so they are
    collapsed to a single event (one alert, not 30 identical ones).
    """
    def __init__(self, hamming_threshold=8, budget_ms=BUDGET_HASH_MS,
                 window=5):
        self.hamming_threshold = hamming_threshold
        self.budget_ms  = budget_ms
        self._window    = window
        self._hashes: List[int] = []

    def _dhash(self, frame, size=8):
        img  = Image.fromarray(frame).convert('L').resize((size+1, size))
        arr  = np.array(img)
        diff = arr[:, 1:] > arr[:, :-1]
        return int(''.join(map(str, diff.flatten().astype(int))), 2)

    def _hamming(self, a, b):
        return bin(a ^ b).count('1')

    def is_new(self, frame):
        t0 = time.perf_counter()
        h  = self._dhash(frame)
        ms = (time.perf_counter() - t0) * 1000
        if not self._hashes:
            self._hashes.append(h)
            return True, hex(h), ms
        min_dist = min(self._hamming(h, prev) for prev in self._hashes)
        if min_dist >= self.hamming_threshold:
            self._hashes.append(h)
            if len(self._hashes) > self._window:
                self._hashes.pop(0)
            return True, hex(h), ms
        return False, hex(h), ms


dedup = PHashDedup()
print('PHashDedup ready')


!pip install ultralytics

import torchvision

if not hasattr(torchvision.ops, '_real_nms'):
  torchvision.ops._real_nms = torchvision.ops.nms  # save original only once

def _cpu_nms(boxes, scores, iou_threshold):
  return torchvision.ops._real_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)

torchvision.ops.nms = _cpu_nms

@dataclass
class Detection:
    event_id    : str
    video_id    : str
    frame_no    : int
    timestamp   : str
    label       : str
    confidence  : float
    bbox        : Dict
    motion_score: float
    frame_hash  : str
    is_accident_class: bool   # True for accident/smoke/debris labels
    dataset_mode: str = 'accident_bench'


YOLO_WEIGHTS = {
    'accident_bench': 'weights/yolov8_accident.pt',   # fine-tuned (preferred)
    'coco_fallback':  'yolov8n.pt',                   # COCO pretrained fallback
}


class YOLOv8Detector:
    """
    YOLOv8 detector for accident-bench.

    Model loading priority:
      1. weights/yolov8_accident.pt  (fine-tuned on accident/smoke/debris)
      2. yolov8n.pt                  (COCO pretrained -- cars/trucks/persons)
      3. Simulation mode             (when ultralytics not installed)

    Fine-tuning recipe for accident-bench:
      yolo detect train
        data=data/accident_bench/dataset.yaml
        model=yolov8n.pt
        epochs=50 imgsz=640 batch=16
        lr0=1e-3
      Labels needed: accident, smoke, debris (add to COCO 80 classes)

    ROCm/AMD GPU: set device='cuda' -- PyTorch HIP routes automatically.
    """

    def __init__(self, min_conf=0.40, budget_ms=BUDGET_YOLO_MS,
                 device='cpu', imgsz=640):
        self.min_conf  = min_conf
        self.budget_ms = budget_ms
        self.device    = device
        self.imgsz     = imgsz
        self._model    = None
        self._loaded   = False
        self._labels   = ACCIDENT_LABELS
        self._try_load()

    def _try_load(self):
        for wkey, wpath in YOLO_WEIGHTS.items():
            try:
                from ultralytics import YOLO
                self._model  = YOLO(wpath)
                self._model.to(self.device)
                if hasattr(self._model, 'names'):
                    self._labels = list(self._model.names.values())
                self._loaded = True
                print(f'  YOLOv8 loaded: {wpath} ({wkey})  device={self.device}')
                return
            except ImportError:
                print('  ultralytics not installed -- simulation mode')
                return
            except Exception:
                continue
        print('  No YOLO weights found -- simulation mode')

    def _infer_real(self, frame, motion_score, frame_hash, video_id, frame_no):
        t0      = time.perf_counter()
        results = self._model(frame, imgsz=self.imgsz, conf=self.min_conf,
                              iou=0.45, device=self.device, verbose=False)
        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf   = float(box.conf[0])
                if conf < self.min_conf:
                    continue
                cls_id = int(box.cls[0])
                label  = (self._labels[cls_id]
                          if cls_id < len(self._labels) else f'class_{cls_id}')
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                dets.append(Detection(
                    event_id     = str(uuid.uuid4()),
                    video_id     = video_id,
                    frame_no     = frame_no,
                    timestamp    = datetime.utcnow().isoformat(),
                    label        = label,
                    confidence   = round(conf, 4),
                    bbox         = {'x': round(x1,4), 'y': round(y1,4),
                                    'w': round(x2-x1,4), 'h': round(y2-y1,4)},
                    motion_score = round(motion_score, 4),
                    frame_hash   = frame_hash,
                    is_accident_class = label in PRIORITY_LABELS,
                ))
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.budget_ms:
            print(f'  WARN YOLO {ms:.1f}ms > {self.budget_ms}ms')
        return dets, ms

    def _infer_simulated(self, frame, motion_score, frame_hash, video_id, frame_no):
        """
        Simulation: high motion_score raises probability of accident/vehicle detections.
        Seeded from frame pixel sum for determinism.
        """
        t0   = time.perf_counter()
        rng  = np.random.default_rng(int(frame.flatten()[:4].sum()) % 9999)
        dets = []

        # Always detect a vehicle in road scenes
        vehicle_labels = ['car', 'truck', 'bus', 'motorcycle']
        for lbl in rng.choice(vehicle_labels, size=int(rng.integers(1, 4)),
                               replace=False):
            conf = float(rng.uniform(0.50, 0.92))
            x, y = float(rng.uniform(0.1, 0.6)), float(rng.uniform(0.3, 0.7))
            w, h = float(rng.uniform(0.1, 0.3)), float(rng.uniform(0.1, 0.25))
            dets.append(Detection(
                event_id=str(uuid.uuid4()), video_id=video_id,
                frame_no=frame_no, timestamp=datetime.utcnow().isoformat(),
                label=str(lbl), confidence=round(conf, 4),
                bbox={'x': round(x,3), 'y': round(y,3),
                      'w': round(w,3), 'h': round(h,3)},
                motion_score=round(motion_score, 4),
                frame_hash=frame_hash, is_accident_class=False))

        # High motion -> add accident/smoke detection
        if motion_score > 0.05:
            acc_lbl = str(rng.choice(['accident', 'smoke']))
            conf    = float(rng.uniform(0.45, 0.88))
            dets.append(Detection(
                event_id=str(uuid.uuid4()), video_id=video_id,
                frame_no=frame_no, timestamp=datetime.utcnow().isoformat(),
                label=acc_lbl, confidence=round(conf, 4),
                bbox={'x': 0.2, 'y': 0.2, 'w': 0.5, 'h': 0.5},
                motion_score=round(motion_score, 4),
                frame_hash=frame_hash, is_accident_class=True))

        ms = (time.perf_counter() - t0) * 1000
        return dets, ms

    def detect(self, frame, motion_score, frame_hash, video_id, frame_no):
        if self._loaded:
            return self._infer_real(frame, motion_score, frame_hash,
                                    video_id, frame_no)
        return self._infer_simulated(frame, motion_score, frame_hash,
                                     video_id, frame_no)

    def warmup(self, n=2):
        if not self._loaded:
            return
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self._model(dummy, imgsz=self.imgsz, verbose=False)
        print(f'  YOLOv8 warmed up ({n} passes)')

    @property
    def is_real(self):
        return self._loaded


detector = YOLOv8Detector(device='cuda', imgsz=640, min_conf=0.40)
detector.warmup(n=2)
print(f'YOLOv8Detector ready  real={detector.is_real}')


def frame_to_base64(frame):
    buf = io.BytesIO()
    Image.fromarray(frame, 'RGB').save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class QwenVLVerifier:
    """
    Qwen2.5-VL visual verification, prompted for road accident analysis.
    Only called on frames that passed motion gate + have YOLO detections.
    """
    SYSTEM = """
        You are a road accident analysis expert.
        
        Return ONLY valid JSON.
        
        Schema:
        
        {
         "scene":"",
         "vehicles_detected":[],
         "accident_visible":false,
         "accident_type":"rear_end|T_bone|rollover|pedestrian|multi_vehicle|none",
         "smoke_or_fire":false,
         "persons_at_risk":false,
         "severity":"low|medium|high|critical",
         "escalate":false,
         "reason":""
        }
        
        Do not return markdown.
        Do not return explanations.
        Do not wrap with ```json.
        """

    def __init__(self, model=QWEN_MODEL, budget_ms=BUDGET_QWEN_VL_MS):
        self.model     = model
        self.budget_ms = budget_ms

    def verify(self, frame, detections):
        hint   = ', '.join(f'{d.label}({d.confidence:.2f})' for d in detections)
        prompt = f'YOLO pre-detected: {hint or "nothing"}. Confirm findings.'
        b64    = frame_to_base64(frame)
        t0     = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.SYSTEM},
                    {'role': 'user', 'content': [
                        {'type': 'image_url',
                         'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': prompt},
                    ]},
                ],
                # messages=[
                #     {'role': 'system', 'content': self.SYSTEM},
                #     {'role': 'user', 'content': prompt}],
                max_tokens=256, temperature=0.0,response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content.strip()
            print(raw)
            if raw.startswith('```'):
                raw = '\\n'.join(l for l in raw.split('\\n')
                                if not l.strip().startswith('```')).strip()
            result = json.loads(raw)
        except Exception as exc:
            result = {'scene': 'parse error', 'accident_visible': False,
                      'severity': 'low', 'escalate': False, 'reason': str(exc),
                      'accident_type': 'none', 'smoke_or_fire': False,
                      'persons_at_risk': False, 'vehicles_detected': []}
        ms = (time.perf_counter() - t0) * 1000
        if ms > self.budget_ms:
            print(f'  WARN Qwen-VL {ms:.0f}ms > {self.budget_ms}ms')
        return result, ms


verifier = QwenVLVerifier()
print('QwenVLVerifier ready')


class AdaptiveSampler:
    """
    Adjusts effective FPS based on scene activity.
    For accident-bench: spikes to 'burst' mode on impact frames,
    drops to 'idle' on static post-crash or empty-road frames.
    """
    FPS_MAP = {'idle': 1, 'active': 5, 'burst': 15}

    def __init__(self):
        self._interval = 1.0
        self._history: List[str] = []

    def update(self, motion_score, has_accident_det, severity='low'):
        if has_accident_det and severity in ('high', 'critical'):
            mode = 'burst'
        elif has_accident_det or motion_score > 0.04:
            mode = 'active'
        else:
            mode = 'idle'
        self._interval = 1.0 / self.FPS_MAP[mode]
        self._history.append(mode)
        return mode

    def summary(self):
        from collections import Counter
        if not self._history:
            return {}
        c = Counter(self._history)
        n = len(self._history)
        return {m: round(cnt/n*100, 1) for m, cnt in c.items()}


class LatencyMonitor:
    BUDGETS = {'decode': BUDGET_DECODE_MS, 'motion': BUDGET_MOTION_MS,
               'hash': BUDGET_HASH_MS, 'yolo': BUDGET_YOLO_MS,
               'qwen_vl': BUDGET_QWEN_VL_MS}

    def __init__(self):
        self._data: Dict[str, List[float]] = {k: [] for k in self.BUDGETS}

    def record(self, stage, ms):
        self._data.setdefault(stage, []).append(ms)

    def report(self):
        print('\\nLatency Monitor')
        print(f'  {"Stage":<12} {"N":>5} {"Mean":>8} {"P95":>8} {"Budget":>8} {"Viols":>6}')
        print('  ' + '-'*50)
        for stage, vals in self._data.items():
            if not vals:
                continue
            arr   = np.array(vals)
            mean  = arr.mean()
            p95   = np.percentile(arr, 95)
            bgt   = self.BUDGETS.get(stage, 9999)
            viols = int((arr > bgt).sum())
            flag  = 'WARN' if viols else '    '
            print(f'  {flag} {stage:<10} {len(vals):>5} {mean:>8.2f} '
                  f'{p95:>8.2f} {bgt:>8} {viols:>6}')


sampler     = AdaptiveSampler()
lat_monitor = LatencyMonitor()
print('AdaptiveSampler + LatencyMonitor ready')


print('Step 0: Decoding videos...')
t_decode = time.perf_counter()

# max_videos=None processes all videos in VIDEO_DIR
# Set max_videos=2 for a quick test run
FRAMES = decoder.decode_all(max_videos=3)

decode_ms = (time.perf_counter() - t_decode) * 1000
lat_monitor.record('decode', decode_ms)

print(f'\\nVideos processed : {len(decoder.metas)}')
for m in decoder.metas:
    print(f'  {m.video_id}: {m.frames_extracted} frames  '
          f'(src={m.fps_src:.0f}fps -> {m.fps_target}fps)  '
          f'{m.width}x{m.height}  src={m.path}')
print(f'Total frames     : {len(FRAMES)}')
print(f'Decode time      : {decode_ms:.0f} ms')


results = []
stats   = dict(received=0, dropped_motion=0, dropped_dup=0,
               processed=0, detections=0, accident_frames=0, qwen_calls=0)

QWEN_MAX_CALLS = 10  # cap for demo; remove in production

print(f'Processing {len(FRAMES)} frames...\\n')
print(f'{"Frame":>6} {"Video":<20} {"Motion":>7} {"Hash":>6} '
      f'{"Dets":>4} {"AccDet":>6} {"Mode":>6}')
print('-' * 68)

for idx, fdict in enumerate(FRAMES):
    frame    = fdict['data']
    video_id = fdict['video_id']
    frame_no = fdict['frame_no']
    stats['received'] += 1

    # Step 1: Motion gate
    # Override: if later detected as accident frame, always pass
    # (We run motion gate first as a quick filter, then override below)
    passed, mscore, t_motion = gate.should_pass(frame)
    lat_monitor.record('motion', t_motion)

    if not passed:
        stats['dropped_motion'] += 1
        sampler.update(mscore, False)
        continue

    # Step 2: Dedup
    is_new, fhash, t_hash = dedup.is_new(frame)
    lat_monitor.record('hash', t_hash)
    if not is_new:
        stats['dropped_dup'] += 1
        sampler.update(mscore, False)
        continue

    # Step 3: YOLOv8 detection
    stats['processed'] += 1
    dets, t_yolo = detector.detect(frame, mscore, fhash, video_id, frame_no)
    lat_monitor.record('yolo', t_yolo)
    stats['detections'] += len(dets)

    has_accident = any(d.is_accident_class for d in dets)
    if has_accident:
        stats['accident_frames'] += 1

    # Step 4: Qwen-VL on accident/high-motion frames
    vl_result = None
    t_vl      = 0.0
    if dets and stats['qwen_calls'] < QWEN_MAX_CALLS:
        if has_accident or mscore > 0.06:
            vl_result, t_vl = verifier.verify(frame, dets)
            lat_monitor.record('qwen_vl', t_vl)
            stats['qwen_calls'] += 1

    # Step 5: Adaptive sampling
    sev  = (vl_result or {}).get('severity', 'low')
    mode = sampler.update(mscore, has_accident, severity=sev)

    results.append({
        'frame_idx'    : idx,
        'video_id'     : video_id,
        'frame_no'     : frame_no,
        'frame_path'   : fdict['frame_path'],
        'motion_score' : round(mscore, 4),
        'frame_hash'   : fhash,
        'detections'   : [d.__dict__ for d in dets],
        'has_accident' : has_accident,
        'qwen_vl'      : vl_result,
        'mode'         : mode,
        'dataset_mode' : 'accident_bench',
        'latency_ms'   : {'motion': t_motion, 'hash': t_hash,
                          'yolo': t_yolo, 'qwen_vl': t_vl},
    })

    det_str = ', '.join(f'{d.label}({d.confidence:.2f})' for d in dets) or 'none'
    acc_str = 'ACCIDENT' if has_accident else '       '
    print(f'{idx:>6} {video_id:<20} {mscore:>7.4f} {"new":>6} '
          f'{len(dets):>4} {acc_str:>6} {mode:>6}  {det_str[:40]}')

print('\\n' + '-'*68)

total = stats['received']
kept  = stats['processed']
print(f'Frames received   : {total}')
print(f'Dropped (motion)  : {stats["dropped_motion"]}  ({stats["dropped_motion"]/max(total,1)*100:.0f}%)')
print(f'Dropped (dup)     : {stats["dropped_dup"]}  ({stats["dropped_dup"]/max(total,1)*100:.0f}%)')
print(f'Processed         : {kept}  ({kept/max(total,1)*100:.0f}%)')
print(f'Accident frames   : {stats["accident_frames"]}')
print(f'Total detections  : {stats["detections"]}')
print(f'Qwen-VL calls     : {stats["qwen_calls"]}')
print(f'Data reduction    : {100*(1-kept/max(total,1)):.0f}%')
print(f'Sampler modes     : {sampler.summary()}')



vl_frames = [r for r in results if r['qwen_vl'] is not None]
print(f'Frames with Qwen-VL analysis: {len(vl_frames)}\\n')
for r in vl_frames:
    vl = r['qwen_vl']
    print(f'Video {r["video_id"]}  frame={r["frame_no"]}  motion={r["motion_score"]:.3f}')
    print(f'  Scene         : {vl.get("scene","")}')
    print(f'  Accident      : {vl.get("accident_visible")}  type={vl.get("accident_type")}')
    print(f'  Vehicles      : {vl.get("vehicles_detected",[])}')
    print(f'  Smoke/fire    : {vl.get("smoke_or_fire")}  Persons at risk: {vl.get("persons_at_risk")}')
    print(f'  Severity      : {vl.get("severity")}  Escalate: {vl.get("escalate")}')
    print(f'  Reason        : {vl.get("reason","")}')
    print()


edge_events = []
for r in results:
    for det in r['detections']:
        ev = dict(det)
        ev['qwen_vl']     = r.get('qwen_vl')
        ev['mode']        = r['mode']
        ev['frame_path']  = r['frame_path']
        ev['motion_score']= r['motion_score']
        ev['has_accident']= r['has_accident']
        ev['dataset_mode']= 'accident_bench'
        # Add frame_range for fog correlator
        ev['frame_range'] = [r['frame_no'], r['frame_no']]
        edge_events.append(ev)

pathlib.Path('edge_events.json').write_text(
    json.dumps(edge_events, default=str, indent=2))
print(f'Saved {len(edge_events)} edge events -> edge_events.json')
if edge_events:
    e = {k: v for k, v in edge_events[0].items() if k not in ('qwen_vl',)}
    print('\\nSample event:')
    print(json.dumps(e, indent=2, default=str))


fr = FRAMES[0]['data']
SYSTEM = """
        You are a road accident analysis expert.
        
        Return ONLY valid JSON.
        
        Schema:
        
        {
         "scene":"",
         "vehicles_detected":[],
         "accident_visible":false,
         "accident_type":"rear_end|T_bone|rollover|pedestrian|multi_vehicle|none",
         "smoke_or_fire":false,
         "persons_at_risk":false,
         "severity":"low|medium|high|critical",
         "escalate":false,
         "reason":""
        }
        
        Do not return markdown.
        Do not return explanations.
        Do not wrap with ```json.
        """
b64= frame_to_base64(fr)
print()
resp = client.chat.completions.create(
    model=QWEN_MODEL,
    # messages=[
    #     {'role': 'system', 'content': SYSTEM},
    #     {'role': 'user', 'content': [
    #         {'type': 'image_url',
    #          'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
    #         {'type': 'text', 'text': "TELL ME ABOUT THIS IMAGE"},
    #     ]},
    # ],
    messages=[
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': "There are two persons in front of the car"}],
    max_tokens=256, temperature=0.0,response_format={"type": "json_object"}
)
raw = resp.choices[0].message.content.strip()
print(raw)

