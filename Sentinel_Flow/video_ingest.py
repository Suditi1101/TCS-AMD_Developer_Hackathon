
import cv2, pathlib
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

lat_monitor = LatencyMonitor()
print('LatencyMonitor ready')

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

    def __init__(self, video_dir=VIDEO_DIR, frames_out=FRAMES_OUT,target_fps=TARGET_FPS, frame_size=FRAME_SIZE):
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

    def frames(self):
        cap=cv2.VideoCapture(str(self.path))
        while True:
            ok,frame=cap.read()
            if not ok: break
            yield frame
        cap.release()
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


def video_ingest_node(state):
    
    path=state.get("video_dir")
    print('Step 0: Decoding videos...')
    t_decode = time.perf_counter()
    
    # max_videos=None processes all videos in VIDEO_DIR
    # Set max_videos=2 for a quick test run
    decoder = VideoDecoder()
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

    state["frames"]=list(FRAMES) if path else []
    return state
