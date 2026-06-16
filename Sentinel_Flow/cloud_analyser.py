import yaml, json, pathlib, time, uuid
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI

cfg           = yaml.safe_load(pathlib.Path('config.yml').read_text())
VLLM_BASE_URL = cfg['vllm_base_url']
QWEN_MODEL    = 'Qwen3-30B-A3B'
client        = OpenAI(base_url=VLLM_BASE_URL, api_key='abc-123')

correlated    = json.loads(pathlib.Path('correlated_events.json').read_text())
DATASET_MODE  = 'accident_bench'

print(f'Loaded {len(correlated)} correlated events')
for c in correlated[:5]:
    print(f'  {c.get("event_type"):<30} severity={c.get("severity")}  '
          f'frames={c.get("frame_range")}')

BUDGET_EMBED_MS    = 50
BUDGET_SEARCH_MS   = 20
BUDGET_ANALYSIS_MS = 5000
ANOMALY_THRESHOLD  = 0.65
HITL_CONF_BAND     = (0.40, 0.70)
RETRAIN_ANOMALY_RATE = 0.30
GPU_AUTOSCALE_QUEUE  = 10


class EventEmbedder:
    """
    128-dim embedding from accident event metadata.
    Encodes: severity, vehicle count, accident type presence,
    frame density, confidence, source video.

    Production: use Qwen text embeddings via vLLM for richer representations.
    """
    DIM = 128
    SEV_MAP = {'low': 0.1, 'medium': 0.4, 'high': 0.75, 'critical': 1.0}
    TYPE_MAP = {
        'accident_detected': 0, 'smoke_detected': 1,
        'vehicle_person_proximity': 2, 'multi_vehicle_incident': 3,
        'road_debris': 4, 'post_impact_freeze': 5, 'qwen_novel': 6,
    }

    def __init__(self):
        rng = np.random.default_rng(99)
        self._type_vecs = {t: rng.standard_normal(self.DIM).astype(np.float32)
                           for t in self.TYPE_MAP}

    def embed(self, event):
        t0  = time.perf_counter()
        vec = np.zeros(self.DIM, dtype=np.float32)
        # Base type vector
        rule = event.get('rule_matched', 'qwen_novel')
        vec += self._type_vecs.get(rule, np.zeros(self.DIM))
        # Structural features
        vec[0] = self.SEV_MAP.get(event.get('severity', 'low'), 0.1)
        vec[1] = min(len(event.get('involved_sources', [])) / 4, 1.0)
        vec[2] = event.get('avg_confidence', 0.5)
        # Frame range density
        fr     = event.get('frame_range', [0, 0])
        vec[3] = min((fr[1] - fr[0]) / 100.0, 1.0)
        # Accident class flag
        vec[4] = 1.0 if 'accident' in event.get('event_type', '') else 0.0
        vec[5] = 1.0 if 'smoke' in event.get('event_type', '') else 0.0
        norm   = np.linalg.norm(vec)
        ms     = (time.perf_counter() - t0) * 1000
        return vec / (norm + 1e-9)


embedder = EventEmbedder()
print('EventEmbedder ready (128-dim, accident-bench aware)')


class VectorStore:
    """In-memory cosine similarity store. Production: Milvus or pgvector."""
    def __init__(self, max_size=100_000):
        self._ids:  List[str]        = []
        self._vecs: List[np.ndarray] = []
        self._meta: List[dict]       = []
        self._max = max_size

    def add(self, eid, vec, meta=None):
        if len(self._ids) >= self._max:
            self._ids.pop(0); self._vecs.pop(0); self._meta.pop(0)
        self._ids.append(eid)
        self._vecs.append(vec / (np.linalg.norm(vec) + 1e-9))
        self._meta.append(meta or {})

    def search(self, query, top_k=5):
        if not self._vecs:
            return []
        q    = query / (np.linalg.norm(query) + 1e-9)
        mat  = np.stack(self._vecs)
        sims = mat @ q
        top  = np.argsort(sims)[::-1][:top_k]
        return [(self._ids[i], float(sims[i]), self._meta[i]) for i in top]

    def __len__(self):
        return len(self._ids)


class AnomalyScorer:
    def __init__(self, store, top_k=5):
        self.store = store
        self.top_k = top_k

    def score(self, vec):
        neighbours = self.store.search(vec, self.top_k)
        if not neighbours:
            return 0.85
        avg_sim = sum(s for _, s, _ in neighbours) / len(neighbours)
        return round(float(np.clip((1.0 - avg_sim) / 2.0, 0.0, 1.0)), 4)


vector_store   = VectorStore()
anomaly_scorer = AnomalyScorer(vector_store)

# Seed baseline with common (non-critical) accident patterns
rng = np.random.default_rng(1)
for i in range(30):
    v    = rng.standard_normal(128).astype(np.float32)
    v[0] = 0.4   # medium severity baseline
    v[4] = 0.0   # not accident class (routine traffic)
    vector_store.add(str(uuid.uuid4()), v,
                     {'baseline': True, 'type': 'routine_traffic'})
print(f'VectorStore seeded with {len(vector_store)} baseline embeddings')


class HITLRouter:
    """
    Routes ambiguous accident frames to human analysts.
    For accident-bench: ambiguity arises from partial occlusion,
    night-time scenes, or near-miss events (no actual collision).

    Production: send frame_path links to Label Studio review task.
    """
    def __init__(self, conf_low=0.40, conf_high=0.70):
        self.conf_low  = conf_low
        self.conf_high = conf_high
        self._pending: List[dict] = []
        self._labels:  List[dict] = []

    def needs_review(self, event, anomaly_score):
        avg_conf = event.get('avg_confidence', 1.0)
        in_band  = self.conf_low <= avg_conf <= self.conf_high
        high_anomaly = anomaly_score >= ANOMALY_THRESHOLD * 1.3
        return in_band or high_anomaly

    def send_to_analyst(self, event, analysis):
        rid  = str(uuid.uuid4())[:8]
        task = {
            'review_id'    : rid,
            'event_type'   : event.get('event_type'),
            'severity'     : event.get('severity'),
            'frame_range'  : event.get('frame_range'),
            'accident_type': analysis.get('accident_type', 'unknown'),
            'anomaly_score': analysis.get('anomaly_score'),
            'summary'      : analysis.get('summary'),
            'analyst_hint' : ('Review frame sequence for actual collision vs near-miss. '
                              'Check for occlusion or camera artifacts.'),
            'queued_at'    : datetime.utcnow().isoformat(),
        }
        # Add frame paths for visual review
        src_events = event.get('source_events', [])
        task['frame_paths'] = [e.get('frame_path','') for e in src_events[:5]
                                if e.get('frame_path')]
        self._pending.append(task)
        print(f'  HITL: queued for analyst review  [rid={rid}]  '
              f'type={task["accident_type"]}')
        return rid

    def simulate_analyst_label(self, review_id, label, correct):
        result = {'review_id': review_id, 'analyst_label': label,
                  'correct': correct, 'labelled_at': datetime.utcnow().isoformat()}
        self._labels.append(result)
        return result

    @property
    def pending_count(self): return len(self._pending)

    @property
    def labels(self): return list(self._labels)


hitl = HITLRouter()
print('HITLRouter ready')




class GPUAutoscaler:
    def __init__(self, threshold=GPU_AUTOSCALE_QUEUE):
        self.threshold     = threshold
        self._current_gpus = 1
        self._events: List[dict] = []

    def check_and_scale(self, queue_depth):
        target = max(1, -(-queue_depth // self.threshold))
        if target != self._current_gpus:
            direction = 'SCALE UP' if target > self._current_gpus else 'SCALE DOWN'
            print(f'  GPU AUTOSCALE: {direction} {self._current_gpus}->{target} '
                  f'(queue={queue_depth})')
            self._events.append({'time': datetime.utcnow().isoformat(),
                                  'from': self._current_gpus, 'to': target})
            self._current_gpus = target
        return target


autoscaler = GPUAutoscaler()
print('GPUAutoscaler ready')

CLOUD_SYSTEM = (
    'You are the Cloud Analyser Agent for SentinelFlow Accident Detection. '
    'You receive an escalated road accident event with detection evidence. '
    'Provide thorough analysis. '
    'Respond ONLY with valid JSON: '
    '{"summary": "<2-3 sentence accident narrative>", '
    '"accident_type": "<rear_end|T_bone|rollover|pedestrian|multi_vehicle|unknown>", '
    '"vehicles_involved": <int>, '
    '"root_cause": "<1 sentence>", '
    '"recommended_action": "<specific emergency/operational action>", '
    '"confidence_assessment": "<1 sentence on detection reliability>", '
    '"retrain_flag": true/false, '
    '"retrain_reason": "<1 sentence or null>", '
    '"anomaly_explanation": "<why novel or common pattern>"}'
)


class QwenDeepAnalyser:
    def __init__(self, model=QWEN_MODEL):
        self.model = model

    def analyse(self, event, anomaly_score, similar_ids):
        src_events   = event.get('source_events', [])
        label_counts = {}
        for e in src_events:
            lbl = e.get('label','?')
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        prompt = (
            f'Event type      : {event.get("event_type")}\\n'
            f'Severity        : {event.get("severity")}\\n'
            f'Frame range     : {event.get("frame_range")}\\n'
            f'Videos          : {event.get("involved_sources")}\\n'
            f'Avg confidence  : {event.get("avg_confidence",0):.3f}\\n'
            f'Anomaly score   : {anomaly_score:.4f}\\n'
            f'Label counts    : {label_counts}\\n'
            f'Similar past    : {len(similar_ids)} events found\\n'
            f'Rule matched    : {event.get("rule_matched")}\\n'
            f'Qwen fog note   : {event.get("qwen_reasoning","none")}\\n'
            'Perform full accident analysis.'
        )
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'system', 'content': CLOUD_SYSTEM},
                          {'role': 'user',   'content': prompt}],
                max_tokens=512, temperature=0.1,response_format={"type": "json_object"})
            raw = resp.choices[0].message.content.strip()
            if raw.startswith('```'):
                raw = '\\n'.join(l for l in raw.split('\\n')
                                if not l.strip().startswith('```')).strip()
            result = json.loads(raw)
        except Exception as exc:
            result = {'summary': f'Analysis failed: {exc}',
                      'accident_type': 'unknown', 'vehicles_involved': 0,
                      'recommended_action': 'Manual review required',
                      'retrain_flag': False, 'retrain_reason': None,
                      'anomaly_explanation': '', 'root_cause': '',
                      'confidence_assessment': ''}
        ms = (time.perf_counter() - t0) * 1000
        if ms > BUDGET_ANALYSIS_MS:
            print(f'  WARN DeepAnalysis {ms:.0f}ms')
        return result


analyser = QwenDeepAnalyser()
print('QwenDeepAnalyser ready')


final_results = []
retrain_flags = []

autoscaler.check_and_scale(len(correlated))
print(f'Analysing {len(correlated)} escalated events...\\n')

for i, event in enumerate(correlated):
    sev = event.get('severity','low')
    print(f'-- Event {i+1}/{len(correlated)} [{sev.upper():<8}] '
          f'{event.get("event_type")} frames={event.get("frame_range")} --')

    vec      = embedder.embed(event)
    similar  = vector_store.search(vec, top_k=5)
    sim_ids  = [s[0] for s in similar]
    a_score  = anomaly_scorer.score(vec)
    a_flag   = 'ANOMALY' if a_score >= ANOMALY_THRESHOLD else 'normal '
    print(f'  Anomaly score  : {a_score:.4f}  [{a_flag}]')

    vector_store.add(event.get('correlation_id', str(uuid.uuid4())), vec,
                     {'event_type': event.get('event_type'),
                      'severity':   event.get('severity')})

    analysis = analyser.analyse(event, a_score, sim_ids)
    analysis['anomaly_score']    = a_score
    analysis['similar_ids']      = sim_ids
    analysis['analysis_id']      = str(uuid.uuid4())
    analysis['correlation_id']   = event.get('correlation_id')
    analysis['frame_range']      = event.get('frame_range')
    analysis['involved_sources'] = event.get('involved_sources')

    print(f'  Accident type  : {analysis.get("accident_type")}')
    print(f'  Vehicles       : {analysis.get("vehicles_involved")}')
    print(f'  Summary        : {str(analysis.get("summary",""))[:120]}')
    print(f'  Action         : {str(analysis.get("recommended_action",""))[:100]}')

    if analysis.get('retrain_flag'):
        print(f'  RETRAIN: {analysis.get("retrain_reason","")}')
        retrain_flags.append(analysis)

    if hitl.needs_review(event, a_score):
        rid   = hitl.send_to_analyst(event, analysis)
        label = hitl.simulate_analyst_label(rid, event.get('event_type','?'), True)

    final_results.append({'event': event, 'analysis': analysis})
    print()

anomaly_rate = (sum(1 for r in final_results
                    if r['analysis'].get('anomaly_score',0) >= ANOMALY_THRESHOLD)
                / max(len(final_results), 1))

print('=' * 60)
print(f'Events analysed     : {len(final_results)}')
print(f'Anomaly rate        : {anomaly_rate*100:.0f}%')
print(f'HITL pending        : {hitl.pending_count}')
print(f'Retrain flagged     : {len(retrain_flags)}')
print(f'Vector store size   : {len(vector_store)}')

if anomaly_rate > RETRAIN_ANOMALY_RATE or retrain_flags:
    print()
    print('CONTINUAL LEARNING TRIGGER: anomaly rate exceeded threshold')
    print('  -> Submit YOLOv8 fine-tuning job with HITL-labelled frames')
    print(f'  -> {len(hitl.labels)} analyst labels available')


videos_seen = set()
for r in final_results:
    for src in r['event'].get('involved_sources', []):
        videos_seen.add(src)

edge_count = len(json.loads(pathlib.Path('edge_events.json').read_text()))

print()
print('=' * 60)
print('  SENTINELFLOW v3 - ACCIDENT DETECTION PIPELINE REPORT')
print('=' * 60)
print(f'  Dataset          : accident-bench')
print(f'  Videos processed : {len(videos_seen)}')
print()
print(f'  [TIER 1 EDGE]')
print(f'    Edge events emitted  : {edge_count}')
print()
print(f'  [TIER 2 FOG]')
print(f'    Correlated events    : {len(correlated)}')
print()
print(f'  [TIER 3 CLOUD]')
print(f'    Events analysed      : {len(final_results)}')
print(f'    Anomaly rate         : {anomaly_rate*100:.0f}%')
print(f'    HITL pending         : {hitl.pending_count}')
print(f'    Retrain triggered    : {anomaly_rate > RETRAIN_ANOMALY_RATE}')
print()
for r in final_results:
    a = r['analysis']
    print(f'  [{a.get("accident_type","?"):<15}] '
          f'sev={r["event"].get("severity"):<8} '
          f'anom={a.get("anomaly_score",0):.3f}  '
          f'frames={r["event"].get("frame_range")}')
"""))

c03.append(md("## 8. Save results"))
c03.append(code("""
pathlib.Path('cloud_results.json').write_text(
    json.dumps(final_results, default=str, indent=2))
pathlib.Path('hitl_labels.json').write_text(
    json.dumps(hitl.labels, default=str, indent=2))
print(f'Saved cloud_results.json  ({len(final_results)} entries)')
print(f'Saved hitl_labels.json    ({len(hitl.labels)} analyst labels)')
print()
print('Pipeline complete -> see 04_orchestrator.ipynb')


