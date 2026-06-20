import yaml, json, pathlib, time, uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI

cfg           = yaml.safe_load(pathlib.Path('config.yml').read_text())
VLLM_BASE_URL = cfg['vllm_base_url']
QWEN_MODEL    = 'Qwen3-30B-A3B'
client        = OpenAI(base_url=VLLM_BASE_URL, api_key='abc-123')

edge_events   = json.loads(pathlib.Path('state_jsons/edge_events.json').read_text())
DATASET_MODE  = 'accident_bench'

print(f'Loaded {len(edge_events)} edge events')
print(f'Videos seen: {sorted(set(e["video_id"] for e in edge_events))}')
acc_events = [e for e in edge_events if e.get('has_accident')]
print(f'Accident events: {len(acc_events)}')

BUDGET_WINDOW_MS = 50
BUDGET_RULES_MS  = 20
BUDGET_QWEN_MS   = 3000
BACKPRESSURE_LIMIT = 50


class BackpressureMonitor:
    def __init__(self, limit=BACKPRESSURE_LIMIT):
        self.limit    = limit
        self._history = deque(maxlen=100)
        self._throttles: List[dict] = []

    def check(self, depth):
        self._history.append((datetime.utcnow().isoformat(), depth))
        if depth > self.limit:
            print(f'  BACKPRESSURE: depth={depth} > {self.limit} -- signalling edge throttle')
            self._throttles.append({'ts': datetime.utcnow().isoformat(), 'depth': depth})
            return True
        return False

    def avg_depth(self):
        if not self._history:
            return 0.0
        return sum(d for _, d in self._history) / len(self._history)

    def summary(self):
        return {'throttle_events': len(self._throttles),
                'avg_depth': round(self.avg_depth(), 2)}


bp = BackpressureMonitor()
print('BackpressureMonitor ready')


class WindowGrouper:
    """
    Groups events by video_id into 5-second windows.
    Uses frame_no as a proxy for time (frame_no / target_fps = seconds).
    This ensures events from the same video and time period are correlated.
    """
    def __init__(self, window_s=5.0, target_fps=5):
        self.window_s  = window_s
        self.target_fps = target_fps

    def group(self, events):
        t0      = time.perf_counter()
        windows = defaultdict(list)
        for ev in events:
            vid    = ev.get('video_id', 'unknown')
            fno    = ev.get('frame_no', 0)
            # Window slot: group by video + 5-second bucket
            t_sec  = fno / max(self.target_fps, 1)
            slot   = int(t_sec // self.window_s)
            key    = f'{vid}_w{slot:04d}'
            windows[key].append(ev)
        ms = (time.perf_counter() - t0) * 1000
        if ms > BUDGET_WINDOW_MS:
            print(f'  WARN WindowGrouper {ms:.1f}ms')
        return dict(windows)




@dataclass
class CorrelatedEvent:
    correlation_id  : str
    event_type      : str
    severity        : str
    involved_sources: List[str]
    source_events   : List[dict]
    description     : str
    should_escalate : bool
    window_id       : str
    rule_matched    : str
    avg_confidence  : float
    frame_range     : List[int]
    dataset_mode    : str = 'accident_bench'
    qwen_reasoning  : Optional[str] = None


# Accident-bench correlation rules (ordered by priority)
ACCIDENT_RULES = [
    # Rule 1: Direct accident detection -- highest priority
    dict(name='accident_detected',
         labels=['accident'],
         min_count=1, min_conf=0.40,
         severity='critical', escalate=True),

    # Rule 2: Smoke detection -- post-crash indicator
    dict(name='smoke_detected',
         labels=['smoke'],
         min_count=1, min_conf=0.40,
         severity='high', escalate=True),

    # Rule 3: Vehicle + person in same window -- pedestrian risk
    dict(name='vehicle_person_proximity',
         labels=['car','truck','bus','motorcycle','bicycle','person'],
         requires_both=['person', ['car','truck','bus','motorcycle']],
         min_count=2, severity='high', escalate=True),

    # Rule 4: High vehicle density -- multi-vehicle incident
    dict(name='multi_vehicle_incident',
         labels=['car','truck','bus','motorcycle'],
         min_count=3, severity='medium', escalate=True),

    # Rule 5: Debris on road
    dict(name='road_debris',
         labels=['debris'],
         min_count=1, min_conf=0.40,
         severity='medium', escalate=True),

    # Rule 6: Post-impact freeze (high motion -> sudden drop)
    # Handled separately in engine via motion pattern analysis
]


class CorrelationEngine:
    def __init__(self):
        self.rules = ACCIDENT_RULES
        print(f'  CorrelationEngine: {len(self.rules)} accident rules loaded')

    def _get_frame_range(self, events):
        frames = [e.get('frame_no', 0) for e in events]
        return [min(frames), max(frames)] if frames else [0, 0]

    def _check_requires_both(self, matched_events, requires_both):
        """Check if both person AND vehicle are present."""
        if not requires_both:
            return True
        person_req, vehicle_req = requires_both[0], requires_both[1]
        has_person  = any(e.get('label') == person_req for e in matched_events)
        has_vehicle = any(e.get('label') in vehicle_req for e in matched_events)
        return has_person and has_vehicle

    def _detect_post_impact_freeze(self, events, window_id):
        """
        Detects post-impact freeze pattern:
        motion_score > 0.05 in early events, then drops to < 0.01.
        This is characteristic of a collision followed by vehicle stopping.
        """
        scores = [e.get('motion_score', 0) for e in events]
        if len(scores) < 4:
            return None
        mid    = len(scores) // 2
        early  = max(scores[:mid]) if scores[:mid] else 0
        late   = max(scores[mid:]) if scores[mid:] else 0
        if early > 0.05 and late < 0.01:
            fr = self._get_frame_range(events)
            return CorrelatedEvent(
                correlation_id   = str(uuid.uuid4()),
                event_type       = 'post_impact_freeze',
                severity         = 'high',
                involved_sources = list({e.get('video_id','?') for e in events}),
                source_events    = events,
                description      = f'Motion spike ({early:.3f}) followed by freeze ({late:.3f}) -- post-impact pattern',
                should_escalate  = True,
                window_id        = window_id,
                rule_matched     = 'post_impact_freeze',
                avg_confidence   = 0.75,
                frame_range      = fr,
            )
        return None

    def apply(self, window_id, events):
        t0     = time.perf_counter()
        by_lbl = defaultdict(list)
        for ev in events:
            by_lbl[ev.get('label','')].append(ev)

        found = []

        # Apply standard rules
        for rule in self.rules:
            matched = []
            for lbl in rule['labels']:
                sub = by_lbl.get(lbl, [])
                if 'min_conf' in rule:
                    sub = [e for e in sub if e.get('confidence', 0) >= rule['min_conf']]
                matched.extend(sub)

            if len(matched) < rule.get('min_count', 1):
                continue
            if 'requires_both' in rule:
                if not self._check_requires_both(matched, rule['requires_both']):
                    continue

            sources  = list({e.get('video_id','?') for e in matched})
            avg_conf = sum(e.get('confidence',0.5) for e in matched) / len(matched)
            fr       = self._get_frame_range(matched)
            found.append(CorrelatedEvent(
                correlation_id   = str(uuid.uuid4()),
                event_type       = rule['name'],
                severity         = rule['severity'],
                involved_sources = sources,
                source_events    = matched,
                description      = (f'{rule["name"]}: {len(matched)} detections '
                                    f'across frames {fr[0]}-{fr[1]}'),
                should_escalate  = rule['escalate'],
                window_id        = window_id,
                rule_matched     = rule['name'],
                avg_confidence   = round(avg_conf, 4),
                frame_range      = fr,
            ))

        # Post-impact freeze check
        freeze = self._detect_post_impact_freeze(events, window_id)
        if freeze:
            found.append(freeze)

        ms = (time.perf_counter() - t0) * 1000
        if ms > BUDGET_RULES_MS:
            print(f'  WARN CorrelationEngine {ms:.1f}ms')
        return found


engine = CorrelationEngine()
print('CorrelationEngine ready')


FOG_SYSTEM = (
    'You are the Fog Correlator Agent for SentinelFlow Accident Detection. '
    'You review road accident detection events that did NOT match any hard-coded rule. '
    'Identify novel accident patterns, assign severity, decide whether to escalate. '
    'Respond ONLY with valid JSON: '
    '{"pattern_detected": true/false, '
    '"pattern_name": "<short name or null>", '
    '"accident_type": "<rear_end|T_bone|rollover|pedestrian|multi_vehicle|unknown|none>", '
    '"severity": "low|medium|high|critical", '
    '"escalate": true/false, '
    '"reasoning": "<2 sentences>", '
    '"involved_labels": ["<label>"]}'
)


class QwenFogReasoner:
    def __init__(self, model=QWEN_MODEL):
        self.model = model

    def reason(self, window_id, events, matched_rules):
        summary = [{
            'label'      : e.get('label'),
            'confidence' : e.get('confidence'),
            'video_id'   : e.get('video_id'),
            'motion'     : e.get('motion_score', 0),
            'has_accident': e.get('has_accident', False),
            'frame_no'   : e.get('frame_no'),
        } for e in events]
        user_msg = (f'Window {window_id}. Events: {json.dumps(summary[:20])}\n'
                    f'Rules already matched: {matched_rules}\n'
                    'Identify any additional accident pattern not covered above.')
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'system', 'content': FOG_SYSTEM},
                          {'role': 'user',   'content': user_msg}],
                max_tokens=256, temperature=0.0,response_format={"type": "json_object"})
            raw = resp.choices[0].message.content.strip()
            if raw.startswith('```'):
                raw = '\\n'.join(l for l in raw.split('\\n')
                                if not l.strip().startswith('```')).strip()
            result = json.loads(raw)
        except Exception as exc:
            result = None
            print(f'  QwenFog error: {exc}')
        ms = (time.perf_counter() - t0) * 1000
        if ms > BUDGET_QWEN_MS:
            print(f'  WARN QwenFog {ms:.0f}ms')
        return result


fog_reasoner = QwenFogReasoner()
print('QwenFogReasoner ready')




def fog_correlator_node(state):
    
    edge_events=state.get("edge_events")    
    grouper = WindowGrouper()
    windows = grouper.group(edge_events)
    print(f'Formed {len(windows)} windows from {len(edge_events)} events')
    for wid, evs in list(windows.items())[:8]:
        labels = [e.get('label','?') for e in evs]
        has_acc = any(e.get('has_accident') for e in evs)
        print(f'  {wid}: {len(evs)} events  accident={has_acc}  labels={set(labels)}')

    all_correlated: List[CorrelatedEvent] = []

    print(f'Processing {len(windows)} windows...\\n')
    SEV_ICONS = {'low': 'LOW ', 'medium': 'MED ', 'high': 'HIGH', 'critical': 'CRIT'}
    
    for wid, w_events in windows.items():
        print(f'-- Window {wid} ({len(w_events)} events) --')
        bp.check(len(w_events))
    
        rule_matches  = engine.apply(wid, w_events)
        matched_names = [c.rule_matched for c in rule_matches]
    
        for c in rule_matches:
            icon = SEV_ICONS.get(c.severity, '????')
            print(f'  [{icon}] RULE: {c.event_type}  frames={c.frame_range}  '
                  f'conf={c.avg_confidence:.3f}  escalate={c.should_escalate}')
    
        qwen_result = fog_reasoner.reason(wid, w_events, matched_names)
        if qwen_result and qwen_result.get('pattern_detected'):
            sev  = qwen_result.get('severity', 'low')
            esc  = qwen_result.get('escalate', False)
            name = qwen_result.get('pattern_name', 'novel_pattern')
            print(f'  [QWEN] {name}  severity={sev}  escalate={esc}')
            print(f'         -> {qwen_result.get("reasoning","")}')
            if rule_matches:
                rule_matches[0].qwen_reasoning = qwen_result.get('reasoning')
            else:
                all_correlated.append(CorrelatedEvent(
                    correlation_id   = str(uuid.uuid4()),
                    event_type       = name or 'novel_accident_pattern',
                    severity         = sev,
                    involved_sources = list({e.get('video_id','?') for e in w_events}),
                    source_events    = w_events,
                    description      = qwen_result.get('reasoning', ''),
                    should_escalate  = esc,
                    window_id        = wid,
                    rule_matched     = 'qwen_novel',
                    avg_confidence   = (sum(e.get('confidence',0.5) for e in w_events)
                                        / max(len(w_events), 1)),
                    frame_range      = [w_events[0].get('frame_no',0),
                                        w_events[-1].get('frame_no',0)],
                    qwen_reasoning   = qwen_result.get('reasoning'),
                ))
        elif qwen_result:
            print('  [QWEN] no novel pattern')
    
        all_correlated.extend(rule_matches)
        print()
    
    escalate = [c for c in all_correlated if c.should_escalate]
    retain   = [c for c in all_correlated if not c.should_escalate]
    print(f'Total correlated : {len(all_correlated)}')
    print(f'Escalate->cloud  : {len(escalate)}')
    print(f'Retain locally   : {len(retain)}')
    print(f'Backpressure     : {bp.summary()}')
    
    
    payload = []
    for c in escalate:
        d = c.__dict__.copy()
        # Limit source_events size for JSON
        d['source_events'] = d['source_events'][:5]
        payload.append(d)
    
    pathlib.Path('state_jsons/correlated_events.json').write_text(
        json.dumps(payload, default=str, indent=2))
    print(f'Saved {len(payload)} escalated events -> correlated_events.json')

    state['fog_events'] = payload
    return state
