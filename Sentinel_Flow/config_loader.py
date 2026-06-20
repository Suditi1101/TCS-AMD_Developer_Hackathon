from dataclasses import dataclass
import yaml, json, os
from typing import TypedDict, Any, Dict, List, Optional
from openai import OpenAI
import yaml, pathlib

cfg = yaml.safe_load(pathlib.Path('config.yml').read_text())
VLLM_BASE_URL = cfg['vllm_base_url']

    
class PipelineState(TypedDict, total=False):
    video_dir: str
    frames: list[Dict]
    edge_events: list[Dict]
    fog_events: list[Dict]
    analysed_events: list
    alerts: list
    metadata: dict

class WorkflowLoader:
    def __init__(self, path="workflow.yml"):
        with open(path) as f:
            self.config=yaml.safe_load(f)

client = OpenAI(base_url=VLLM_BASE_URL, api_key="abc-123")

def qwen_json(prompt, model="Qwen3-30B-A3B"):
    resp = client.chat.completions.create(
                model=self.model,
                # messages=[
                #     {'role': 'system', 'content': self.SYSTEM},
                #     {'role': 'user', 'content': [
                #         {'type': 'image_url',
                #          'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                #         {'type': 'text', 'text': prompt},
                #     ]},
                # ],
                messages=[
                    {'role': 'user', 'content': prompt}],
                max_tokens=256, temperature=0.0,response_format={"type": "json_object"}
            )
    return json.loads(resp.choices[0].message.content)
