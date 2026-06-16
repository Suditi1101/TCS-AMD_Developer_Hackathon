
import yaml, json, os
from typing import TypedDict, Any
from openai import OpenAI

class PipelineState(TypedDict, total=False):
    video_dir: str
    frames: list
    edge_events: list
    fog_events: list
    analysed_events: list
    alerts: list
    metadata: dict

class WorkflowLoader:
    def __init__(self, path="workflow.yml"):
        with open(path) as f:
            self.config=yaml.safe_load(f)

client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL","http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY","EMPTY")
)

def qwen_json(prompt, model="Qwen3-30B-A3B"):
    r=client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"}
    )
    return json.loads(r.choices[0].message.content)
