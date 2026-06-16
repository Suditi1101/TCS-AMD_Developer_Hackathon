
from config_loader import qwen_json

def alert_learn_node(state):
    prompt=f"Route these alerts for action: {state.get('alerts',[])}"
    try:
        state["routing"]=qwen_json(prompt)
    except Exception:
        state["routing"]={"action":"human_review"}
    return state
