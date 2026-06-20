
from config_loader import qwen_json

def alert_learn_node(state):
    # print('[alert_learn] routing alerts...')
    # t0      = time.perf_counter()
    # results = state.get('analysed_events', [])
    # events  = state.get('fog_events', [])
    # acc_sum = state.get('accident_summary', {})

    # prompt = (f'Accident-bench results ({len(results)}): '
    #           f'{json.dumps(results, default=str)[:1500]}\\n'
    #           f'Severities: {[e.get("severity") for e in events]}\\n'
    #           f'Accident summary: {acc_sum}\\n'
    #           f'Retrain triggered: {state.get("cloud_stats",{}).get("retrain_triggered",False)}')
    # plan = qwen_json(wf_loader.get_system_prompt('alert_learn'), prompt, max_tokens=512)

    # ms = (time.perf_counter() - t0) * 1000
    # state['alerts_sent']       = plan.get('alerts_sent', [])
    # state['notifications']     = plan.get('notifications', [])
    # state['retrain_submitted'] = plan.get('retrain_submitted', False)
    # state['hitl_pending']      = plan.get('hitl_pending', 0)
    # completed = list(state.get('completed_nodes',[])); completed.append('alert_learn')
    # state['completed_nodes'] = completed
    # log = list(state.get('latency_log',[])); log.append({'node':'alert_learn','ms':round(ms,1)})
    # state['latency_log'] = log
    # print(f'  OK {len(plan.get("notifications",[]))} notifications  '
    #       f'retrain={plan.get("retrain_submitted")}  {ms:.0f}ms')
    # CHAN_MAP = {'webhook':'[WH]','slack':'[SL]','dashboard':'[DB]','log':'[LG]'}
    # for n in plan.get('notifications', []):
    #     print(f'    {CHAN_MAP.get(n.get("channel",""),"[?]")} '
    #           f'[{n.get("severity","?").upper():<8}] '
    #           f'{str(n.get("message",""))[:80]}')
    # return state
    prompt=f"Route these alerts for action: {state.get('alerts',[])}"
    try:
        state["routing"]=qwen_json(prompt)
    except Exception:
        state["routing"]={"action":"human_review"}
    return state
