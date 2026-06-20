
from langgraph.graph import StateGraph, END
from config_loader import PipelineState
from video_ingest import video_ingest_node
from edge_filter import edge_filter_node
from fog_correlator import fog_correlator_node
from cloud_analyser import cloud_analyser_node
from alert_learn import alert_learn_node
from dashboard import create_dashboard

def orchestrator_node(state):
    return state

class GraphBuilder:
    def build(self):
        g=StateGraph(PipelineState)

        g.add_node("orchestrator", orchestrator_node)
        g.add_node("video_ingest", video_ingest_node)
        g.add_node("edge_filter", edge_filter_node)
        g.add_node("fog_correlator", fog_correlator_node)
        g.add_node("cloud_analyser", cloud_analyser_node)
        g.add_node("alert_learn", alert_learn_node)

        g.set_entry_point("orchestrator")

        g.add_edge("orchestrator","video_ingest")
        g.add_edge("video_ingest","edge_filter")
        g.add_edge("edge_filter","fog_correlator")
        g.add_edge("fog_correlator","cloud_analyser")
        g.add_edge("cloud_analyser","alert_learn")
        g.add_edge("alert_learn",END)

        return g.compile()

def main():
    app=GraphBuilder().build()
    result=app.invoke({
        "video_dir":"sample.mp4",
        "frames":[],
        "alerts":[]
    })
 #   print(result)
    create_dashboard(result)

if __name__=="__main__":
    main()
