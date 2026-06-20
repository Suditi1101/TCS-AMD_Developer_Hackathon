import pandas as pd
from IPython.display import display

def create_dashboard(result):

    summary={

        "Videos":[len(result.get("video_metas",[]))],

        "Frames":[len(result.get("raw_frames",[]))],

        "Edge Events":[len(result.get("edge_events",[]))],

        "Correlated Events":[len(result.get("correlated_events",[]))],

        "Alerts":[len(result.get("alerts",[]))]
    }

    display(pd.DataFrame(summary))

    alerts=result.get("alerts",[])

    if alerts:

        display(pd.DataFrame(alerts))