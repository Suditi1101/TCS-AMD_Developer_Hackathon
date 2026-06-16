
import cv2, pathlib

class VideoDecoder:
    def __init__(self,path):
        self.path=path
    def frames(self):
        cap=cv2.VideoCapture(str(self.path))
        while True:
            ok,frame=cap.read()
            if not ok: break
            yield frame
        cap.release()

def video_ingest_node(state):
    path=state.get("video_dir")
    state["frames"]=list(VideoDecoder(path).frames()) if path else []
    return state
