import cv2, os
from roboflow import Roboflow
# Your Roboflow info
os.environ["ROBOFLOW_API_KEY"] = "WoJAOshqna7GYr263JKf"
os.environ["ROBOFLOW_MODEL_ID"] = "faces-2-pcmbv"

# Initialize
rf = Roboflow(api_key="WoJAOshqna7GYr263JKf")
workspace = rf.workspace("youssef-cqeep")
project = workspace.project("classstudents-mwhin")

def extract_and_upload(video_path, duration=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret:
            print(f"❌ No frame at {sec}s")
            continue

        tmp = f"tmp_frame_{sec:03}.jpg"
        cv2.imwrite(tmp, frame)

        project.upload(
            image_path=tmp,
            split="train",
            batch_name="video_frames",
            num_retry_uploads=3
        )
        print(f"✅ Uploaded frame {sec}")

        os.remove(tmp)

    cap.release()

extract_and_upload("../Source Video/Real_Video/vid3.mp4", duration=20)