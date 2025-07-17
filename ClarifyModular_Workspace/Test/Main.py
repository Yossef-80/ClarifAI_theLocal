

from Capture import VideoCaptureFrameProvider
from Attention import YOLOAttentionModel, Attention
from Display import DisplayFrame
from Detection import YOLODetectionModel
video_path = "path/to/your/video.mp4"
weights_path = "../../yolo_attention_model/attention_weights_16july2025.pt"
face_weights_path = "../../yolo_attention_model/face_weights_16july2025.pt"
frame_provider = VideoCaptureFrameProvider(video_path)
yolo_model = YOLOAttentionModel(weights_path)
attention = Attention(frame_provider, yolo_model)
detection = YOLODetectionModel(face_weights_path)
display = DisplayFrame(frame_provider, attention)

display.show(verbose=False, imgsz=960)
