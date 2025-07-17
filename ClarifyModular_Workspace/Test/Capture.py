import cv2




class VideoCaptureFrameProvider:
    """
    Frame provider that captures frames from a video file using OpenCV.
    Provides getters for frame width, height, and fps.
    """
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None  # End of video or error
        return frame

    def get_frame_width(self):
        return self._frame_width

    def get_frame_height(self):
        return self._frame_height

    def get_fps(self):
        return self._fps

    def release(self):
        self.cap.release()
