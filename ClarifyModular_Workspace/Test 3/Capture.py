import cv2

class VideoCaptureHandler:
    def __init__(self, video_path, output_path, frame_width, frame_height, fps):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        # self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    def read(self):
        return self.cap.read()

    # def write(self, frame):
        # self.out.write(frame)

    def release(self):
        self.cap.release()
        # self.out.release()

    def get_frame_size(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame_idx(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
