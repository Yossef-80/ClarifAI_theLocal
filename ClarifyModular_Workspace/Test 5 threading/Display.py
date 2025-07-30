import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class DisplayManager:
    def __init__(self, color_map):
        self.color_map = color_map

    def draw(self, frame, tracked_faces, attention_labels, attention_model_names):
        if tracked_faces is None:
            return frame
        if attention_labels is None or len(attention_labels) == 0:
            # Just draw tracked faces without attention labels
            for (matched_id, x1, y1, x2, y2, name, conf) in tracked_faces:
                label = f'{name} ({matched_id}) {conf:.2f}'
                color = (255, 255, 255)  # white
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Draw with attention labels if available
            if len(tracked_faces) != len(attention_labels):
                print(f"Warning: tracked_faces ({len(tracked_faces)}) != attention_labels ({len(attention_labels)})")
                # Just draw tracked faces without attention labels
                for (matched_id, x1, y1, x2, y2, name, conf) in tracked_faces:
                    label = f'{name} ({matched_id}) {conf:.2f}'
                    color = (255, 255, 255)  # white
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Both lists have same length, can zip them
                for (matched_id, x1, y1, x2, y2, name, conf), attn_label in zip(tracked_faces, attention_labels):
                    label = f'{name} ({matched_id}) {conf:.2f} | {attn_label}'
                    color = self.color_map.get(attn_label, (128, 128, 128))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

class DisplayWorker(QThread):
    display_done = pyqtSignal(object)  # emits the processed frame (numpy array)
    def __init__(self, display_manager):
        super().__init__()
        self.display_manager = display_manager
        self.frame = None
        self.tracked_faces = None
        self.attention_labels = None
        self.attention_model_names = None
        self.running = True

    def set_data(self, frame, tracked_faces, attention_labels, attention_model_names):
        self.frame = frame
        self.tracked_faces = tracked_faces
        self.attention_labels = attention_labels
        self.attention_model_names = attention_model_names

    def run(self):
        while self.running:
            if self.frame is not None and self.tracked_faces is not None and self.attention_labels is not None and self.attention_model_names is not None:
                # Always use a copy!
                processed_frame = self.display_manager.draw(self.frame.copy(), self.tracked_faces, self.attention_labels, self.attention_model_names)
                self.display_done.emit(processed_frame)
                self.frame = None
                self.tracked_faces = None
                self.attention_labels = None
                self.attention_model_names = None
            self.msleep(10)

    def stop(self):
        self.running = False
        self.wait()
