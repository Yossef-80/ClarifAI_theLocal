import cv2

class DisplayManager:
    def __init__(self, color_map):
        self.color_map = color_map

    def draw(self, frame, tracked_faces, attention_labels, attention_model_names):
        for (matched_id, x1, y1, x2, y2, name, conf), attn_label in zip(tracked_faces, attention_labels):
            label = f'{name} ({matched_id}) {conf:.2f} | {attn_label}'
            color = self.color_map.get(attn_label, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
