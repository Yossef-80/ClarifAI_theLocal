import cv2

class DisplayFrame:
    """
    Class to display frames processed by the attention model, with bounding boxes drawn.
    """
    def __init__(self, frame_provider, attention_model, window_name="Processed Frame"):
        self.frame_provider = frame_provider
        self.attention_model = attention_model
        self.window_name = window_name
        self._current_frame = None
        self._current_boxes = None

    def get_current_frame(self):
        """Returns the current frame (with boxes drawn if available)."""
        return self._current_frame

    def get_current_boxes(self):
        """Returns the current detection results (boxes)."""
        return self._current_boxes

    

    def show(self, verbose=False, imgsz=960):
        while True:
            frame = self.frame_provider.get_frame()
            if frame is None:
                break  # End of video

            boxes, classes, confidences = self.attention_model.process(verbose=verbose, imgsz=imgsz)
            self._current_boxes = boxes
            self._current_classes = classes
            self._current_confidences = confidences

            # Draw boxes on a copy of the frame
            frame_with_boxes = frame.copy()
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Add label class and confidence if available
                    label = ""
                    if classes is not None and i < len(classes):
                        class_id = int(classes[i])
                        # Try to get class name from the model if available
                        class_name = None
                        if hasattr(self.attention_model.yolo_model.model, 'names'):
                            names = self.attention_model.yolo_model.model.names
                            if isinstance(names, dict):
                                class_name = names.get(class_id, str(class_id))
                            elif isinstance(names, list) and class_id < len(names):
                                class_name = names[class_id]
                        if class_name is not None:
                            label += f"{class_name} "
                        else:
                            label += f"Class: {class_id} "
                    if confidences is not None and i < len(confidences):
                        label += f"Conf: {confidences[i]:.2f}"
                    if label:
                        cv2.putText(
                            frame_with_boxes,
                            label,
                            (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
            self._current_frame = frame_with_boxes

            cv2.imshow(self.window_name, frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.close()

    def close(self):
        self.frame_provider.release()
        cv2.destroyAllWindows()
