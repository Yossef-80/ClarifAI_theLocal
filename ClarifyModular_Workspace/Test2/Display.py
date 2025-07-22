import cv2

class DisplayFrame:
    """
    Class to display frames with annotations from any module.
    Communicates with other modules through getters.
    """
    def __init__(self, window_name="Display Frame"):
        self._window_name = window_name
        self._current_frame = None
        self._current_annotated_frame = None
        self._current_results = None
        self._is_displaying = False

    def display_frame(self, frame, results=None, color_map=None):
        """
        Display a frame with optional annotations.
        Args:
            frame: Input frame to display
            results: Optional results to annotate (list of dicts with bbox, label, etc.)
            color_map: Optional color mapping for different labels
        """
        self._current_frame = frame.copy()
        self._current_results = results
        
        if results is not None:
            self._current_annotated_frame = self._annotate_frame(frame.copy(), results, color_map)
        else:
            self._current_annotated_frame = frame.copy()
        
        cv2.imshow(self._window_name, self._current_annotated_frame)
        return cv2.waitKey(1) & 0xFF

    def _annotate_frame(self, frame, results, color_map=None):
        """Annotate frame with results."""
        if color_map is None:
            color_map = {"default": (0, 255, 0)}
        
        for result in results:
            if 'bbox' in result:
                x1, y1, x2, y2 = result['bbox']
                label = result.get('label', '')
                color = color_map.get(result.get('type', 'default'), (0, 255, 0))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if label:
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
        
        return frame

    def display_continuous(self, frame_provider, processor=None, verbose=False):
        """
        Display frames continuously from a frame provider.
        Args:
            frame_provider: Object with get_frame() method
            processor: Optional processor object with process_frame() method
            verbose: Verbose output
        """
        self._is_displaying = True
        
        while self._is_displaying:
            frame = frame_provider.get_frame()
            if frame is None:
                break
            
            if processor is not None:
                # Use processor if provided
                if hasattr(processor, 'process_frame'):
                    processed_frame, results = processor.process_frame(frame)
                else:
                    processed_frame = frame
                    results = None
            else:
                processed_frame = frame
                results = None
            
            key = self.display_frame(processed_frame, results)
            if key == ord('q'):
                break
        
        self.close()

    def close(self):
        """Close the display window."""
        self._is_displaying = False
        cv2.destroyAllWindows()

    # ========== GETTERS ==========
    
    @property
    def current_frame(self):
        """Get the current frame."""
        return self._current_frame
    
    @property
    def current_annotated_frame(self):
        """Get the current annotated frame."""
        return self._current_annotated_frame
    
    @property
    def current_results(self):
        """Get the current results."""
        return self._current_results
    
    @property
    def is_displaying(self):
        """Check if currently displaying."""
        return self._is_displaying
    
    @property
    def window_name(self):
        """Get the window name."""
        return self._window_name
