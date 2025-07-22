import cv2

class VideoCaptureFrameProvider:
    """
    Frame provider that captures frames from a video file using OpenCV.
    Communicates with other modules through getters.
    """
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Store video properties
        self._frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Current frame state
        self._current_frame = None
        self._current_frame_number = 0
        self._is_reading = True

    def get_frame(self):
        """
        Get the next frame from the video.
        Returns:
            frame: Next frame or None if end of video
        """
        if not self._is_reading:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            self._is_reading = False
            return None
        
        self._current_frame = frame
        self._current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return frame

    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._current_frame_number = 0
        self._is_reading = True

    def seek_frame(self, frame_number):
        """
        Seek to a specific frame number.
        Args:
            frame_number: Frame number to seek to
        """
        if 0 <= frame_number < self._frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame_number = frame_number
            self._is_reading = True

    def release(self):
        """Release the video capture."""
        self._is_reading = False
        self.cap.release()

    # ========== GETTERS ==========
    
    @property
    def frame_width(self):
        """Get the frame width."""
        return self._frame_width
    
    @property
    def frame_height(self):
        """Get the frame height."""
        return self._frame_height
    
    @property
    def fps(self):
        """Get the frames per second."""
        return self._fps
    
    @property
    def frame_count(self):
        """Get the total number of frames."""
        return self._frame_count
    
    @property
    def current_frame(self):
        """Get the current frame."""
        return self._current_frame
    
    @property
    def current_frame_number(self):
        """Get the current frame number."""
        return self._current_frame_number
    
    @property
    def is_reading(self):
        """Check if currently reading frames."""
        return self._is_reading
    
    @property
    def video_properties(self):
        """Get all video properties as a dictionary."""
        return {
            'width': self._frame_width,
            'height': self._frame_height,
            'fps': self._fps,
            'total_frames': self._frame_count,
            'current_frame': self._current_frame_number
        }
    
    def get_progress(self):
        """Get video progress as percentage."""
        if self._frame_count > 0:
            return (self._current_frame_number / self._frame_count) * 100
        return 0
