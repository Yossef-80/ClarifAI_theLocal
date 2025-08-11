from moviepy import VideoFileClip

# Load the original clip
clip = VideoFileClip("../../Source Video/combined videos.mp4")

# Calculate new duration for 1.5x speed
new_duration = clip.duration / 1.5

# Apply new duration
faster_clip = clip.with_duration(new_duration)

# Write output
faster_clip.write_videofile("output_1.5x.mp4", codec="libx264", audio_codec="aac")
