function VideoStream() {
  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%' }}>
      <h5 className="p-2 m-0">🎥 Live Video Stream</h5>
      <img
        src="http://localhost:8000/video_feed"
        alt="Live video"
        style={{ width: '100%', height: 'calc(100% - 40px)', objectFit: 'contain', background: '#000' }}
      />
    </div>
  );
}
export default VideoStream;