import { useEffect, useState } from 'react';
import socket from '../socket';

function VideoStream() {
  const [frame, setFrame] = useState(null);

  useEffect(() => {
    socket.on('video_frame_data', (data) => {
      console.log("Received video_frame_data", data);
      setFrame(data.frame);
    });
    socket.emit('start_stream');
    return () => socket.off('video_frame_data');
  }, []);

  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%' }}>
      <h5 className="p-2 m-0">🎥 Live Video Stream</h5>
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Live video"
          style={{ width: '100%', height: 'calc(100% - 40px)', objectFit: 'contain', background: '#000' }}
        />
      ) : (
        <div style={{ width: '100%', height: 'calc(100% - 40px)', background: '#000', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          Waiting for video...
        </div>
      )}
    </div>
  );
}

export default VideoStream;