import { useEffect, useState } from 'react';
import { socket } from '../socket'; // Shared Socket.IO client instance

function VideoStream() {
  const [frame, setFrame] = useState(null);

  useEffect(() => {
    const handleFrame = (data) => {
      if (data.frame) setFrame(data.frame);
    };

    socket.on('video_frame', handleFrame); // 👈 Listen to video_frame event

    return () => {
      socket.off('video_frame', handleFrame); // 👈 Clean up on unmount
    };
  }, []);

  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%' }}>
      <h5 className="p-2 m-0">🎥 Live Video Stream</h5>
      {frame ? (
        <img
          src={`data:image/jpeg;base64,${frame}`}
          alt="Live video"
          style={{
            width: '100%',
            height: 'calc(100% - 40px)',
            objectFit: 'contain',
            background: '#000',
          }}
        />
      ) : (
        <div
          style={{
            width: '100%',
            height: 'calc(100% - 40px)',
            background: '#000',
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          Waiting for video...
        </div>
      )}
    </div>
  );
}

export default VideoStream;
