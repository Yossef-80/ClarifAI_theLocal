import { useEffect, useState } from 'react';
import socket from '../socket';

function TranscriptBox() {
  const [transcripts, setTranscripts] = useState([]);

  useEffect(() => {
    const handleTranscription = (data) => {
      setTranscripts((prev) => [...prev, data]);
    };
    socket.on('transcription', handleTranscription);
    return () => socket.off('transcription', handleTranscription);
  }, []);

  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%' }}>
      <h5 className="p-2 m-0">📝 Live Transcription</h5>
      {transcripts.length === 0 ? (
        <div className="text-muted" style={{ height: 'calc(100% - 40px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No transcriptions yet.</div>
      ) : (
        <ul className="list-unstyled mb-0" style={{ height: 'calc(100% - 40px)', overflowY: 'auto' }}>
          {transcripts.map((item, idx) => (
            <li
              key={idx}
              className={`mb-2 p-2 rounded ${
                item.score >= 75
                  ? 'bg-success-subtle text-success'
                  : item.score >= 60
                  ? 'bg-warning-subtle text-warning'
                  : 'bg-danger-subtle text-danger'
              }`}
            >
              <strong>[{item.time}]</strong> {item.text}
              <span className="float-end fw-bold">{item.score}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default TranscriptBox; 