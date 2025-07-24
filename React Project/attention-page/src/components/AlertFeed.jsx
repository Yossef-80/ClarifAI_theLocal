import { useEffect, useState } from 'react';
import { socket } from '../socket'; // Shared socket instance

function AlertFeed() {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const handleAlert = (data) => {
      setAlerts((prev) => [...prev, data]);
    };

    socket.on('alert', handleAlert); // 👈 Listen on "alert" endpoint

    return () => {
      socket.off('alert', handleAlert); // 👈 Cleanup
    };
  }, []);

  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%' }}>
      <h5 className="p-2 m-0">🚨 Alerts</h5>
      {alerts.length === 0 ? (
        <div
          className="text-muted"
          style={{
            height: 'calc(100% - 40px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          No alerts yet.
        </div>
      ) : (
        <ul
          className="list-unstyled mb-0"
          style={{ height: 'calc(100% - 40px)', overflowY: 'auto' }}
        >
          {alerts.map((alert, idx) => (
            <li
              key={idx}
              className={`mb-2 p-2 rounded ${
                alert.type === 'success'
                  ? 'bg-success-subtle text-success'
                  : alert.type === 'warning'
                  ? 'bg-warning-subtle text-warning'
                  : 'bg-danger-subtle text-danger'
              }`}
            >
              <strong>[{alert.time}]</strong> {alert.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default AlertFeed;
