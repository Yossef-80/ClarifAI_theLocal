import { useEffect, useState } from 'react';
import socket from '../socket';

function ClassMetrics() {
  const [metrics, setMetrics] = useState({ attention: 0, comprehension: 0, active: 0 });

  useEffect(() => {
    socket.on('classroom_metrics', (data) => {
      console.log("Received classroom_metrics:", data);
      setMetrics(data);
    });
    return () => socket.off('classroom_metrics');
  }, []);

  return (
    <div className="card h-100 w-100 p-0" style={{ height: '100%', width: '100%', background: 'linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%)', border: 'none', boxShadow: '0 4px 24px rgba(0,0,0,0.08)' }}>
      <div className="h-100 d-flex flex-column justify-content-center align-items-center p-3">
        <h4 className="fw-bold mb-4" style={{ letterSpacing: 1, color: '#2d3748' }}>📊 Class Metrics</h4>
        <div className="w-100 mb-3">
          <div className="d-flex justify-content-between align-items-center mb-1">
            <span className="fw-semibold" style={{ color: '#3b82f6' }}>Attention</span>
            <span className="fw-bold" style={{ color: '#3b82f6' }}>{metrics.attention}%</span>
          </div>
          <div className="progress" style={{ height: 12, background: '#e5e7eb' }}>
            <div className="progress-bar" role="progressbar" style={{ width: `${metrics.attention}%`, background: 'linear-gradient(90deg, #60a5fa 0%, #2563eb 100%)' }} aria-valuenow={metrics.attention} aria-valuemin={0} aria-valuemax={100}></div>
          </div>
        </div>
        <div className="w-100 mb-3">
          <div className="d-flex justify-content-between align-items-center mb-1">
            <span className="fw-semibold" style={{ color: '#10b981' }}>Comprehension</span>
            <span className="fw-bold" style={{ color: '#10b981' }}>{metrics.comprehension}%</span>
          </div>
          <div className="progress" style={{ height: 12, background: '#e5e7eb' }}>
            <div className="progress-bar" role="progressbar" style={{ width: `${metrics.comprehension}%`, background: 'linear-gradient(90deg, #6ee7b7 0%, #059669 100%)' }} aria-valuenow={metrics.comprehension} aria-valuemin={0} aria-valuemax={100}></div>
          </div>
        </div>
        <div className="w-100 mt-4 d-flex flex-column align-items-center">
          <span className="fw-semibold text-secondary" style={{ fontSize: 18 }}>Active Students</span>
          <span className="fw-bold" style={{ fontSize: 36, color: '#6366f1', letterSpacing: 2 }}>{metrics.active}</span>
        </div>
      </div>
    </div>
  );
}

export default ClassMetrics;
