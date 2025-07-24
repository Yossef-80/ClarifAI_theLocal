import VideoStream from './components/VideoStream';
import AlertFeed from './components/AlertFeed';
import TranscriptBox from './components/TranscriptBox';
import ClassMetrics from './components/ClassMetrics';

function App() {
  return (
    <div
      className="container-fluid p-0 m-0"
      style={{
        width: '100vw',
        height: '100vh',
        maxWidth: '100vw',
        maxHeight: '100vh',
        overflow: 'hidden',
      }}
    >
      <div className="row g-0 m-0" style={{ height: '100vh' }}>
        {/* Left column: 70% width */}
        <div
          className="col-12 col-lg-8 d-flex flex-column p-0 m-0"
          style={{ height: '100vh', maxHeight: '100vh' }}
        >
          <div
            style={{
              flex: '0 0 60%',
              height: '60%',
              minHeight: 0,
              maxHeight: '60%',
            }}
            className="d-flex flex-column"
          >
            <div style={{ flex: 1, minHeight: 0 }}>
              <VideoStream />
            </div>
          </div>
          <div
            style={{
              flex: '0 0 40%',
              height: '40%',
              minHeight: 0,
              maxHeight: '40%',
            }}
            className="d-flex flex-column"
          >
            <div style={{ flex: 1, minHeight: 0 }}>
              <TranscriptBox />
            </div>
          </div>
        </div>
        {/* Right column: 30% width */}
        <div
          className="col-12 col-lg-4 d-flex flex-column p-0 m-0"
          style={{ height: '100vh', maxHeight: '100vh' }}
        >
          <div
            style={{
              flex: '0 0 40%',
              height: '40%',
              minHeight: 0,
              maxHeight: '40%',
            }}
            className="d-flex flex-column"
          >
            <div style={{ flex: 1, minHeight: 0 }}>
              <AlertFeed />
            </div>
          </div>
          <div
            style={{
              flex: '0 0 60%',
              height: '60%',
              minHeight: 0,
              maxHeight: '60%',
            }}
            className="d-flex flex-column"
          >
            <div style={{ flex: 1, minHeight: 0 }}>
              <ClassMetrics />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
