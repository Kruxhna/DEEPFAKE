import React, { useState, useRef, useEffect } from 'react';
import { Shield, Info, Github, ExternalLink, RefreshCw } from 'lucide-react';
import UploadZone from './components/upload/UploadZone';
import VerdictCard from './components/analysis/VerdictCard';
import ProbabilityChart from './components/charts/ProbabilityChart';
import SuspiciousGallery from './components/analysis/SuspiciousGallery';
import { analyzeVideo, analyzeImage, checkHealth } from './services/api';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [systemOnline, setSystemOnline] = useState(false);

  const videoRef = useRef(null);

  useEffect(() => {
    // Check if backend is running
    checkHealth()
      .then(() => setSystemOnline(true))
      .catch(() => setSystemOnline(false));
  }, []);

  const handleFileSelect = async (selectedFile) => {
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setFileUrl(url);
    const isVideo = selectedFile.type.startsWith('video/');
    setFileType(isVideo ? 'video' : 'image');

    setIsLoading(true);
    setResults(null);

    try {
      let data;
      if (isVideo) {
        data = await analyzeVideo(selectedFile);
      } else {
        data = await analyzeImage(selectedFile);
      }
      setResults(data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please ensure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileUrl(null);
    setResults(null);
    setFileType(null);
  };

  const handleTimeUpdate = () => {
    if (videoRef.current && results?.frame_predictions) {
      const time = videoRef.current.currentTime;
      const duration = videoRef.current.duration;
      // Map time to frame number based on total frames in results
      const totalFrames = results.total_frames || 100;
      const currentFrame = (time / duration) * totalFrames;
      setCurrentTime(currentFrame);
    }
  };

  const handleChartSeek = (frameIndex) => {
    if (videoRef.current && results) {
      const totalFrames = results.total_frames;
      const duration = videoRef.current.duration;
      const seekTime = (frameIndex / totalFrames) * duration;
      videoRef.current.currentTime = seekTime;
    }
  };

  return (
    <div className="app-container">
      <header className="navbar forensic-panel">
        <div className="logo">
          <Shield className="logo-icon" size={28} />
          <h1>TRINETRA<span>AI</span></h1>
        </div>
        <div className="nav-right">
          <div className={`status-badge ${systemOnline ? 'online' : 'offline'}`}>
            <div className="status-dot"></div>
            {systemOnline ? 'Neural Core Online' : 'Core Offline'}
          </div>
          <Github className="nav-icon" size={20} />
        </div>
      </header>

      <main className="content">
        {!results && !isLoading ? (
          <div className="hero-section animate-fade-in">
            <h1 className="hero-title">Deepfake <span>Forensics</span></h1>
            <p className="hero-subtitle">
              Advanced neural network analysis for media authentication.
              Upload a video or image to detect AI-generated manipulation.
            </p>
            <UploadZone onFileSelect={handleFileSelect} isLoading={isLoading} />
          </div>
        ) : (
          <div className="analysis-grid animate-fade-in">
            <aside className="sidebar">
              <div className="sticky-sidebar">
                <button className="reset-btn forensic-panel" onClick={handleReset}>
                  <RefreshCw size={18} /> New Analysis
                </button>

                {results && (
                  <VerdictCard
                    verdict={results.verdict || results.classification}
                    confidence={results.confidence}
                    averageProb={results.average_fake_probability}
                  />
                )}

                <div className="info-card forensic-panel">
                  <div className="info-header">
                    <Info size={16} />
                    <span>Analysis Info</span>
                  </div>
                  <div className="info-body">
                    <p>Filename: <span className="mono-data">{file?.name}</span></p>
                    <p>Format: <span className="mono-data">{fileType?.toUpperCase()}</span></p>
                    <p>Model: <span className="mono-data">EfficientNet-B0</span></p>
                  </div>
                </div>
              </div>
            </aside>

            <section className="main-analysis">
              <div className="media-container forensic-panel">
                {fileType === 'video' ? (
                  <video
                    ref={videoRef}
                    src={fileUrl}
                    controls
                    className="analysis-video"
                    onTimeUpdate={handleTimeUpdate}
                  />
                ) : (
                  <div className="image-preview-container">
                    <img src={fileUrl} alt="Analyzed media" className="analysis-image" />
                    {/* Add overlays here if needed */}
                  </div>
                )}
              </div>

              {fileType === 'video' && results?.frame_predictions && (
                <>
                  <ProbabilityChart
                    data={results.frame_predictions}
                    currentTime={currentTime}
                    totalFrames={results.total_frames}
                    onSeek={handleChartSeek}
                  />
                  <SuspiciousGallery
                    framePredictions={results.frame_predictions}
                    videoUrl={fileUrl}
                  />
                </>
              )}
            </section>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2026 TRINETRA Deepfake Detection System. Built for high-stakes verification.</p>
        <div className="footer-links">
          <span>Documentation <ExternalLink size={14} /></span>
          <span>Privacy Policy</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
