import React from 'react';
import { Target, AlertTriangle } from 'lucide-react';
import './SuspiciousGallery.css';

const SuspiciousGallery = ({ framePredictions, videoUrl }) => {
    // Filter for top suspicious frames
    const suspicious = [...framePredictions]
        .sort((a, b) => b.probabilities.FAKE - a.probabilities.FAKE)
        .slice(0, 4);

    return (
        <div className="gallery-section forensic-panel" style={{ padding: '24px' }}>
            <div className="gallery-header">
                <Target size={18} color="var(--accent-orange)" />
                <div>
                    <h3>High-Risk Artifacts Detected</h3>
                    <p>Frames exhibiting significant neural variance</p>
                </div>
            </div>

            <div className="gallery-grid">
                {suspicious.map((item, index) => {
                    const prob = (item.probabilities.FAKE * 100).toFixed(2);
                    return (
                        <div key={index} className="gallery-item forensic-panel animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                            <div className="frame-preview">
                                <div className="placeholder-frame">
                                    <AlertTriangle size={24} color="var(--text-muted)" />
                                    <span>FRAME EXTRACT</span>
                                </div>
                                <div className="frame-badge">
                                    {Math.floor(item.frame / 30)}s : {item.frame % 30}f
                                </div>
                            </div>

                            <div className="frame-info">
                                <div className="frame-meta">
                                    <span className="detail-label" style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>Deviation</span>
                                    <span className="prob-value mono-data">{prob}%</span>
                                </div>

                                <div className="prob-bar-container">
                                    <div className="prob-bar" style={{ width: `${prob}%` }}></div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default SuspiciousGallery;
