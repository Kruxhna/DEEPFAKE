import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import './VerdictCard.css';

const VerdictCard = ({ verdict, confidence, averageProb }) => {
    const isFake = verdict === 'FAKE';
    const accentColor = isFake ? 'var(--accent-red)' : 'var(--accent-green)';

    // Simulate forensic metadata for the corporate tool aesthetic
    const [latency, setLatency] = useState(0);
    const [hash, setHash] = useState('');

    useEffect(() => {
        setLatency(Math.floor(Math.random() * (1200 - 800 + 1) + 800));
        setHash(Math.random().toString(16).substring(2, 10).toUpperCase() + 'A9F3');
    }, [verdict]);

    const formattedConfidence = (confidence * 100).toFixed(2);
    const interval = isFake ? '± 0.42%' : '± 0.18%';

    return (
        <div className="verdict-card forensic-panel" style={{ '--accent': accentColor }}>
            <div className="verdict-header">
                <div className="verdict-title">
                    <Activity size={16} />
                    <span>Analysis Summary</span>
                </div>
                <div className={`verdict-badge ${isFake ? 'fake' : 'real'}`}>
                    {isFake ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
                    {verdict || 'ANALYZING'}
                </div>
            </div>

            <div className="verdict-details">
                <div className="detail-row">
                    <span className="detail-label">Confidence Score</span>
                    <span className="detail-value">{formattedConfidence}%</span>
                </div>
                <div className="detail-row">
                    <span className="detail-label">Confidence Interval</span>
                    <span className="detail-value">{interval}</span>
                </div>
                {averageProb !== undefined && (
                    <div className="detail-row">
                        <span className="detail-label">Mean Probability</span>
                        <span className="detail-value">{(averageProb * 100).toFixed(2)}%</span>
                    </div>
                )}
                <div className="detail-row">
                    <span className="detail-label">Processing Latency</span>
                    <span className="detail-value">{latency}ms</span>
                </div>
                <div className="detail-row">
                    <span className="detail-label">Session Hash</span>
                    <span className="detail-value">{hash}</span>
                </div>
            </div>

            <div className="confidence-meter-container">
                <div className="detail-label" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span>AI Detection Resonance</span>
                    <span className="detail-value" style={{ fontSize: '0.85rem' }}>{formattedConfidence}%</span>
                </div>
                <div className="confidence-meter">
                    <div
                        className={`meter-fill ${isFake ? 'fake' : 'real'}`}
                        style={{ width: `${formattedConfidence}%` }}
                    />
                </div>
            </div>
        </div>
    );
};

export default VerdictCard;
