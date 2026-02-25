import React, { useCallback, useState, useRef } from 'react';
import { Upload, FileVideo, FileImage, ShieldAlert } from 'lucide-react';
import './UploadZone.css';

const UploadZone = ({ onFileSelect, isLoading }) => {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) onFileSelect(file);
    }, [onFileSelect]);

    const handleChange = (e) => {
        const file = e.target.files[0];
        if (file) onFileSelect(file);
    };

    const triggerFileInput = () => {
        if (!isLoading && fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    return (
        <div
            className={`upload-zone forensic-panel ${isDragging ? 'dragging' : ''} ${isLoading ? 'loading' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={triggerFileInput}
            style={{ cursor: isLoading ? 'default' : 'pointer' }}
        >
            <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleChange}
                accept="video/*,image/*"
            />
            <div className="upload-content">
                <div className="icon-stack">
                    <div className="icon-bg main"><Upload size={48} /></div>
                    <div className="icon-bg sub left"><FileVideo size={24} /></div>
                    <div className="icon-bg sub right"><FileImage size={24} /></div>
                </div>
                <h2>Drop your media here</h2>
                <p>Supports MP4, AVI, MOV, JPG, PNG, WebP</p>
                <button className="select-btn" type="button" onClick={(e) => { e.stopPropagation(); triggerFileInput(); }}>Select File</button>
            </div>

            {isLoading && (
                <div className="loading-overlay">
                    <div className="scanner"></div>
                    <ShieldAlert className="animate-pulse-subtle" size={40} color="var(--accent-blue)" />
                    <p>Analyzing Neural Patterns...</p>
                </div>
            )}
        </div>
    );
};

export default UploadZone;
