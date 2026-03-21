import { useEffect, useRef, useState } from 'react';
import { Maximize2, Settings, Camera, FileVideo, Plus, Upload } from 'lucide-react';
import { useSettings } from '../SettingsContext';
import { apiBase } from '../lib/api';
import GpuVideoFeed from './GpuVideoFeed';
import { checkBackendHealth } from '../lib/backendHealth';

export default function VideoPlayer({ isVideoLoaded, videoName, videoUrl, onVideoLoaded }) {
    const { settings } = useSettings();
    const fileInputRef = useRef(null);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(null);
    const [error, setError] = useState('');
    const [streamKey, setStreamKey] = useState(0);

    useEffect(() => {
        if (isVideoLoaded) {
            setStreamKey(Date.now());
        }
    }, [isVideoLoaded, videoName, videoUrl]);

    const handleUploadClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleFileChange = async (event) => {
        const file = event.target.files?.[0];
        if (!file) {
            console.warn('[VideoPlayer] No file selected');
            return;
        }

        console.log('[VideoPlayer] Upload initiated', {
            fileName: file.name,
            fileSize: `${(file.size / 1024 / 1024).toFixed(2)}MB`,
            fileType: file.type,
            apiBase: apiBase
        });

        // Clear previous errors
        setError('');

        // Validate apiBase
        if (!apiBase) {
            const msg = 'API endpoint not configured. Please check your settings.';
            console.error('[VideoPlayer]', msg);
            setError(msg);
            return;
        }

        // Validate file type
        if (file.type && !file.type.startsWith('video/')) {
            const msg = `Invalid file type: ${file.type}. Please upload a video file.`;
            console.error('[VideoPlayer]', msg);
            setError(msg);
            event.target.value = '';
            return;
        }

        // Validate file size
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (file.size > maxSize) {
            const msg = `File too large: ${(file.size / 1024 / 1024 / 1024).toFixed(2)}GB. Maximum size is 2GB.`;
            console.error('[VideoPlayer]', msg);
            setError(msg);
            event.target.value = '';
            return;
        }

        if (file.size === 0) {
            const msg = 'File is empty. Please choose a valid video file.';
            console.error('[VideoPlayer]', msg);
            setError(msg);
            event.target.value = '';
            return;
        }

        // Check backend health before upload
        console.log('[VideoPlayer] Checking backend health...');
        const health = await checkBackendHealth(apiBase);
        if (!health.healthy) {
            const msg = `Cannot connect to backend: ${health.error}. Please ensure the backend server is running on port 8000.`;
            console.error('[VideoPlayer]', msg, health);
            setError(msg);
            event.target.value = '';
            return;
        }
        console.log('[VideoPlayer] Backend health check passed', health);

        setUploading(true);
        setUploadProgress(0);

        try {
            // Helper function for fetch with timeout
            const fetchWithTimeout = async (url, options, timeout = 30000) => {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), timeout);

                try {
                    const response = await fetch(url, {
                        ...options,
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);
                    return response;
                } catch (err) {
                    clearTimeout(timeoutId);
                    throw err;
                }
            };

            const initResponse = await fetchWithTimeout(`${apiBase}/upload_init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: file.name,
                    content_type: file.type,
                    size: file.size
                })
            }, 30000);

            if (!initResponse.ok) {
                throw new Error('Upload init failed');
            }

            const initData = await initResponse.json();
            const uploadId = initData.upload_id;
            if (!uploadId) {
                throw new Error('Upload init failed');
            }

            console.log('[VideoPlayer] Upload session initialized', {
                uploadId: uploadId,
                chunkSize: '5MB',
                totalChunks: Math.ceil(file.size / (5 * 1024 * 1024))
            });

            const chunkSize = 5 * 1024 * 1024;
            let offset = 0;

            const uploadChunk = (chunk, chunkOffset) => new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('PUT', `${apiBase}/upload_chunk/${uploadId}?offset=${chunkOffset}`, true);
                xhr.setRequestHeader('Content-Type', file.type || 'application/octet-stream');
                xhr.upload.onprogress = (event) => {
                    const total = file.size || 0;
                    if (!total) {
                        return;
                    }
                    const loaded = Math.min(chunk.size, event.loaded);
                    const percent = Math.min(100, Math.round(((chunkOffset + loaded) / total) * 100));
                    setUploadProgress(percent);
                };
                xhr.onload = () => {
                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error('Upload failed'));
                        return;
                    }
                    console.log('[VideoPlayer] Chunk uploaded', {
                        offset: chunkOffset,
                        bytesWritten: chunk.size,
                        progress: Math.min(100, Math.round(((chunkOffset + chunk.size) / file.size) * 100)) + '%'
                    });
                    resolve();
                };
                xhr.onerror = () => reject(new Error('Upload failed'));
                xhr.send(chunk);
            });

            while (offset < file.size) {
                const chunk = file.slice(offset, offset + chunkSize);
                await uploadChunk(chunk, offset);
                offset += chunk.size;
            }

            const completeResponse = await fetchWithTimeout(
                `${apiBase}/upload_complete/${uploadId}`,
                { method: 'POST' },
                30000
            );
            if (!completeResponse.ok) {
                let detail = 'Upload failed';
                try {
                    const payload = await completeResponse.json();
                    detail = payload.detail || detail;
                } catch {
                    // Ignore JSON parse errors
                }
                throw new Error(detail);
            }

            const data = await completeResponse.json();
            console.log('[VideoPlayer] Upload completed successfully', {
                filename: data.filename,
                videoUrl: data.video_url,
                totalSize: `${(file.size / 1024 / 1024).toFixed(2)}MB`
            });

            setStreamKey(Date.now());
            if (onVideoLoaded) {
                const nextUrl = data.video_url ? `${apiBase}${data.video_url}` : '';
                onVideoLoaded(data.filename || file.name, nextUrl);
            }
        } catch (err) {
            console.error('[VideoPlayer] Upload failed', {
                error: err.message,
                stack: err.stack,
                fileName: file?.name
            });

            let userMessage = 'Upload failed';

            // Categorize errors for user-friendly messages
            if (err.name === 'TypeError' && (err.message.includes('fetch') || err.message.includes('Failed to fetch'))) {
                userMessage = 'Cannot connect to backend. Please ensure the backend server is running on port 8000.';
            } else if (err.name === 'AbortError') {
                userMessage = 'Upload timeout. The backend might be overloaded or the file is too large.';
            } else if (err.message.includes('Upload init failed')) {
                userMessage = 'Failed to initialize upload. Check backend logs for details.';
            } else if (err.message.includes('Upload incomplete')) {
                userMessage = err.message; // Use backend's detailed message
            } else if (err.message) {
                userMessage = err.message;
            }

            setError(userMessage);
        } finally {
            setUploading(false);
            setUploadProgress(null);
            if (event?.target) {
                event.target.value = '';
            }
        }
    };

    return (
        <div className="bg-white/50 backdrop-blur rounded-xl overflow-hidden shadow-2xl border border-gray-200 relative group h-full flex flex-col transition-colors duration-300">
            {/* Header Overlay */}
            <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/80 to-transparent z-10 flex justify-between items-start opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div>
                    <h2 className="text-lg font-bold text-white flex items-center gap-2">
                        <Camera size={18} className="text-gray-400" />
                        Video PPE Analysis
                    </h2>
                    <p className="text-xs text-blue-200 flex items-center gap-1 mt-1">
                        <FileVideo size={12} />
                        {isVideoLoaded ? (videoName || 'Uploaded video loaded') : 'Awaiting upload'}
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={handleUploadClick}
                        disabled={uploading}
                        className={`p-2 rounded-lg text-white transition-colors ${uploading ? 'opacity-60 cursor-not-allowed' : 'hover:bg-white/10'}`}
                        aria-label="Upload new video"
                    >
                        <Upload size={18} />
                    </button>
                    <button className="p-2 hover:bg-white/10 rounded-lg text-white transition-colors">
                        <Settings size={18} />
                    </button>
                    <button className="p-2 hover:bg-white/10 rounded-lg text-white transition-colors">
                        <Maximize2 size={18} />
                    </button>
                </div>
            </div>

            {/* Live Indicator (Always Visible) */}
            <div className="absolute top-4 right-4 z-20">
                <span className={`flex items-center gap-2 px-3 py-1 backdrop-blur rounded text-white text-xs font-bold tracking-wider shadow-lg ${isVideoLoaded ? 'bg-emerald-500/90 animate-pulse' : 'bg-slate-500/80'}`}>
                    {isVideoLoaded ? 'ANALYZING' : 'NO VIDEO'}
                </span>
            </div>

            {/* Main Video Area */}
            <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden">
                {/* Grid Overlay for technical feel */}
                <div className="absolute inset-0 pointer-events-none opacity-10"
                    style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '100px 100px' }}>
                </div>

                {!isVideoLoaded && (
                    <div className="relative z-10 flex flex-col items-center gap-4 text-gray-200">
                        <button
                            onClick={handleUploadClick}
                            disabled={uploading}
                            className={`flex items-center justify-center w-20 h-20 rounded-full border-2 border-dashed border-gray-400/60 text-gray-200 transition-colors ${uploading ? 'cursor-not-allowed opacity-60' : 'hover:text-white hover:border-white'}`}
                            aria-label="Add video"
                        >
                            <Plus size={32} />
                        </button>
                        <div className="text-center space-y-1">
                            <p className="text-sm font-semibold">Upload a video to start PPE analysis</p>
                            <p className="text-xs text-gray-400">Detection stays idle until footage is loaded.</p>
                            {uploading && (
                                <p className="text-xs text-blue-300">
                                    Uploading {uploadProgress ?? 0}%...
                                </p>
                            )}
                            {error && <p className="text-xs text-red-300">{error}</p>}
                        </div>
                    </div>
                )}

                {isVideoLoaded && (
                    <GpuVideoFeed
                        key={streamKey}
                        src={`${apiBase}/video_feed?ts=${streamKey}`}
                        alt="Processed video stream with PPE detection"
                        className="w-full h-full object-contain"
                    />
                )}

                {/* Safety Zones Overlay */}
                {settings.showSafetyZones && (
                    <div className="absolute inset-0 pointer-events-none">
                        <svg className="w-full h-full opacity-30">
                            {/* Danger Zone Mockup */}
                            <path d="M 100 100 L 300 100 L 300 400 L 100 400 Z" fill="red" stroke="red" strokeWidth="2" />
                            <text x="110" y="130" fill="white" className="text-xs font-bold">DANGER ZONE</text>

                            {/* Safe Zone Mockup */}
                            <path d="M 400 200 L 600 200 L 600 500 L 400 500 Z" fill="green" stroke="green" strokeWidth="2" />
                            <text x="410" y="230" fill="white" className="text-xs font-bold">SAFE WALKWAY</text>
                        </svg>
                    </div>
                )}
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileChange}
            />

            {/* Footer Overlay */}
            <div className="flex justify-between items-center bg-gray-50 px-4 py-2 border-t border-gray-200 text-xs text-gray-500 shrink-0">
                <div className="flex gap-4">
                    <span>Res: 1080p</span>
                    <span>FPS: 30</span>
                    <span>HSN: 85258020</span>
                </div>
                <div className="font-mono text-[10px] opacity-80">
                    Model: TV-ULT-IP30PO0400N-4G-SD-ATC
                </div>
            </div>
        </div>
    );
}
