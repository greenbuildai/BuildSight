import React, { useEffect, useState } from 'react';
import VideoPlayer from './VideoPlayer';
import AlertsFeed from './AlertsFeed';
import StatsPanel from './StatsPanel';
import SettingsModal from './SettingsModal';
import { Settings } from 'lucide-react';
import { apiBase } from '../lib/api';
import { checkBackendHealth } from '../lib/backendHealth';

export default function Dashboard() {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [videoLoaded, setVideoLoaded] = useState(false);
    const [videoName, setVideoName] = useState('');
    const [videoUrl, setVideoUrl] = useState('');
    const [backendStatus, setBackendStatus] = useState('checking');
    const [backendError, setBackendError] = useState('');

    useEffect(() => {
        let active = true;
        let pollInterval = null;

        const performHealthCheck = async () => {
            console.log('[Dashboard] Checking backend health...');
            const health = await checkBackendHealth(apiBase);

            if (!active) return;

            console.log('[Dashboard] Health check result:', health);

            if (health.healthy) {
                setBackendStatus('healthy');

                // Stop polling when healthy
                if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    console.log('[Dashboard] Backend healthy, stopped polling');
                }

                // Fetch config
                try {
                    const res = await fetch(`${apiBase}/config`);
                    const data = await res.json();

                    if (!active) return;

                    const loaded = Boolean(data.video_loaded);
                    setVideoLoaded(loaded);
                    if (loaded && data.video_source) {
                        const parts = data.video_source.split(/[/\\\\]/);
                        setVideoName(parts[parts.length - 1]);
                    }
                    if (data.video_url) {
                        setVideoUrl(`${apiBase}${data.video_url}`);
                    } else {
                        setVideoUrl('');
                    }
                } catch (err) {
                    console.error('[Dashboard] Config fetch failed:', err);
                }
            } else {
                setBackendStatus('unhealthy');
                setBackendError(health.error);
                setVideoLoaded(false);
                setVideoUrl('');

                // Start polling if not already polling
                if (!pollInterval) {
                    console.log('[Dashboard] Backend unhealthy, starting poll every 8 seconds');
                    pollInterval = setInterval(() => {
                        if (active) {
                            performHealthCheck();
                        }
                    }, 8000); // Poll every 8 seconds
                }
            }
        };

        // Initial health check
        performHealthCheck();

        return () => {
            active = false;
            if (pollInterval) {
                clearInterval(pollInterval);
                console.log('[Dashboard] Cleanup: stopped polling');
            }
        };
    }, []);

    const handleRetryConnection = async () => {
        console.log('[Dashboard] Manual retry triggered');
        setBackendStatus('checking');
        setBackendError('');

        const health = await checkBackendHealth(apiBase);

        if (health.healthy) {
            setBackendStatus('healthy');
            // Fetch config
            try {
                const res = await fetch(`${apiBase}/config`);
                const data = await res.json();
                const loaded = Boolean(data.video_loaded);
                setVideoLoaded(loaded);
                if (loaded && data.video_source) {
                    const parts = data.video_source.split(/[/\\\\]/);
                    setVideoName(parts[parts.length - 1]);
                }
                if (data.video_url) {
                    setVideoUrl(`${apiBase}${data.video_url}`);
                } else {
                    setVideoUrl('');
                }
            } catch (err) {
                console.error('[Dashboard] Config fetch failed:', err);
            }
        } else {
            setBackendStatus('unhealthy');
            setBackendError(health.error);
            setVideoLoaded(false);
            setVideoUrl('');
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 text-gray-900 p-6 font-sans selection:bg-blue-500/30 gpu-accelerated video-surface">



            <div className="relative max-w-8xl mx-auto space-y-6">
                <header className="flex justify-between items-center py-2 h-16">
                    <div className="flex items-center gap-5">
                        <div className="transition-transform hover:scale-105 duration-300">
                            <img src="/logo.png" alt="BuildSight Logo" className="w-16 h-16 object-contain mix-blend-multiply" />
                        </div>
                        <div className="flex flex-col justify-center gap-1.5">
                            <h1 className="text-3xl leading-none font-extrabold tracking-tight text-gray-900">
                                BuildSight <span className="text-blue-600">AI</span>
                            </h1>
                            <p className="text-[11px] leading-tight font-semibold text-gray-500 tracking-wider uppercase">
                                AI-enabled Construction Safety Monitoring System
                            </p>
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        <div className={`bg-white px-4 py-2 rounded-full border flex items-center gap-3 shadow-sm ${
                            backendStatus === 'healthy' ? 'border-gray-200' :
                            backendStatus === 'checking' ? 'border-blue-300 bg-blue-50' :
                            'border-red-300 bg-red-50'
                        }`}>
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] uppercase text-gray-500 font-bold tracking-wider">System Status</span>
                                <span className={`text-xs font-bold ${
                                    backendStatus === 'healthy' ? 'text-green-600' :
                                    backendStatus === 'checking' ? 'text-blue-600' :
                                    'text-red-600'
                                }`}>
                                    {backendStatus === 'healthy' ? 'Operational' :
                                     backendStatus === 'checking' ? 'Checking...' :
                                     'Backend Offline'}
                                </span>
                            </div>
                            <span className="relative flex h-3 w-3">
                                {backendStatus === 'healthy' && (
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                                )}
                                {backendStatus === 'checking' && (
                                    <span className="animate-spin absolute inline-flex h-full w-full rounded-full border-2 border-blue-400 border-t-transparent"></span>
                                )}
                                <span className={`relative inline-flex rounded-full h-3 w-3 ${
                                    backendStatus === 'healthy' ? 'bg-green-500' :
                                    backendStatus === 'checking' ? 'bg-blue-500' :
                                    'bg-red-500'
                                }`}></span>
                            </span>
                        </div>

                        <button
                            onClick={() => setIsSettingsOpen(true)}
                            className="p-2.5 rounded-full bg-white hover:bg-gray-50 border border-gray-200 text-gray-600 shadow-sm transition-colors"
                            aria-label="Settings"
                        >
                            <Settings size={20} />
                        </button>
                    </div>
                </header>

                {backendStatus === 'checking' && (
                    <div className="p-4 bg-blue-50 border border-blue-300 rounded-lg">
                        <p className="text-sm text-blue-800 font-semibold">🔄 Checking backend connection...</p>
                    </div>
                )}

                {backendStatus === 'unhealthy' && (
                    <div className="p-4 bg-red-50 border border-red-300 rounded-lg">
                        <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                                <p className="text-sm text-red-800 font-semibold">⚠️ Backend Server Not Running</p>
                                <p className="text-xs text-red-700 mt-1">{backendError}</p>
                                <p className="text-xs text-red-600 mt-2">
                                    Please start the backend: <code className="bg-red-100 px-2 py-1 rounded font-mono">python -m uvicorn backend.main:app --reload</code>
                                </p>
                                <p className="text-xs text-red-500 mt-2 italic">Automatically checking every 8 seconds...</p>
                            </div>
                            <button
                                onClick={handleRetryConnection}
                                className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded transition-colors whitespace-nowrap"
                            >
                                Retry Connection
                            </button>
                        </div>
                    </div>
                )}

                <StatsPanel />

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-280px)] min-h-[600px]">
                    <div className="lg:col-span-2 h-full">
                        <VideoPlayer
                            isVideoLoaded={videoLoaded}
                            videoName={videoName}
                            videoUrl={videoUrl}
                            onVideoLoaded={(name, url) => {
                                setVideoLoaded(true);
                                if (name) setVideoName(name);
                                if (url) setVideoUrl(url);
                            }}
                        />
                    </div>
                    <div className="lg:col-span-1 h-full">
                        <AlertsFeed isVideoLoaded={videoLoaded} />
                    </div>
                </div>
            </div>

            <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
        </div>
    );
}
