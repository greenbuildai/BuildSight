import { useEffect, useState, useRef } from 'react';
import { AlertCircle, AlertTriangle, Info, Bell, Trash2 } from 'lucide-react';
import { useSettings } from '../SettingsContext';
import { apiBase } from '../lib/api';

export default function AlertsFeed({ isVideoLoaded }) {
    const { settings } = useSettings();
    const [alerts, setAlerts] = useState([]);
    const ws = useRef(null);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (!isVideoLoaded) {
            setAlerts([]);
            return undefined;
        }

        ws.current = new WebSocket(`${apiBase.replace('http', 'ws')}/ws/alerts`);

        ws.current.onmessage = (event) => {
            try {
                const alert = JSON.parse(event.data);
                // Add ID if not present for keying
                const timestamp = alert.timestamp ? new Date(alert.timestamp) : new Date();
                const alertWithId = { ...alert, id: Date.now() + Math.random(), timestamp };
                setAlerts(prev => [alertWithId, ...prev].slice(0, 50));

                // Audio Alert
                if (settings.audibleAlerts && alert.severity === 'high') {
                    // Simple beep using Web Audio API to avoid external file
                    playBeep();
                }

                // Desktop Notification
                if (settings.desktopNotifications && alert.severity === 'high' && Notification.permission === "granted") {
                    new Notification("Safety Alert", { body: alert.message });
                }
            } catch (e) {
                console.error("Failed to parse alert", e);
            }
        };

        return () => ws.current?.close();
    }, [isVideoLoaded, settings, apiBase]);

    useEffect(() => {
        if (settings.desktopNotifications && Notification.permission === "default") {
            Notification.requestPermission();
        }
    }, [settings.desktopNotifications]);

    // Simple beep generator
    const playBeep = () => {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        const ctx = new AudioContext();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880; // A5
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        osc.start();
        osc.stop(ctx.currentTime + 0.1);
    };

    const clearAlerts = () => setAlerts([]);

    const getAlertStyle = (severity) => {
        switch (severity) {
            case 'high': return {
                bg: 'bg-red-50',
                border: 'border-red-200',
                icon: <AlertCircle className="text-red-600 shrink-0" size={20} />,
                text: 'text-red-700'
            };
            case 'medium': return {
                bg: 'bg-yellow-50',
                border: 'border-yellow-200',
                icon: <AlertTriangle className="text-yellow-600 shrink-0" size={20} />,
                text: 'text-yellow-700'
            };
            default: return {
                bg: 'bg-blue-50',
                border: 'border-blue-200',
                icon: <Info className="text-blue-600 shrink-0" size={20} />,
                text: 'text-blue-700'
            };
        }
    };

    return (
        <div className="bg-white/50 backdrop-blur rounded-xl border border-gray-200 h-[600px] flex flex-col shadow-xl overflow-hidden transition-colors duration-300">
            <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                <div className="flex items-center gap-2">
                    <Bell className="text-blue-600" size={20} />
                    <h2 className="text-lg font-bold text-gray-900">Live Alerts</h2>
                    <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-0.5 rounded-full">
                        {alerts.length}
                    </span>
                </div>
                <button
                    onClick={clearAlerts}
                    className="text-xs text-gray-500 hover:text-red-500 flex items-center gap-1 transition-colors"
                >
                    <Trash2 size={14} /> Clear
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin scrollbar-thumb-gray-200" ref={scrollRef}>
                {alerts.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-gray-400 space-y-4 opacity-70">
                        <ShieldIcon size={48} />
                        <p>{isVideoLoaded ? 'System Secure. No active violations.' : 'Upload a video to begin detection.'}</p>
                    </div>
                )}

                {alerts.map((alert) => {
                    const style = getAlertStyle(alert.severity);
                    return (
                        <div
                            key={alert.id}
                            className={`p-4 rounded-lg border-l-4 ${style.bg} ${style.border} flex gap-3 animate-in fade-in slide-in-from-right-4 duration-300`}
                        >
                            {style.icon}
                            <div className="flex-1 min-w-0">
                                <div className="flex justify-between items-start mb-1">
                                    <span className={`text-xs font-bold uppercase tracking-wider ${style.text}`}>
                                        {alert.severity} Violation
                                    </span>
                                    <span className="text-xs text-gray-500 whitespace-nowrap ml-2">
                                        {new Date(alert.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>
                                <p className="text-gray-800 text-sm font-medium leading-relaxed break-words">
                                    {alert.message}
                                </p>
                                <div className="mt-2 flex items-center gap-2 flex-wrap">
                                    <span className="text-xs text-gray-600 bg-white/50 px-2 py-1 rounded">
                                        Zone: {alert.zone_id || 'Unknown'}
                                    </span>
                                    <span className="text-xs text-gray-600 bg-white/50 px-2 py-1 rounded">
                                        Worker: {alert.worker_id ?? 'Unassigned'}
                                    </span>
                                    <span className="text-xs text-gray-600 bg-white/50 px-2 py-1 rounded">
                                        Violation: {alert.violation || 'PPE'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

const ShieldIcon = ({ size }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
    </svg>
);
