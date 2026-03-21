import React from 'react';
import { X, Bell, Shield, Volume2, Mail, Sliders, Calendar, Save, Camera } from 'lucide-react';
import { useSettings } from '../SettingsContext'; // Import context
import { apiBase } from '../lib/api';

export default function SettingsModal({ isOpen, onClose }) {
    const { settings, updateSetting } = useSettings(); // Use context

    const handleLiveToggle = async (enabled) => {
        if (enabled) {
            return;
        }
        updateSetting('liveDetection', false);
        try {
            const mode = 'video';
            await fetch(`${apiBase}/set_input_mode?mode=${mode}`, { method: 'POST' });
        } catch (error) {
            console.error("Failed to sync mode with backend", error);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <div className="bg-white w-full max-w-2xl rounded-2xl shadow-2xl border border-gray-200 overflow-hidden animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[90vh]">

                {/* Header */}
                <div className="flex items-center justify-between p-5 border-b border-gray-100 shrink-0">
                    <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                        <SettingsIcon /> Settings & Configuration
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors"
                    >
                        <X size={22} />
                    </button>
                </div>

                {/* Content - Scrollable */}
                <div className="p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-200">
                    <div className="space-y-8">

                        {/* Detection & AI Section */}
                        <section>
                            <h3 className="text-sm font-bold text-blue-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <Sliders size={16} /> AI & Detection Calibration
                            </h3>
                            <div className="bg-gray-50 rounded-xl p-5 space-y-6 border border-gray-100">
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <label className="text-sm font-semibold text-gray-700">Motion Sensitivity</label>
                                        <span className="text-xs font-mono font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">High ({settings.motionSensitivity}%)</span>
                                    </div>
                                    <input
                                        type="range"
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                        min="0"
                                        max="100"
                                        value={settings.motionSensitivity}
                                        onChange={(e) => updateSetting('motionSensitivity', parseInt(e.target.value))}
                                    />
                                    <p className="text-xs text-gray-500">Adjust how sensitive the computer vision model is to movement variations.</p>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <label className="text-sm font-semibold text-gray-700">Confidence Threshold</label>
                                        <span className="text-xs font-mono font-medium text-purple-600 bg-purple-50 px-2 py-1 rounded">{settings.confidenceThreshold}%</span>
                                    </div>
                                    <input
                                        type="range"
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                                        min="0"
                                        max="100"
                                        value={settings.confidenceThreshold}
                                        onChange={(e) => updateSetting('confidenceThreshold', parseInt(e.target.value))}
                                    />
                                    <p className="text-xs text-gray-500">Only alert when the AI is at least this confident in its detection.</p>
                                </div>

                                <div className="pt-2 border-t border-gray-200/50">
                                    <ToggleOption
                                        label="Show Safety Zones Overlay"
                                        icon={Shield}
                                        checked={settings.showSafetyZones}
                                        onChange={(val) => updateSetting('showSafetyZones', val)}
                                        description="Visualizes the designated safety and danger zones on the video feed."
                                    />
                                </div>

                                <div className="pt-2 border-t border-gray-200/50">
                                    <ToggleOption
                                        label="Live CCTV Detection"
                                        icon={Camera}
                                        checked={false}
                                        disabled
                                        onChange={handleLiveToggle}
                                        description="Live CCTV input is disabled until rollout. Video upload is the default."
                                    />
                                </div>
                            </div>
                        </section>

                        {/* Notifications Section */}
                        <section>
                            <h3 className="text-sm font-bold text-orange-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <Bell size={16} /> Notifications & Alerts
                            </h3>
                            <div className="bg-gray-50 rounded-xl p-5 space-y-4 border border-gray-100">
                                <ToggleOption
                                    label="Desktop Push Notifications"
                                    icon={Bell}
                                    checked={settings.desktopNotifications}
                                    onChange={(val) => updateSetting('desktopNotifications', val)}
                                    description="Receive instant browser notifications for high-priority safety violations."
                                />
                                <ToggleOption
                                    label="Audible Alert Chime"
                                    icon={Volume2}
                                    checked={settings.audibleAlerts}
                                    onChange={(val) => updateSetting('audibleAlerts', val)}
                                    description="Play a sound when a severe safety incident is detected."
                                />
                                <ToggleOption
                                    label="Daily Email Digest"
                                    icon={Mail}
                                    checked={settings.emailDigest}
                                    onChange={(val) => updateSetting('emailDigest', val)}
                                    description="Send a summary of all site activity and safety scores at 6:00 PM."
                                />
                            </div>
                        </section>

                        {/* System & Storage Section */}
                        <section>
                            <h3 className="text-sm font-bold text-green-600 uppercase tracking-widest mb-4 flex items-center gap-2">
                                <Save size={16} /> System & Data
                            </h3>
                            <div className="bg-gray-50 rounded-xl p-5 space-y-4 border border-gray-100">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Calendar size={18} className="text-gray-500" />
                                        <div>
                                            <p className="text-sm font-semibold text-gray-700">Video Retention Period</p>
                                            <p className="text-xs text-gray-500">How long to keep footage</p>
                                        </div>
                                    </div>
                                    <select
                                        className="bg-white border border-gray-200 text-gray-700 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
                                        value={settings.retentionPeriod}
                                        onChange={(e) => updateSetting('retentionPeriod', e.target.value)}
                                    >
                                        <option>24 Hours</option>
                                        <option>3 Days</option>
                                        <option>7 Days</option>
                                        <option>30 Days</option>
                                    </select>
                                </div>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Save size={18} className="text-gray-500" />
                                        <div>
                                            <p className="text-sm font-semibold text-gray-700">Auto-Backup Logs</p>
                                            <p className="text-xs text-gray-500">Sync incident logs to cloud</p>
                                        </div>
                                    </div>
                                    <select
                                        className="bg-white border border-gray-200 text-gray-700 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
                                        value={settings.autoBackup}
                                        onChange={(e) => updateSetting('autoBackup', e.target.value)}
                                    >
                                        <option>Disabled</option>
                                        <option>Real-time</option>
                                        <option>Hourly</option>
                                        <option>Daily</option>
                                    </select>
                                </div>
                            </div>
                        </section>

                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 bg-gray-50 border-t border-gray-100 flex justify-end gap-3 shrink-0">
                    <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-200 rounded-lg transition-colors">
                        Close
                    </button>
                    <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors flex items-center gap-2">
                        <Save size={16} /> Save Changes
                    </button>
                </div>

            </div>
        </div>
    );
}

function ToggleOption({ label, icon: Icon, checked, onChange, description, disabled }) {
    return (
        <div className="flex items-start justify-between py-1">
            <div className="flex gap-3">
                <div className="mt-0.5 text-gray-500">
                    <Icon size={18} />
                </div>
                <div>
                    <span className="text-sm font-medium text-gray-700 block">{label}</span>
                    {description && <p className="text-xs text-gray-500 mt-0.5 leading-relaxed max-w-[300px]">{description}</p>}
                </div>
            </div>
            <button
                onClick={() => !disabled && onChange(!checked)}
                className={`flex-shrink-0 w-11 h-6 rounded-full p-1 transition-colors duration-200 ease-in-out ${disabled ? 'cursor-not-allowed bg-gray-100' : 'cursor-pointer'} ${checked ? 'bg-blue-600' : 'bg-gray-200'}`}
            >
                <div className={`w-4 h-4 rounded-full bg-white shadow-sm transform transition-transform duration-200 ${checked ? 'translate-x-5' : 'translate-x-0'}`} />
            </button>
        </div>
    )
}

const SettingsIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path><circle cx="12" cy="12" r="3"></circle></svg>
)
