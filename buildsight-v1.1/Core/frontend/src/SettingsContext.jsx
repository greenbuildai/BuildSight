import React, { createContext, useContext, useState, useEffect } from 'react';

const SettingsContext = createContext();

export const defaultSettings = {
    // AI & Detection
    motionSensitivity: 85,
    confidenceThreshold: 70,
    showSafetyZones: true,
    liveDetection: false,

    // Notifications
    desktopNotifications: true,
    audibleAlerts: false,
    emailDigest: true,

    // System
    retentionPeriod: '7 Days',
    autoBackup: 'Real-time'
};

export function SettingsProvider({ children }) {
    // Load from localStorage if available, else default
    const [settings, setSettings] = useState(() => {
        const saved = localStorage.getItem('buildsight_settings');
        return saved ? JSON.parse(saved) : defaultSettings;
    });

    // Save to localStorage whenever settings change
    useEffect(() => {
        localStorage.setItem('buildsight_settings', JSON.stringify(settings));
    }, [settings]);

    const updateSetting = (key, value) => {
        setSettings(prev => ({
            ...prev,
            [key]: value
        }));
    };

    const resetSettings = () => {
        setSettings(defaultSettings);
    };

    return (
        <SettingsContext.Provider value={{ settings, updateSetting, resetSettings }}>
            {children}
        </SettingsContext.Provider>
    );
}

export function useSettings() {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error('useSettings must be used within a SettingsProvider');
    }
    return context;
}
