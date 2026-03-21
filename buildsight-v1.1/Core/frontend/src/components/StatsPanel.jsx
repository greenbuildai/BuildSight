import React from 'react';
import { Users, AlertTriangle, ShieldCheck, Activity } from 'lucide-react';

export default function StatsPanel() {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <StatCard
                title="Active Workers"
                value="12"
                icon={Users}
                trend="+2 from yesterday"
                trendColor="text-green-500"
                color="blue"
            />
            <StatCard
                title="PPE Violations"
                value="3"
                icon={AlertTriangle}
                trend="Needs attention"
                trendColor="text-red-500"
                color="red"
            />
            <StatCard
                title="Safety Score"
                value="94%"
                icon={ShieldCheck}
                trend="Top 10%"
                trendColor="text-green-500"
                color="green"
            />
            <StatCard
                title="Active Zones"
                value="4"
                icon={Activity}
                trend="All Systems Normal"
                trendColor="text-gray-400"
                color="purple"
            />
        </div>
    );
}

function StatCard({ title, value, icon: Icon, trend, trendColor, color }) {
    const colorClasses = {
        blue: "bg-blue-50 text-blue-600 border-blue-200",
        red: "bg-red-50 text-red-600 border-red-200",
        green: "bg-green-50 text-green-600 border-green-200",
        purple: "bg-purple-50 text-purple-600 border-purple-200",
    };

    return (
        <div className="bg-white/50 backdrop-blur border border-gray-200 p-5 rounded-xl flex flex-col justify-between shadow-sm hover:shadow-md transition-shadow duration-200">
            <div className="flex justify-between items-start mb-2">
                <div className={`p-2.5 rounded-lg ${colorClasses[color]}`}>
                    <Icon size={20} strokeWidth={2.5} />
                </div>
                {trend && (
                    <span className={`text-xs font-medium ${trendColor} bg-gray-50 px-2 py-1 rounded-full border border-gray-200`}>
                        {trend}
                    </span>
                )}
            </div>
            <div>
                <p className="text-gray-500 text-sm font-medium mb-1">{title}</p>
                <p className="text-3xl font-bold text-gray-900 tracking-tight">{value}</p>
            </div>
        </div>
    )
}
