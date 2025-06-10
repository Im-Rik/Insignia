// src/components/DeveloperAnalytics.jsx

import React from 'react';

// A reusable, styled card component for different sections
const StatCard = ({ children, className = '' }) => (
  <div className={`bg-gray-800/50 border border-gray-700/60 rounded-lg p-3 ${className}`}>
    {children}
  </div>
);

// A reusable component for the main performance metrics (FPS, PPS, RTT)
const PerfMetric = ({ icon, label, value, unit, valueClassName = 'text-gray-50' }) => (
  <div className="flex-1 flex flex-col items-center justify-center text-center bg-gray-900/50 p-2 rounded-md">
    <div className="flex items-center gap-x-1.5 text-xs text-gray-400">
      {icon}
      <span>{label}</span>
    </div>
    <div className="mt-1">
      <span className={`text-2xl font-bold font-mono ${valueClassName}`}>{value}</span>
      <span className="ml-1 text-sm text-gray-400 font-mono">{unit}</span>
    </div>
  </div>
);


function DeveloperAnalytics({
  isConnected,
  isRecording,
  accuracy,
  bufferSize,
  latency,
  pps,
  videoResolution,
  fps,
}) {
  // --- UI Logic & Color Calculation ---
  const latencyDisplay = latency > 0 ? latency.toFixed(0) : 'N/A';
  const latencyColorClass = latency > 250 ? 'text-yellow-400' : (latency > 0 ? 'text-green-400' : 'text-gray-50');

  const getConfidenceColorClass = (conf) => {
    if (conf >= 85) return 'from-teal-500 to-cyan-500';
    if (conf >= 60) return 'from-yellow-500 to-amber-500';
    return 'from-red-500 to-rose-500';
  };
  
  const getBufferColorClass = (size) => {
    if (size >= 55) return 'from-red-500 to-rose-500';
    if (size >= 30) return 'from-yellow-500 to-amber-500';
    return 'from-teal-500 to-cyan-500';
  };
  const bufferPercentage = (bufferSize / 60) * 100;

  return (
    <div className="p-3 sm:p-4 space-y-4">
      {/* FIX: Title is now left-aligned with a bottom border for consistency */}
      <p className="text-sm sm:text-base font-semibold text-cyan-400 border-b border-gray-700 pb-2">
        Developer Analytics
      </p>

      {/* --- Box 1: Live Status --- */}
      <StatCard className="flex items-center justify-around">
          <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-green-400' : 'bg-yellow-400'}`}></span>
              <span className="text-sm font-medium text-gray-300">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${isRecording ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`}></span>
              <span className="text-sm font-medium text-gray-300">{isRecording ? 'Recording' : 'Stopped'}</span>
          </div>
      </StatCard>
      
      {/* --- Box 2: Performance Metrics (3 smaller boxes in a line) --- */}
      <div className="flex items-center gap-2">
        <PerfMetric icon={<IconFps />} label="Frontend" value={fps} unit="FPS" />
        <PerfMetric icon={<IconPps />} label="Backend" value={pps.toFixed(1)} unit="PPS" />
        <PerfMetric icon={<IconLatency />} label="Latency" value={latencyDisplay} unit={latency > 0 ? 'ms' : ''} valueClassName={latencyColorClass} />
      </div>

      {/* --- Box 3: Data & Model Pipeline --- */}
      <StatCard className="space-y-4">
        <div>
          <div className="flex justify-between items-baseline mb-1 text-xs">
            <span className="text-gray-400">Model Confidence</span>
            <span className="font-mono text-gray-200">{accuracy.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-900/50 rounded-full h-2">
            <div className={`h-2 rounded-full bg-gradient-to-r ${getConfidenceColorClass(accuracy)}`} style={{ width: `${accuracy}%` }}></div>
          </div>
        </div>
        
        <div>
          <div className="flex justify-between items-baseline mb-1 text-xs">
            <span className="text-gray-400">Data Buffer</span>
            <span className="font-mono text-gray-200">{bufferSize}/60</span>
          </div>
          <div className="w-full bg-gray-900/50 rounded-full h-2">
            <div className={`h-2 rounded-full bg-gradient-to-r ${getBufferColorClass(bufferSize)}`} style={{ width: `${bufferPercentage}%` }}></div>
          </div>
        </div>

        <div className="flex items-center justify-center gap-2 text-sm text-gray-500 pt-1">
          <IconResolution />
          <span>{videoResolution || 'N/A'}</span>
        </div>
      </StatCard>
    </div>
  );
}

// --- Icons ---
const IconFps = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" /></svg>;
const IconPps = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" /></svg>;
const IconLatency = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
const IconResolution = () => <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>;

DeveloperAnalytics.defaultProps = { latency: 0, pps: 0, fps: 0, videoResolution: 'N/A' };

export default DeveloperAnalytics;