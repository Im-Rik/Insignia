// src/components/HistoryList.jsx

import React from 'react';

const HistoryItem = ({ item }) => {
  const confidence = item.confidence * 100;

  const getConfidenceColor = (conf) => {
    if (conf >= 85) return 'border-green-500/50';
    if (conf >= 60) return 'border-yellow-500/50';
    return 'border-red-500/50';
  };

  return (
    <div className={`
      flex justify-between items-center p-2.5 bg-gray-800/50 rounded-lg 
      border-l-4 ${getConfidenceColor(confidence)} animate-fade-in
    `}>
      <span className="font-mono font-semibold text-gray-200">{item.prediction}</span>
      <div className="text-right">
        <span className="font-mono text-xs text-gray-400 block">{confidence.toFixed(1)}%</span>
        <span className="font-mono text-xs text-gray-500 block">{item.time}</span>
      </div>
    </div>
  );
};

const HistoryList = ({ history }) => {
  return (
    <div className="flex flex-col h-full p-3 sm:p-4">
      <p className="text-sm sm:text-base font-semibold text-cyan-400 border-b border-gray-700 pb-2">
        Prediction History
      </p>
      <div className="flex-grow space-y-2 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-600 hover:scrollbar-thumb-gray-500 scrollbar-track-gray-800/50 scrollbar-thumb-rounded-full">
        {history.length > 0 ? (
          history.map((item, index) => (
            <HistoryItem key={`${item.time}-${index}`} item={item} />
          ))
        ) : (
          <div className="text-center text-gray-500 pt-10">
            <p>History will appear here...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HistoryList;