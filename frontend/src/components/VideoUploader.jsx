// src/components/VideoUploader.js

import React, { useRef } from 'react';

// The component now receives an `onFileSelect` callback.
function VideoUploader({ onFileSelect }) {
  const fileInputRef = useRef(null);

  // This handler now simply calls the callback with the selected file.
  const handleFile = (file) => {
    if (!file || !file.type.startsWith('video/')) {
      alert('Please provide a valid video file.');
      return;
    }
    onFileSelect(file);
  };

  const handleFileChange = (e) => handleFile(e.target.files[0]);
  const handleDrop = (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove('border-cyan-400');
    handleFile(e.dataTransfer.files[0]);
  };
  const handleDragOver = (e) => e.preventDefault();
  const handleDragEnter = (e) => e.currentTarget.classList.add('border-cyan-400');
  const handleDragLeave = (e) => e.currentTarget.classList.remove('border-cyan-400');

  return (
    <div
      className="w-full h-full flex flex-col items-center justify-center bg-gray-800/50 rounded-lg border-2 border-dashed border-gray-600 transition-colors duration-200 p-6 hover:border-cyan-500 cursor-pointer"
      onClick={() => fileInputRef.current?.click()}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
    >
      <input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
      <svg className="w-16 h-16 text-gray-500 mb-4" fill="currentColor" viewBox="0 0 16 16">
        <path d="M10.5 8.5a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0"/>
        <path d="M2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4zm.5 2a.5.5 0 1 1 0-1 .5.5 0 0 1 0 1m9 2.5a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0"/>
      </svg>
      <h3 className="text-lg sm:text-xl font-semibold text-gray-300 mb-2">Upload Video</h3>
      <p className="text-sm sm:text-base text-gray-500 text-center">Drag & drop a video, or click to browse</p>
    </div>
  );
}

export default VideoUploader;