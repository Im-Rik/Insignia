import React from 'react';
import PropTypes from 'prop-types';

function Loader({ message, progress }) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-gray-900/50 rounded-xl text-center">
      {/* Spinner Element */}
      <div className="w-12 h-12 border-4 border-dashed rounded-full animate-spin border-cyan-500 mb-4"></div>
      
      {/* Status Message */}
      <p className="text-white font-semibold px-4">{message}</p>
      
      {/* Progress Bar (only shown if the progress prop is a number) */}
      {typeof progress === 'number' && (
        <div className="w-3/4 h-2 bg-gray-600 rounded-full mt-4 overflow-hidden">
          <div 
            className="h-full bg-cyan-500 transition-all duration-300 ease-in-out" 
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      )}
    </div>
  );
}

// Define prop types for better component API and error checking
Loader.propTypes = {
  /** The message to display below the spinner */
  message: PropTypes.string,
  /** The progress percentage (0-100) for the progress bar */
  progress: PropTypes.number,
};

// Set default props for when they aren't provided
Loader.defaultProps = {
  message: 'Processing...',
  progress: 0,
};

export default Loader;