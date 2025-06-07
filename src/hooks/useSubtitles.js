// hooks/useSubtitles.js
import { useEffect, useState } from 'react';

const useSubtitles = (isRecording, mockSubtitles) => {
  const [subtitles, setSubtitles] = useState('');
  const [currentSubtitleIndex, setIndex] = useState(0);

  useEffect(() => {
    if (!isRecording) return;
    const id = setInterval(() => {
      setSubtitles(prev =>
        `${prev}\n${mockSubtitles[currentSubtitleIndex % mockSubtitles.length]}`
          .split('\n').slice(-10).join('\n')
      );
      setIndex(prev => prev + 1);
    }, 2200);
    return () => clearInterval(id);
  }, [isRecording, mockSubtitles, currentSubtitleIndex]);

  return subtitles;
};

export default useSubtitles;
