import React from 'react';

// Data for the symptom cards, including names, images, and corresponding emojis
const symptoms = [
  { name: 'Headache', image: 'headache.png', emoji: '🤕' },
  { name: 'Back Pain', image: 'backpain.png', emoji: '😖' },
  { name: 'Neck Pain', image: 'neckpain.png', emoji: '😫' },
  { name: 'Stomach Ache', image: 'stomache.png', emoji: '🤢' },
  { name: 'Running Nose', image: 'runningnose.png', emoji: '🤧' },
  { name: 'Watery Nose', image: 'runnignosewater.png', emoji: '💧' },
  { name: 'Sick', image: 'sick.png', emoji: '🤒' },
  { name: 'Tired', image: 'tired.png', emoji: '😴' },
  { name: 'Dizziness', image: 'dizziness.png', emoji: '😵' },
  { name: 'Dry Mouth', image: 'drymouth.png', emoji: '🌵' },
];

/**
 * A single symptom card component.
 * @param {object} props - The properties for the component.
 * @param {object} props.symptom - The symptom data to display.
 * @param {function} props.onSelect - The function to call when the card is clicked.
 */
const SymptomCard = ({ symptom, onSelect }) => (
  <div
    onClick={() => onSelect(symptom)}
    className="group relative flex flex-col items-center justify-center p-3 bg-gray-700/50 rounded-xl cursor-pointer
               border border-gray-600/80 hover:border-cyan-400/50
               transform transition-all duration-300 ease-in-out hover:scale-105 hover:bg-gray-700/80 shadow-md hover:shadow-cyan-400/20"
  >
    {/* Placeholder for the image */}
  <img
    src={`/images/${symptom.image}`}
    alt={symptom.name}
    onError={(e) => { e.target.onerror = null; e.target.src = `https://placehold.co/100x100/FF0000/FFFFFF?text=Error`; }}
    className="w-16 h-16 sm:w-20 sm:h-20 object-cover rounded-lg mb-2"
/>
    <span className="absolute top-1 right-1 text-2xl sm:text-3xl transition-transform duration-300 group-hover:scale-125">
      {symptom.emoji}
    </span>
    <p className="text-center text-xs sm:text-sm font-medium text-gray-200 group-hover:text-cyan-300">
      {symptom.name}
    </p>
  </div>
);

/**
 * The main component for selecting symptoms.
 * @param {object} props - The properties for the component.
 * @param {function} props.onSymptomSelect - The handler for when a symptom is selected.
 */
const SymptomSelector = ({ onSymptomSelect }) => {
  return (
    <div className="w-full bg-gray-800/80 p-3 sm:p-4 rounded-b-xl border-t border-gray-700/60">
      <h3 className="text-center text-sm sm:text-base font-semibold text-gray-300 mb-3 tracking-wider uppercase">
        Select Symptoms
      </h3>
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-10 gap-2 sm:gap-3">
        {symptoms.map((symptom) => (
          <SymptomCard key={symptom.name} symptom={symptom} onSelect={onSymptomSelect} />
        ))}
      </div>
    </div>
  );
};

export default SymptomSelector;