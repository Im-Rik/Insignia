import React from 'react';

// Data for the symptom cards
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
 * A single symptom card component, restored to its original design with an image and top-right emoji.
 * @param {object} props - The properties for the component.
 * @param {object} props.symptom - The symptom data to display.
 * @param {function} props.onSelect - The function to call when the card is clicked.
 */
const SymptomCard = ({ symptom, onSelect }) => (
  <div
    onClick={() => onSelect(symptom)}
    className="group relative flex flex-col items-center justify-center p-3 bg-gray-700/50 rounded-xl cursor-pointer
               border border-gray-600/80 hover:border-cyan-400/50
               transform transition-all duration-300 ease-in-out hover:scale-105 hover:bg-gray-700/80 
               shadow-md hover:shadow-cyan-400/20 min-h-[120px]"
  >
    {/* Image in the center */}
    <img
      src={`/images/${symptom.image}`}
      alt={symptom.name}
      onError={(e) => { e.target.onerror = null; e.target.src = `https://placehold.co/48x48/7f1d1d/ffffff?text=IMG`; }}
      className="w-16 h-16 object-cover rounded-lg mb-3 transition-transform duration-300 group-hover:scale-110"
    />
    
    {/* Emoji on the top-right corner */}
    <span className="absolute top-2 right-2 text-xl transition-transform duration-300 group-hover:scale-125">
      {symptom.emoji}
    </span>
    
    {/* Text at the bottom */}
    <p className="text-center text-xs font-medium text-gray-200 group-hover:text-cyan-300 leading-tight">
      {symptom.name}
    </p>
  </div>
);


/**
 * The main component for selecting symptoms.
 * @param {object} props - The properties for the component.
 * @param {function} props.onSymptomSelect - The handler for when a symptom is selected.
 * @param {boolean} props.isMobile - Flag to determine layout style.
 */
const SymptomSelector = ({ onSymptomSelect, isMobile }) => {
  return (
    <div className="flex flex-col h-full w-full">
      {/* Scrollable Symptom Grid */}
      <div className="flex-grow overflow-y-auto scrollbar-thin p-3">
        {/* The grid layout changes based on view. Desktop is single column. */}
        <div className={`grid gap-3 ${isMobile ? 'grid-cols-3 sm:grid-cols-4' : 'grid-cols-1'}`}>
          {symptoms.map((symptom) => (
            <SymptomCard key={symptom.name} symptom={symptom} onSelect={onSymptomSelect} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default SymptomSelector;