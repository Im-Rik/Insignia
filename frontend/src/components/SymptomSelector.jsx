import React from 'react';

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

const SymptomCard = ({ symptom, onSelect, isSelected }) => (
  <div
    onClick={() => onSelect(symptom)}
    className={`group relative flex flex-col items-center justify-center p-3 rounded-xl cursor-pointer
                transform transition-all duration-200 ease-in-out hover:scale-105
                shadow-md hover:shadow-cyan-400/20 min-h-[120px]
                ${isSelected
                  ? 'bg-gray-700 border-2 border-cyan-400'
                  : 'bg-gray-700/50 border-2 border-transparent hover:border-cyan-400/50'
                }`}
  >
    <img
      src={`/images/${symptom.image}`}
      alt={symptom.name}
      onError={(e) => { e.target.onerror = null; e.target.src = `https://placehold.co/48x48/7f1d1d/ffffff?text=IMG`; }}
      className="w-16 h-16 object-cover rounded-lg mb-3 transition-transform duration-300 group-hover:scale-110"
    />
    <span className="absolute top-2 right-2 text-xl transition-transform duration-300 group-hover:scale-125">
      {symptom.emoji}
    </span>
    {/* This text will now also change color on selection */}
    <p className={`text-center text-xs font-medium leading-tight ${isSelected ? 'text-cyan-300' : 'text-gray-200 group-hover:text-cyan-300'}`}>
      {symptom.name}
    </p>
  </div>
);

const SymptomSelector = ({ onSymptomSelect, isMobile, selectedSymptoms = [] }) => {
  return (
    <div className="flex flex-col h-full w-full">
      <div className="flex-grow overflow-y-auto scrollbar-thin p-3">
        <div className={`grid gap-3 ${isMobile ? 'grid-cols-3 sm:grid-cols-4' : 'grid-cols-1'}`}>
          {symptoms.map((symptom) => {
            const isSelected = selectedSymptoms.some(s => s.name === symptom.name);
            
            // --- ADD THIS CONSOLE.LOG ---
            // This logs the status of every card whenever the component re-renders.
            console.log(`RENDERING CARD: ${symptom.name}, isSelected: ${isSelected}`);
            
            return (
              <SymptomCard
                key={symptom.name}
                symptom={symptom}
                onSelect={onSymptomSelect}
                isSelected={isSelected}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default SymptomSelector;