import React, { useState, useRef } from 'react';
import jsPDF from 'jspdf';

const Prescription = ({ selectedSymptoms, onBack }) => {
  const [patientName, setPatientName] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [patientGender, setPatientGender] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [medications, setMedications] = useState([{ name: '', dosage: '', duration: '' }]);
  const [additionalNotes, setAdditionalNotes] = useState('');
  const [doctorName, setDoctorName] = useState('Dr. ');
  const [doctorQualification, setDoctorQualification] = useState('');
  const [clinicName, setClinicName] = useState('');
  const prescriptionRef = useRef(null);

  // Function to remove emojis and return clean text for PDF
  const removeEmojis = (text) => {
    return text.replace(/[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu, '').trim();
  };

  // Function to get symptom name without emoji for PDF
  const getSymptomForPDF = (symptom) => {
    return symptom.name; // Just return the name without emoji
  };

  const addMedication = () => {
    setMedications([...medications, { name: '', dosage: '', duration: '' }]);
  };

  const removeMedication = (index) => {
    setMedications(medications.filter((_, i) => i !== index));
  };

  const updateMedication = (index, field, value) => {
    const updated = [...medications];
    updated[index][field] = value;
    setMedications(updated);
  };

  const generatePDF = () => {
    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const margin = 20;
    let yPosition = margin;

    // Header
    pdf.setFontSize(20);
    pdf.setFont(undefined, 'bold');
    pdf.text(clinicName || 'Medical Clinic', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 10;

    pdf.setFontSize(12);
    pdf.setFont(undefined, 'normal');
    pdf.text(doctorName + (doctorQualification ? `, ${doctorQualification}` : ''), pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    // Line separator
    pdf.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;

    // Date
    pdf.setFontSize(10);
    pdf.text(`Date: ${new Date().toLocaleDateString()}`, margin, yPosition);
    yPosition += 10;

    // Patient Information
    pdf.setFont(undefined, 'bold');
    pdf.text('Patient Information:', margin, yPosition);
    yPosition += 7;
    pdf.setFont(undefined, 'normal');
    pdf.text(`Name: ${patientName}`, margin, yPosition);
    yPosition += 7;
    pdf.text(`Age: ${patientAge} | Gender: ${patientGender}`, margin, yPosition);
    yPosition += 15;

    // Symptoms (without emojis)
    if (selectedSymptoms && selectedSymptoms.length > 0) {
      pdf.setFont(undefined, 'bold');
      pdf.text('Reported Symptoms:', margin, yPosition);
      yPosition += 7;
      pdf.setFont(undefined, 'normal');
      selectedSymptoms.forEach(symptom => {
        // Use bullet point instead of emoji
        pdf.text(`• ${getSymptomForPDF(symptom)}`, margin + 5, yPosition);
        yPosition += 7;
      });
      yPosition += 8;
    }

    // Diagnosis
    pdf.setFont(undefined, 'bold');
    pdf.text('Diagnosis:', margin, yPosition);
    yPosition += 7;
    pdf.setFont(undefined, 'normal');
    const cleanDiagnosis = removeEmojis(diagnosis);
    const diagnosisLines = pdf.splitTextToSize(cleanDiagnosis, pageWidth - 2 * margin);
    diagnosisLines.forEach(line => {
      pdf.text(line, margin, yPosition);
      yPosition += 7;
    });
    yPosition += 8;

    // Medications
    pdf.setFont(undefined, 'bold');
    pdf.text('Prescription:', margin, yPosition);
    yPosition += 7;
    pdf.setFont(undefined, 'normal');
    medications.forEach((med, index) => {
      if (med.name) {
        const cleanMedName = removeEmojis(med.name);
        const cleanDosage = removeEmojis(med.dosage);
        const cleanDuration = removeEmojis(med.duration);
        
        pdf.text(`${index + 1}. ${cleanMedName}`, margin + 5, yPosition);
        yPosition += 7;
        pdf.text(`   Dosage: ${cleanDosage} | Duration: ${cleanDuration}`, margin + 5, yPosition);
        yPosition += 10;
      }
    });

    // Additional Notes
    if (additionalNotes) {
      yPosition += 5;
      pdf.setFont(undefined, 'bold');
      pdf.text('Additional Notes:', margin, yPosition);
      yPosition += 7;
      pdf.setFont(undefined, 'normal');
      const cleanNotes = removeEmojis(additionalNotes);
      const notesLines = pdf.splitTextToSize(cleanNotes, pageWidth - 2 * margin);
      notesLines.forEach(line => {
        pdf.text(line, margin, yPosition);
        yPosition += 7;
      });
    }

    // Footer
    yPosition = pdf.internal.pageSize.getHeight() - 30;
    pdf.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;
    pdf.text(doctorName, pageWidth - margin, yPosition, { align: 'right' });

    // Save PDF
    pdf.save(`prescription_${patientName}_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  const shareOnWhatsApp = () => {
    // Keep emojis for WhatsApp version
    const prescriptionText = `
*PRESCRIPTION*
${clinicName ? `*${clinicName}*\n` : ''}
${doctorName}${doctorQualification ? `, ${doctorQualification}` : ''}
Date: ${new Date().toLocaleDateString()}

*Patient:* ${patientName}
*Age:* ${patientAge} | *Gender:* ${patientGender}

*Symptoms:*
${selectedSymptoms.map(s => `• ${s.emoji} ${s.name}`).join('\n')}

*Diagnosis:*
${diagnosis}

*Prescription:*
${medications.filter(m => m.name).map((m, i) => `${i + 1}. ${m.name}\n   Dosage: ${m.dosage} | Duration: ${m.duration}`).join('\n')}

${additionalNotes ? `*Notes:* ${additionalNotes}` : ''}

_Generated on ${new Date().toLocaleString()}_
    `.trim();

    const encodedText = encodeURIComponent(prescriptionText);
    window.open(`https://wa.me/?text=${encodedText}`, '_blank');
  };

  return (
    <div className="flex flex-col h-full w-full bg-gray-900 text-gray-100">
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl sm:text-3xl font-bold text-cyan-400">Create Prescription</h2>
            <button
              onClick={onBack}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors duration-200"
            >
              Back
            </button>
          </div>

          {/* Doctor Information */}
          <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
            <h3 className="text-lg font-semibold mb-4 text-cyan-300">Doctor Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <input
                type="text"
                placeholder="Doctor Name"
                value={doctorName}
                onChange={(e) => setDoctorName(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <input
                type="text"
                placeholder="Qualification (e.g., MBBS, MD)"
                value={doctorQualification}
                onChange={(e) => setDoctorQualification(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <input
                type="text"
                placeholder="Clinic/Hospital Name"
                value={clinicName}
                onChange={(e) => setClinicName(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
            </div>
          </div>

          {/* Patient Information */}
          <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
            <h3 className="text-lg font-semibold mb-4 text-cyan-300">Patient Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <input
                type="text"
                placeholder="Patient Name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <input
                type="text"
                placeholder="Age"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              />
              <select
                value={patientGender}
                onChange={(e) => setPatientGender(e.target.value)}
                className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          {/* Selected Symptoms */}
          {selectedSymptoms && selectedSymptoms.length > 0 && (
            <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
              <h3 className="text-lg font-semibold mb-4 text-cyan-300">Patient's Reported Symptoms</h3>
              <div className="flex flex-wrap gap-2">
                {selectedSymptoms.map((symptom, index) => (
                  <div key={index} className="flex items-center gap-1 bg-gray-600/50 px-3 py-1.5 rounded-full">
                    <span>{symptom.emoji}</span>
                    <span className="text-gray-200">{symptom.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Diagnosis */}
          <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
            <h3 className="text-lg font-semibold mb-4 text-cyan-300">Diagnosis</h3>
            <textarea
              placeholder="Enter diagnosis..."
              value={diagnosis}
              onChange={(e) => setDiagnosis(e.target.value)}
              className="w-full bg-gray-700/50 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-400 min-h-[100px]"
            />
          </div>

          {/* Medications */}
          <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-cyan-300">Medications</h3>
              <button
                onClick={addMedication}
                className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 rounded-lg transition-colors duration-200 text-sm"
              >
                + Add Medication
              </button>
            </div>
            {medications.map((med, index) => (
              <div key={index} className="mb-4 p-4 bg-gray-700/30 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <input
                    type="text"
                    placeholder="Medication Name"
                    value={med.name}
                    onChange={(e) => updateMedication(index, 'name', e.target.value)}
                    className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                  />
                  <input
                    type="text"
                    placeholder="Dosage (e.g., 500mg twice daily)"
                    value={med.dosage}
                    onChange={(e) => updateMedication(index, 'dosage', e.target.value)}
                    className="bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                  />
                  <div className="flex gap-2">
                    <input
                      type="text"
                      placeholder="Duration (e.g., 7 days)"
                      value={med.duration}
                      onChange={(e) => updateMedication(index, 'duration', e.target.value)}
                      className="flex-1 bg-gray-700/50 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                    />
                    {medications.length > 1 && (
                      <button
                        onClick={() => removeMedication(index)}
                        className="px-3 py-2 bg-red-600 hover:bg-red-500 rounded-lg transition-colors duration-200"
                      >
                        ×
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Additional Notes */}
          <div className="bg-gray-800/70 backdrop-blur-md rounded-lg p-4 sm:p-6 mb-6 border border-gray-700/50">
            <h3 className="text-lg font-semibold mb-4 text-cyan-300">Additional Notes</h3>
            <textarea
              placeholder="Any additional instructions or notes..."
              value={additionalNotes}
              onChange={(e) => setAdditionalNotes(e.target.value)}
              className="w-full bg-gray-700/50 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-400 min-h-[80px]"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 justify-center">
            <button
              onClick={generatePDF}
              disabled={!patientName || !diagnosis || medications.every(m => !m.name)}
              className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors duration-200 font-semibold"
            >
              📄 Download PDF
            </button>
            <button
              onClick={shareOnWhatsApp}
              disabled={!patientName || !diagnosis || medications.every(m => !m.name)}
              className="px-6 py-3 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors duration-200 font-semibold"
            >
              💬 Share on WhatsApp
            </button>
          </div>

          {/* Info message */}
          <div className="mt-4 p-3 bg-blue-900/30 border border-blue-700/50 rounded-lg">
            <p className="text-sm text-blue-300">
              <strong>Note:</strong> Emojis will be preserved in WhatsApp sharing but converted to text in PDF format for better compatibility.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Prescription;