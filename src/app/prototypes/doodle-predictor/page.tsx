import BackButton from "@/components/BackButton";
import DetailsModal from "@/components/DetailsModal";
import prototypeLists from "@/utils/json/prototypeList.json";
import DoodlePredictor from "@/components/DoodlePredictor";
// import HomeButton from "@/components/HomeButton";
import React, { useState } from "react";

export default function DoodlePredictorPage() {
  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'doodle-predictor');
  const [showDetails, setShowDetails] = useState(false);

  return (
    <main className="relative min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      {/* Floating Details Button */}
      <button
        onClick={() => setShowDetails(true)}
        className="absolute top-4 right-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
        title="View Details"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'Doodle Predictor'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      <div className="absolute top-4 left-4 flex items-center gap-4">
        <BackButton />
        {/* <HomeButton /> */}
      </div>
      <h1 className="text-4xl font-bold mb-4">Doodle Predictor</h1>
      <p className="text-lg mb-8">Draw something and see if the AI can guess it!</p>
      <DoodlePredictor />
    </main>
  );
}