"use client";

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
    <main className="relative flex h-dvh flex-col overflow-hidden bg-[#f5f4ef] px-3 pb-3 pt-3 text-[#252821] sm:px-5 sm:pb-4 sm:pt-4 lg:px-8">
      {/* Floating Details Button */}
      <button
        onClick={() => setShowDetails(true)}
        className="absolute right-3 top-3 z-10 rounded-full border border-[#e1dfd6] bg-white p-2 text-[#62665c] transition-colors hover:bg-[#eeede7] sm:right-5 sm:top-4"
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

      <div className="absolute left-3 top-3 z-10 flex items-center gap-3 sm:left-5 sm:top-4">
        <BackButton />
        {/* <HomeButton /> */}
      </div>
      <div className="mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col pt-12 sm:pt-14">
        <div className="mb-3 flex shrink-0 items-end justify-between gap-4 sm:mb-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#718455] sm:text-xs">A tiny machine learning experiment</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-[-0.045em] sm:text-3xl">Doodle Predictor</h1>
          </div>
          <p className="hidden max-w-sm pb-1 text-right text-sm leading-5 text-[#70736a] sm:block">Draw an object and see what the model recognizes.</p>
        </div>
        <DoodlePredictor />
      </div>
    </main>
  );
}
