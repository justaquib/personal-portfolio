import BackButton from "@/components/BackButton";
import DoodlePredictor from "@/components/DoodlePredictor";
// import HomeButton from "@/components/HomeButton";
import React from "react";

export default function DoodlePredictorPage() {
  return (
    <main className="relative min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
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