"use client";
import BackButton from "@/components/BackButton";
import DotGrid from "@/props/creative/DotGrid";
import React from "react";

export default function Prototypes() {
  return (
    <div className="fixed h-screen w-screen overflow-hidden bg-gradient-to-br from-gray-700 via-gray-900 to-black text-black">
      <DotGrid
        className="absolute inset-0"
        dotSize={2}
        gap={15}
        baseColor="#ffffff"
        activeColor="#ffffff"
        proximity={140}
        shockRadius={250}
        shockStrength={8}
        resistance={840}
        returnDuration={1.5}
      />

      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="relative w-10/12 max-w-lg h-screen bg-white/10 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden pointer-events-auto">
          <div
            className="
                absolute inset-0
                bg-white/10
                backdrop-blur-xl
                backdrop-saturate-150
                border border-white/40
                rounded-3xl
                shadow-2xl
            "
          />

          <div className="relative z-10 h-full flex flex-col p-8 text-white pointer-events-auto">
            <BackButton />
            <div className="flex-grow flex flex-col justify-center items-center text-center">
              <h1 className="text-3xl font-bold mb-2">Prototypes</h1>
              <p className="text-lg">Interactive Dot Grid Prototype</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
