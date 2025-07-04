"use client";
import BackButton from "@/components/BackButton";
import Breadcrumb from "@/components/Breadcrumb";
import HomeButton from "@/components/HomeButton";
import PrototypeList from "@/components/PrototypeList";
import DotGrid from "@/props/creative/DotGrid";
import React from "react";

export default function Prototypes() {
  return (
    <div className="fixed h-screen w-screen overflow-hidden bg-gradient-to-br from-gray-700 via-gray-900 to-black text-black">
      <DotGrid
        className="absolute inset-0"
        dotSize={1}
        gap={8}
        baseColor="#ffffff"
        activeColor="#ffffff"
        proximity={120}
        shockRadius={240}
        shockStrength={8}
        resistance={840}
        returnDuration={1.5}
      />

      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="relative w-10/12 max-w-lg h-screen bg-neutral-700/10 backdrop-blur-md rounded-3xl shadow-2xl overflow-hidden pointer-events-auto">
          <div className="absolute inset-0 bg-white/10 backdrop-blur-xl backdrop-saturate-150 border border-white/40 rounded-3xl shadow-2xl"/>
          <div className="relative z-10 w-full h-full flex flex-col p-4 text-white pointer-events-auto">
            <div className="flex flex-row justify-between items-center w-full">
                <div className="flex flex-row justify-between items-center gap-2">
                    <BackButton />
                    <Breadcrumb />
                </div>
                <HomeButton />
            </div>
            <div className="flex flex-col h-full justify-start items-start text-center pb-12">
                <PrototypeList />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
