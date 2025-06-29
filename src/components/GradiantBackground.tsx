"use client";

import { useState, useEffect, useRef } from "react";
import clsx from "clsx";

const gradients = [
  "from-gray-700 via-gray-900 to-black",
  "from-slate-800 via-slate-900 to-black",
  "from-zinc-700 via-zinc-900 to-black",
  "from-gray-900 via-slate-800 to-neutral-800",
  "from-black via-gray-800 to-black",
  "from-gray-900 via-neutral-800 to-gray-900",
  "from-gray-800 via-gray-700 to-gray-600",
  "from-gray-800 via-black to-gray-900",
  "from-gray-900 via-gray-800 to-gray-700",
  "from-gray-700 via-gray-600 to-gray-500",
];

export default function GradientBackground() {
  const [gradientIndex, setGradientIndex] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleClick = () => {
    let nextIndex = Math.floor(Math.random() * gradients.length);
    while (nextIndex === gradientIndex) {
      nextIndex = Math.floor(Math.random() * gradients.length);
    }
    setGradientIndex(nextIndex);

    // Play click sound
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
    }
  };

  return (
    <div
      onClick={handleClick}
      className={clsx(
        "fixed inset-0 transition-all duration-1000 bg-gradient-to-br",
        gradients[gradientIndex]
      )}
    >
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src="/assets/sounds/water-drop.mp3"
        preload="auto"
      />
    </div>
  );
}
