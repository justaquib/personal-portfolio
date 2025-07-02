"use client";

import {
  useState,
  useRef,
  forwardRef,
  useImperativeHandle,
  useCallback,
} from "react";
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

export interface GradientBackgroundHandle {
  next: () => void;
}
const GradientBackground = forwardRef<GradientBackgroundHandle>((_, ref) => {
  const [idx, setIdx] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);

  const next = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.play().catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    }

    setIdx((prev) => {
      let nextIdx = Math.floor(Math.random() * gradients.length);
      while (nextIdx === prev) {
        nextIdx = Math.floor(Math.random() * gradients.length);
      }
      return nextIdx;
    });
  }, [gradients.length]);

  useImperativeHandle(ref, () => ({ next }), [next]);

  return (
    <div
      className={clsx(
        "fixed inset-0 z-0 pointer-events-none",
        "bg-gradient-to-br transition-all duration-1000",
        gradients[idx]
      )}
    >
      <audio
        ref={audioRef}
        src="/assets/sounds/water-drop.mp3"
        preload="auto"
      />
    </div>
  );
});
GradientBackground.displayName = "GradientBackground";
export default GradientBackground;
