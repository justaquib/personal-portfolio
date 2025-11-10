"use client";

import { motion, AnimatePresence } from "framer-motion";
import React, { useState, useEffect, useRef } from "react";
import * as Tone from "tone";

interface TyperwriterTextProps {
  text: string;
  className?: string;
  delay?: number;
  speed?: number;
  loop?: boolean;
}

export const TyperwriterText: React.FC<TyperwriterTextProps> = ({
  text,
  className = "",
  delay = 500,
  speed = 40,
  loop = false,
}) => {
  const [displayedText, setDisplayedText] = useState("");
  const [index, setIndex] = useState(0);
  const synthRef = useRef<Tone.Synth | null>(null);

  useEffect(() => {
    synthRef.current = new Tone.Synth({
      oscillator: {
        type: "square"
      },
      envelope: {
        attack: 0.01,
        decay: 0.1,
        sustain: 0.1,
        release: 0.1
      }
    }).toDestination();

    return () => {
      if (synthRef.current) {
        synthRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    let timeout: NodeJS.Timeout;

    const startTyping = () => {
      if (index < text.length) {
        timeout = setTimeout(() => {
          const newIndex = index + 1;
          setDisplayedText((prev) => prev + text.charAt(index));
          setIndex(newIndex);
          if (synthRef.current) {
            const notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4"];
            const randomNote = notes[Math.floor(Math.random() * notes.length)];
            synthRef.current.triggerAttackRelease(randomNote, "8n");
          }
        }, speed);
      } else if (loop) {
        setTimeout(() => {
          setDisplayedText("");
          setIndex(0);
        }, 1200);
      }
    };

    if (delay && index === 0) {
      timeout = setTimeout(startTyping, delay);
    } else {
      startTyping();
    }

    return () => clearTimeout(timeout);
  }, [index, text, delay, speed, loop]);

  return (
    <div className={`font-mono whitespace-pre text-white ${className}`}>
      {displayedText}
      <AnimatePresence>
        <motion.span
          key="cursor"
          className="inline-block"
          initial={{ opacity: 1 }}
          animate={{ opacity: [1, 0.2, 1] }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          |
        </motion.span>
      </AnimatePresence>
    </div>
  );
};
