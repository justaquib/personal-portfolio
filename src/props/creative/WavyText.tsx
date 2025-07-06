"use client";

import { easeInOut, motion } from "framer-motion";
import React from "react";

interface WavyTextProps {
  text: string;
  durationPerChar?: number; // in seconds
  className?: string;
}

export const WavyText: React.FC<WavyTextProps> = ({
  text,
  durationPerChar = 0.05,
  className = "",
}) => {
  const letters = Array.from(text);

  const container = {
    hidden: { opacity: 1 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: durationPerChar,
      },
    },
  };

  const child = {
    hidden: { opacity: 0, y: `0.25em` },
    visible: {
      opacity: 1,
      y: `0em`,
      transition: {
        duration: 0.3,
        ease: easeInOut,
      },
    },
  };

  return (
    <motion.span
      variants={container}
      initial="hidden"
      animate="visible"
      className={`inline-flex overflow-hidden ${className}`}
      aria-label={text}
    >
      {letters.map((char, index) => (
        <motion.span
          key={index}
          variants={child}
          className="whitespace-pre"
        >
          {char}
        </motion.span>
      ))}
    </motion.span>
  );
};
