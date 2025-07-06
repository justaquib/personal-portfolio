"use client";

import { motion, AnimatePresence } from "framer-motion";
import React, { useState, useEffect } from "react";

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
  speed = 60,
  loop = false,
}) => {
  const [displayedText, setDisplayedText] = useState("");
  const [index, setIndex] = useState(0);

  useEffect(() => {
    let timeout: NodeJS.Timeout;

    const startTyping = () => {
      if (index < text.length) {
        timeout = setTimeout(() => {
          setDisplayedText((prev) => prev + text.charAt(index));
          setIndex((i) => i + 1);
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
