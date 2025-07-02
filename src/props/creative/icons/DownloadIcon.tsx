"use client";

import React from "react";
import { motion, useAnimation, type Transition } from "framer-motion";

interface DownloadProps extends React.SVGAttributes<SVGSVGElement> {
  width?: number;
  height?: number;
  strokeWidth?: number;
  stroke?: string;
}

const defaultTransition: Transition = {
  type: "spring",
  stiffness: 250,
  damping: 25,
};

export const DownloadIcon: React.FC<DownloadProps> = ({
  width = 28,
  height = 28,
  strokeWidth = 2,
  stroke = "#ffffff",
  ...props
}) => {
  const controls = useAnimation();

  const pathVariants = {
    normal: { pathLength: 1, opacity: 1 },
    animate: { pathLength: 1, opacity: 1 },
  };

  const arrowVariants = {
    normal: { y: 0 },
    animate: {
      y: [0, 3, 0],
      transition: {
        type: "tween" as const,
        ease: "easeInOut" as const,
        duration: 0.6,
        repeat: Infinity,
      },
    },
  };

  return (
    <motion.div
      style={{
        cursor: "pointer",
        userSelect: "none",
        padding: 8,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onHoverStart={() => controls.start("animate")}
      onHoverEnd={() => controls.start("normal")}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width={width}
        height={height}
        viewBox="0 0 24 24"
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        {...props}
      >
        <motion.path
          d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"
          variants={pathVariants}
          initial="normal"
          animate={controls}
          transition={defaultTransition}
        />

        <motion.g variants={arrowVariants} initial="normal" animate={controls}>
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" x2="12" y1="15" y2="3" />
        </motion.g>
      </svg>
    </motion.div>
  );
};
