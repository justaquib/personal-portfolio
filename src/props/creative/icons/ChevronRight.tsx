"use client";

import React from "react";
import { motion, useAnimation, type Variants } from "framer-motion";

interface ChevronRightProps extends React.SVGAttributes<SVGSVGElement> {
  width?: number;
  height?: number;
  strokeWidth?: number;
  stroke?: string;
}

const chevronVariants: Variants = {
  normal: {
    x: 0,
    opacity: 1,
  },
  animate: {
    x: [4, 0],
    opacity: [0.3, 1],
    transition: {
      duration: 0.5,
      ease: "easeOut",
    },
  },
};

export const ChevronRight: React.FC<ChevronRightProps> = ({
  width = 28,
  height = 28,
  strokeWidth = 2,
  stroke = "#ffffff",
  ...props
}) => {
  const controls = useAnimation();

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
          d="m9 18 6-6-6-6"
          variants={chevronVariants}
          initial="normal"
          animate={controls}
        />
      </svg>
    </motion.div>
  );
};
