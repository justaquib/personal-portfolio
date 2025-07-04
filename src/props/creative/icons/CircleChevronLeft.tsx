"use client";

import React from "react";
import {
  motion,
  useAnimation,
  type Transition,
  type Variants,
} from "framer-motion";

interface CircleChevronLeftProps extends React.SVGAttributes<SVGSVGElement> {
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

const variants: Variants = {
  normal: { x: 0 },
  animate: { x: -2 },
};

export const CircleChevronLeft: React.FC<CircleChevronLeftProps> = ({
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
        <circle cx="12" cy="12" r="10" />
        <motion.path
          d="m14 16-4-4 4-4"
          variants={variants}
          animate={controls}
          initial="normal"
          transition={defaultTransition}
        />
      </svg>
    </motion.div>
  );
};
