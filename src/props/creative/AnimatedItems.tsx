"use client";

import React, {
  useRef,
  MouseEventHandler,
} from "react";
import { motion, useInView } from "framer-motion";

export interface AnimatedItemProps {
  children: React.ReactNode;
  index: number;
  delay?: number;
  isSelected?: boolean;
  onMouseEnter?: MouseEventHandler<HTMLDivElement>;
  onClick?: MouseEventHandler<HTMLDivElement>;
  className?: string;
}

export const AnimatedItem: React.FC<AnimatedItemProps> = ({
  children,
  index,
  delay = 0,
  isSelected = false,
  onMouseEnter,
  onClick,
  className = "",
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { amount: 0.5 });

  return (
    <motion.div
      ref={ref}
      data-index={index}
      onMouseEnter={onMouseEnter}
      onClick={onClick}
      initial={{ scale: 0.7, opacity: 0 }}
      animate={
        inView || isSelected
          ? { scale: 1, opacity: 1 }
          : { scale: 0.7, opacity: 0 }
      }
      transition={{ duration: 0.2, delay }}
      className={`mb-4 cursor-pointer ${className}`}
    >
      {children}
    </motion.div>
  );
};
