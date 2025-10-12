"use client";

import React from "react";
import { twMerge } from "tailwind-merge";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement>;

export default function Button({
  className,
  children,
  ...props
}: ButtonProps) {
  const baseClasses =
    "px-6 py-2 rounded-lg font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900";
  const defaultVariantClasses = "bg-blue-600 hover:bg-blue-700 focus:ring-blue-500";

  return (
    <button className={twMerge(baseClasses, defaultVariantClasses, className)} {...props}>
      {children}
    </button>
  );
}