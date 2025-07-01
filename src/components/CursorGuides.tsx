"use client";

import { useEffect, useRef } from "react";

export default function CursorGuides() {
  const verticalRef = useRef<HTMLDivElement>(null);
  const horizontalRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const rippleContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = e.clientX;
      const y = e.clientY;

      if (verticalRef.current)
        verticalRef.current.style.transform = `translateX(${x}px)`;
      if (horizontalRef.current)
        horizontalRef.current.style.transform = `translateY(${y}px)`;
      if (dotRef.current)
        dotRef.current.style.transform = `translate(${x}px, ${y}px)`;
    };

    const handleClick = (e: MouseEvent) => {
      const ripple = document.createElement("span");
      ripple.className =
        "absolute w-12 h-12 bg-white/10 rounded-full pointer-events-none animate-ripple";
      ripple.style.left = `${e.clientX - 20}px`;
      ripple.style.top = `${e.clientY - 20}px`;

      rippleContainerRef.current?.appendChild(ripple);
      setTimeout(() => {
        ripple.remove();
      }, 600);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("click", handleClick);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("click", handleClick);
    };
  }, []);

  return (
    <>
      <div
        ref={verticalRef}
        className="fixed top-0 left-0 w-px h-screen bg-white/10 pointer-events-none z-50 transition-transform duration-75 ease-linear"
      />
      <div
        ref={horizontalRef}
        className="fixed top-0 left-0 w-screen h-px bg-white/10 pointer-events-none z-50 transition-transform duration-75 ease-linear"
      />
      <div
        ref={dotRef}
        className="fixed -top-1 -left-1 w-[6px] h-[6px] bg-white/80 rounded-full pointer-events-none z-50 transition-transform duration-75 ease-linear"
      />
      <div
        ref={rippleContainerRef}
        className="fixed top-0 left-0 w-full h-full pointer-events-none z-40"
      />
    </>
  );
}
