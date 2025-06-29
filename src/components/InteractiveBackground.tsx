"use client";

import { useEffect, useState } from "react";
import { Particles, initParticlesEngine } from "@tsparticles/react";
import { loadAll } from "@tsparticles/all";

export default function InteractiveBackground() {
  const [engineReady, setEngineReady] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadAll(engine);
      setEngineReady(true);
    });
  }, []);

  if (!engineReady) return null;

  return (
    <Particles
      id="tsparticles"
      options={{
        fullScreen: { enable: true, zIndex: -1 },
        background: { color: { value: "#0f0f0f" } },
        fpsLimit: 60,
        interactivity: {
          events: {
            onHover: { enable: true, mode: "repulse" },
            resize: { enable: true },
          },
          modes: {
            repulse: { distance: 100, duration: 0.4 },
          },
        },
        particles: {
          number: {
            value: 40,
            density: { enable: true },
          },
          color: { value: "#ffffff" },
          links: {
            enable: true,
            distance: 150,
            color: "#ffffff",
            opacity: 0.2,
            width: 1,
          },
          move: {
            enable: true,
            speed: 1,
            outModes: { default: "bounce" },
          },
          size: { value: { min: 1, max: 3 } },
          opacity: { value: 0.3 },
        },
        detectRetina: true,
      }}
    />
  );
}
