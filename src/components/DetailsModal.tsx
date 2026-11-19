"use client";

import React, { useEffect } from "react";

export type PrototypeDetails = {
  problem: string;
  approach: string;
  challenges: string;
  optimizations: string;
  improvements: string;
};

type DetailsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  details: PrototypeDetails;
};

const DetailsModal = ({
  isOpen,
  onClose,
  title,
  details,
}: DetailsModalProps) => {
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sections = [
    { label: "Problem", content: details.problem },
    { label: "Approach", content: details.approach },
    { label: "Challenges", content: details.challenges },
    { label: "Optimizations", content: details.optimizations },
    { label: "What I’d improve next", content: details.improvements },
  ].filter((section) => section.content?.trim());

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#17251f]/45 p-3 backdrop-blur-sm sm:p-6"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="prototype-details-title"
        className="flex max-h-[min(88dvh,820px)] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-[#dce6df] bg-[#fbfcfa] shadow-[0_28px_90px_-28px_rgba(12,34,24,.42)]"
      >
        <header className="flex shrink-0 items-start justify-between gap-5 border-b border-[#e3eae4] bg-white px-5 py-5 sm:px-7 sm:py-6">
          <div className="min-w-0">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#64836e]">Project notes</p>
            <h2 id="prototype-details-title" className="text-xl font-semibold tracking-tight text-[#20382d] sm:text-2xl">{title}</h2>
          </div>
          <button
            onClick={onClose}
            className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-[#e0e8e1] bg-white text-[#687a6e] transition hover:border-[#bed3c3] hover:bg-[#f2f7f2] hover:text-[#264e38] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#8fb39a]"
            aria-label="Close project notes"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-2 sm:px-7 sm:py-3">
          <div className="grid gap-x-10 md:grid-cols-2">
            {sections.map((section, index) => (
              <article key={section.label} className={`border-b border-[#e5ebe6] py-4 sm:py-5 ${index === 0 ? "md:col-span-2" : ""}`}>
                <div className="mb-2 flex items-center gap-2">
                  <span className="font-mono text-[10px] text-[#8da596]">{String(index + 1).padStart(2, "0")}</span>
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#4b7659]">{section.label}</h3>
                </div>
                <p className="text-[13px] leading-6 text-[#56675c] sm:text-sm sm:leading-7">{section.content}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default DetailsModal;
