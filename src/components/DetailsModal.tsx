"use client";

import React from "react";

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
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="relative max-w-2xl w-full max-h-[80vh] overflow-y-auto bg-gray-900 border border-gray-700 rounded-xl p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>

        <h2 className="text-2xl font-bold text-white mb-6">{title}</h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wide">
              Problem
            </h3>
            <p className="text-gray-300 mt-1">{details.problem}</p>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h3 className="text-sm font-semibold text-green-400 uppercase tracking-wide">
              Approach
            </h3>
            <p className="text-gray-300 mt-1">{details.approach}</p>
          </div>

          <div className="border-l-4 border-yellow-500 pl-4">
            <h3 className="text-sm font-semibold text-yellow-400 uppercase tracking-wide">
              Challenges
            </h3>
            <p className="text-gray-300 mt-1">{details.challenges}</p>
          </div>

          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="text-sm font-semibold text-purple-400 uppercase tracking-wide">
              Optimizations
            </h3>
            <p className="text-gray-300 mt-1">{details.optimizations}</p>
          </div>

          <div className="border-l-4 border-pink-500 pl-4">
            <h3 className="text-sm font-semibold text-pink-400 uppercase tracking-wide">
              What I'd improve next
            </h3>
            <p className="text-gray-300 mt-1">{details.improvements}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetailsModal;
