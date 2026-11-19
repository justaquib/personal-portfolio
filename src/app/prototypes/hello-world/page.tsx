"use client";
import BackButton from '@/components/BackButton';
import DetailsModal from '@/components/DetailsModal';
import prototypeLists from '@/utils/json/prototypeList.json';
import { TyperwriterText } from '@/props/creative/TypewriterText';
import { RotateCcw } from 'lucide-react';
import React, { useState } from 'react';

export default function HelloWorld() {
  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'hello-world');
  const [showDetails, setShowDetails] = useState(false);
  const [currentText, setCurrentText] = useState('Hello, World!');
  const [draftText, setDraftText] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [animationKey, setAnimationKey] = useState(0);

  const startEditing = () => {
    setDraftText(currentText);
    setIsEditing(true);
  };

  const submitText = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextText = draftText.trim();
    if (!nextText) return;
    setCurrentText(nextText);
    setAnimationKey(key => key + 1);
    setIsEditing(false);
  };

  return (
    <main className="relative min-h-screen bg-black text-white flex items-center justify-center text-3xl font-mono p-8">
      {/* Floating Details Button */}
      <button
        onClick={() => setShowDetails(true)}
        className="absolute top-4 right-4 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
        title="View Details"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'Hello World'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      <div className="absolute top-4 left-4">
        <BackButton />
      </div>
      <div className="flex w-full flex-col items-center gap-8 text-center">
        {isEditing ? (
          <form onSubmit={submitText} className="flex w-full max-w-xl flex-col items-center gap-5">
            <label className="flex w-full items-center gap-3 border-b border-white/40 px-2 pb-3 text-left focus-within:border-cyan-300">
              <span className="animate-pulse text-cyan-300" aria-hidden="true">_</span>
              <input
                autoFocus
                type="text"
                value={draftText}
                onChange={(event) => setDraftText(event.target.value)}
                placeholder="Type your own word or phrase"
                aria-label="Type your own word or phrase"
                maxLength={80}
                className="min-w-0 flex-1 bg-transparent text-2xl text-white outline-none placeholder:text-base placeholder:text-gray-500 sm:text-4xl"
              />
            </label>
            <div className="flex items-center gap-3 text-sm">
              <button
                type="submit"
                disabled={!draftText.trim()}
                className="rounded-full bg-white px-5 py-2 font-semibold text-black transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Type it out
              </button>
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="rounded-full px-4 py-2 text-gray-400 transition hover:text-white"
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <button
              type="button"
              onClick={startEditing}
              className="group flex flex-col items-center gap-5 rounded-xl px-3 py-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              aria-label={`${currentText}. Click to type your own text.`}
            >
              <TyperwriterText key={`${currentText}-${animationKey}`} text={currentText} className="text-3xl sm:text-5xl md:text-6xl" />
              <span className="text-xs text-gray-500 transition group-hover:text-gray-300 group-focus-visible:text-gray-300">
                Click to type your own text
              </span>
            </button>
            <button
              type="button"
              onClick={() => setAnimationKey(key => key + 1)}
              className="inline-flex items-center gap-2 rounded-full border border-white/15 px-4 py-2 text-sm text-gray-400 transition hover:border-white/40 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              aria-label="Replay the typing animation"
            >
              <RotateCcw className="h-4 w-4" /> Redo
            </button>
          </div>
        )}
      </div>
    </main>
  );
}
