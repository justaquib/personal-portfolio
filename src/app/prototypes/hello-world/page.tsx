"use client";
import BackButton from '@/components/BackButton';
import { TyperwriterText } from '@/props/creative/TypewriterText';
import React from 'react';

export default function HelloWorld() {
  return (
    <main className="relative min-h-screen bg-black text-white flex items-center justify-center text-3xl font-mono p-8">
      <div className="absolute top-4 left-4">
        <BackButton />
      </div>
      <TyperwriterText text="Hello, World!" />
    </main>
  );
}
