import { TyperwriterText } from '@/props/creative/TypewriterText'
import React from 'react'

export default function HelloWorld() {
  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center text-3xl font-mono p-8">
      <TyperwriterText text="Hello World" />
    </main>
  )
}
