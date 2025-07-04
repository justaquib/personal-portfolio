import AnimatedList from "@/props/creative/AnimatedList";
import React from "react";

const PrototypeList = () => {
  const items = [
    "Hello World ✋",
    "Sonic ⚡️", // maybe a file converter or speech-to-text?
    "Doc Reader 📖", // PDF/Doc AI reader
    "PDF Generator 📄", // Custom resume or report generator
    "AI Chatbot 🤖", // ChatGPT clone with your own UI
    "Doodle Predictor ✏️", // Draw-and-guess AI using TensorFlow.js
    "Code Summarizer 🧠", // AI tool that explains code snippets
    "TaskFlow ✅", // Kanban board or productivity app
    "Mood Journal 🌈", // Daily emotion tracker + AI insights
    "Stock Tracker 📈", // Real-time stock dashboard (maybe use RowGap idea?)
  ];

  return (
    <AnimatedList
      items={items}
      onItemSelect={(item, index) => console.log(item, index)}
      showGradients={false}
      enableArrowNavigation={true}
      displayScrollbar={false}
      itemClassName="!bg-transparent !text-start !border-dashed !border-white border-b !rounded-none !p-2 hover:!ps-4 transition-[padding-inline-start] duration-500 ease-in-out"
    />
  );
};

export default PrototypeList;
