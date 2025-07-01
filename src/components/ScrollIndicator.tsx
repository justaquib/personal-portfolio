"use client";

export default function ScrollIndicator() {
  return (
    <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-50">
      <div className="w-[24px] h-[40px] rounded-full border-2 border-white/40 flex items-start justify-center p-1">
        <div className="w-[4px] h-[6px] bg-white/70 rounded-full animate-scroll" />
      </div>
    </div>
  );
}
