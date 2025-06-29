import CursorGuides from "@/components/CursorGuides";
import GradientBackground from "@/components/GradiantBackground";
import InteractiveBackground from "@/components/InteractiveBackground";

export default function Home() {
  return (
    <div className="relative min-h-screen flex items-center justify-center text-white">
      <InteractiveBackground />
      <CursorGuides />
      <GradientBackground />
      <div className="z-10 text-center">
        <h1 className="text-5xl font-bold text-testBlue">Hi, I’m Aquib Shahbaz</h1>
        <p className="text-xl mt-4">Sr Frontend Developer • UI/UX Enthusiast</p>
      </div>
    </div>
  );
}
