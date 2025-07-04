import { BASE_URL } from "@/constants";
import GlareHover from "@/props/creative/GlareHover";
import { Home } from "lucide-react";
import Link from "next/link";
import React from "react";

const HomeButton = () => {
  return (
    <Link href={BASE_URL} className="p-2">
      <GlareHover
        glareColor="#ffffff"
        glareOpacity={0.3}
        glareAngle={-30}
        glareSize={300}
        transitionDuration={800}
        playOnce={false}
        className="!w-8 !h-8 !rounded-full !bg-slate-700/40 !border-white hover:!bg-slate-700"
      >
        <Home className="w-4.5 h-4.5" />
      </GlareHover>
    </Link>
  );
};

export default HomeButton;
