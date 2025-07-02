"use client";
import CursorGuides from "@/components/CursorGuides";
import GradientBackground, {
  GradientBackgroundHandle,
} from "@/components/GradiantBackground";
import ScrollIndicator from "@/components/ScrollIndicator";
import ShinyText from "@/props/creative/ShinyText";
import Image from "next/image";
import { useRef } from "react";
import { motion } from "framer-motion";
import TrueFocus from "@/props/creative/TrueFocus";
import { CodeXml, Download, Mail } from "lucide-react";
import Magnet from "@/props/creative/Magnet";
import SpotlightCard from "@/props/creative/SpotlightCard";
import Link from "next/link";

export default function Home() {
  const bgRef = useRef<GradientBackgroundHandle>(null);
  const sectionVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: { opacity: 1, y: 0 },
  };
  return (
    <main
      onClick={() => bgRef.current?.next()}
      className="snap-y snap-mandatory scroll-smooth overflow-y-auto h-screen hide-scrollbar"
    >
      <CursorGuides />
      <GradientBackground ref={bgRef} />
      <motion.section
        className="relative z-10 h-screen snap-start flex items-center justify-center text-white"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ amount: 0.3, once: false }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        <div className="z-10 text-center">
          <Image
            src={"/assets/imgs/justaquib.png"}
            alt="Aquib Shahbaz"
            width={400}
            height={400}
            className="rounded-full mx-auto mb-4"
          />
          <ShinyText
            className="text-lg"
            // text={"Engineer by Logic • Designer by Heart"}
            text="A passionate engineer and designer blending logic with creativity."
          />
        </div>
        <ScrollIndicator />
      </motion.section>
      <motion.section
        className="relative z-10 h-screen snap-start flex flex-col items-center justify-center mt-8 text-white"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ amount: 0.3, once: false }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        <div className="w-full h-screen max-w-xl overflow-y-auto hide-scrollbar">
          <div className="w-full mx-auto px-4 sm:p-6 md:p-8">
            <div className="flex justify-between items-center mb-6">
              <TrueFocus
                sentence="Just Aquib"
                manualMode={false}
                blurAmount={5}
                borderColor="white"
                animationDuration={3}
                pauseBetweenAnimations={1}
              />
              <Magnet padding={50} disabled={false} magnetStrength={50}>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.9 }}
                  transition={{ type: "spring", stiffness: 300 }}
                  className="group cursor-pointer border border-white/20 rounded-md px-3 py-1.5 bg-white/10 transition-colors duration-200"
                  onClick={() => window.open("mailto:developer@justaquib.com")}
                  title="Email me @"
                >
                  <Mail className="w-4 h-4 text-white opacity-70 group-hover:opacity-100 transition-opacity duration-200" />
                </motion.div>
              </Magnet>
            </div>
            <div>
              <p className="text-base mb-6 text-wrap">
                Hi, I&apos;m Aquib Shahbaz — a software engineer and creative
                frontend developer focused on building smooth, scalable web apps
                with human-friendly design.
                <br />
                <br />
                I&apos;ve worked across India and the Middle East, crafting
                tools from dashboards to internal platforms — always focused on
                performance and clarity.
                <br />
                <br />
                Recently at Codelogicx, I led the frontend for data-rich
                reporting tools using React, Chart.js, and real-time APIs.
                <br />
                <br />
                Before that, I built Google Calendar-integrated apps and modular
                UI systems at Collegify, and delivered secure full-stack
                projects across domains like finance, education, and
                manufacturing.
                <br />
                <br />
                My goal: take complex ideas and turn them into fast, elegant
                experiences.
              </p>
            </div>
            <div className="flex flex-col z-50 sm:flex-row items-center justify-between gap-4">
              <Link href="/prototypes" className="block">
                <SpotlightCard
                  className="custom-spotlight-card !px-4 !py-2 !bg-slate-800/30 flex flex-row gap-2 justify-center items-center"
                  spotlightColor="rgba(0, 229, 255, 0.2)"
                >
                  <CodeXml />
                  <span className="font-extrabold">prototypes</span>
                </SpotlightCard>
              </Link>
            </div>
          </div>
          <div className="mt-8 p-8 min-h-[400px] bg-black/30 bg-opacity-10 backdrop-blur-md rounded-4xl shadow-lg">
            <div className="flex flex-col sm:flex-row items-center justify-between">
              <div>
                <h2 className="text-3xl font-bold mb-2 text-iceland">Experience</h2>
              </div>
              <a
                href="/assets/docs/Aquib_Shahbaz_Resume.pdf"
                className="flex justify-center items-center w-10 h-10 group cursor-pointer border border-white/20 rounded-full p-3 bg-white/10 transition-colors duration-200"
                title="Download Resume"
                download={true}
              >
                <Download className="opacity-70 group-hover:opacity-100 transition-opacity duration-200" />
              </a>
            </div>
            <hr className="w-full border-t border-white/20 my-2" />
          </div>
        </div>
      </motion.section>
    </main>
  );
}
