"use client";

import React, { useEffect, useMemo } from "react";
import { motion, Variants } from "framer-motion";

interface ExperienceItem {
  company: string;
  position: string;
  duration: string;
  description: string;
}

const ExperienceList: ExperienceItem[] = [
  {
    company: "Codelogicx",
    position: "Sr. Frontend Developer",
    duration: "2023 - Present",
    description:
      "Led development of data visualization tools using React and Chart.js, focusing on performance and user experience.",
  },
  {
    company: "Collegify",
    position: "Software Engineer",
    duration: "2022 - 2023",
    description:
      "Built Google Calendar–integrated apps and modular UI systems, enhancing user engagement and productivity.",
  },
  {
    company: "Abdullah Dossary Group",
    position: "IT Engineer",
    duration: "2019 - 2022",
    description:
      "Delivered secure, scalable web applications across various industries including finance, education, and manufacturing.",
  },
  {
    company: "Arthadut Pvt Ltd",
    position: "Full Stack Developer",
    duration: "2018 - 2019",
    description:
      "Developed custom web solutions for clients, focusing on performance and user experience.",
  },
  {
    company: "Andaman Live Holidays",
    position: "Full Stack Developer",
    duration: "2017 - 2018",
    description:
      "Developed custom web solutions for clients, focusing on performance and user experience.",
  },
];

const listVariants: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2,
    },
  },
};

const itemVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, ease: "easeOut" },
  },
};

interface ExperienceProps {
  setTotalExp: (years: string) => void;
}

const Experience: React.FC<ExperienceProps> = ({ setTotalExp }) => {
  const totalYears = useMemo(() => {
    const currentYear = new Date().getFullYear();
    return ExperienceList.reduce((sum, { duration }) => {
      const [startStr, endStr] = duration.split(" - ").map((s) => s.trim());
      const start = parseInt(startStr, 10);
      const end = endStr === "Present" ? currentYear : parseInt(endStr, 10);
      return sum + (end - start);
    }, 0);
  }, []);

  useEffect(() => {
    const totalExp = `${totalYears} yr${totalYears > 1 && "s"}`
    setTotalExp(totalExp);
  }, [setTotalExp, totalYears]);

  return (
    <motion.ul
      className="max-w-xl mx-auto p-4 space-y-6 list-none"
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      variants={listVariants}
    >
      {ExperienceList.map((exp, idx) => (
        <motion.li
          key={idx}
          variants={itemVariants}
          className="relative pl-8"
        >
          <span className="absolute left-1.5 top-0 bottom-0 w-1 bg-cyan-400 rounded" />

          <div className="flex justify-between items-baseline">
            <h3 className="text-lg font-semibold text-white">
              {exp.company}
            </h3>
            <span className="text-sm text-gray-300">{exp.duration}</span>
          </div>
          <p className="mt-1 text-cyan-200 italic">{exp.position}</p>
          <p className="mt-2 text-gray-200">{exp.description}</p>
        </motion.li>
      ))}
    </motion.ul>
  );
};

export default Experience;
