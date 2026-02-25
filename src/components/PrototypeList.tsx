"use client";

import React, { useState } from "react";
import prototypeLists from "@/utils/json/prototypeList.json";
import { AnimatedList } from "@/props/creative/AnimatedList";
import { makeGetHref } from "@/utils/misc";
import DetailsModal, { PrototypeDetails } from "./DetailsModal";

type Prototype = {
  id: string | number;
  href: string;
  title: string;
  stack: string;
  slug: string;
  comment: string;
  details?: PrototypeDetails;
};

type PrototypeListItem = {
  id: string | number;
  title: string;
  stack: string;
  slug: string;
  comment: string;
  problem?: string;
  approach?: string;
  challenges?: string;
  optimizations?: string;
  improvements?: string;
};

const PrototypeList = () => {
  const [selectedPrototype, setSelectedPrototype] = useState<{
    title: string;
    details: PrototypeDetails;
  } | null>(null);

  const projects: Prototype[] = prototypeLists.map((item: PrototypeListItem) => ({
    id: item.id,
    href: item.slug,
    title: item.title,
    stack: item.stack,
    slug: item.slug,
    comment: item.comment,
    details: item.problem
      ? {
          problem: item.problem || "",
          approach: item.approach || "",
          challenges: item.challenges || "",
          optimizations: item.optimizations || "",
          improvements: item.improvements || "",
        }
      : undefined,
  }));

  const openModal = (title: string, details: PrototypeDetails) => {
    setSelectedPrototype({ title, details });
  };

  const closeModal = () => {
    setSelectedPrototype(null);
  };

  return (
    <>
      <AnimatedList
        items={projects}
        getKey={(p: Prototype) => String(p.id)}
        getHref={makeGetHref<Prototype>("prototypes")}
        renderItem={(p: Prototype, _, isSel) => (
          <div
            className={`flex flex-row justify-between text-left text-white border-b border-dashed py-2 transition-[padding-inline-start] duration-500 ease-in-out ${
              isSel ? "font-bold ps-2" : ""
            }`}
          >
            <div className="flex items-center gap-3">
              <div>{p.title}</div>
              {p.details && (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    openModal(p.title, p.details!);
                  }}
                  className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded transition-colors"
                >
                  Details
                </button>
              )}
            </div>
            <div className="text-sm">{p.stack}</div>
          </div>
        )}
        showGradients={false}
        displayScrollbar={false}
      />

      {selectedPrototype && (
        <DetailsModal
          isOpen={!!selectedPrototype}
          onClose={closeModal}
          title={selectedPrototype.title}
          details={selectedPrototype.details}
        />
      )}
    </>
  );
};

export default PrototypeList;
