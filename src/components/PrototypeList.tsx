import React from "react";
import prototypeLists from "@/utils/json/prototypeList.json";
import { AnimatedList } from "@/props/creative/AnimatedList";
import { makeGetHref } from "@/utils/misc";

type Prototype = {
  id: string | number;
  href: string;
  title: string;
  stack: string;
  slug: string;
  comment: string;
};

type PrototypeListItem = {
  id: string | number;
  title: string;
  stack: string;
  slug: string;
  comment: string;
};

const PrototypeList = () => {
  const projects: Prototype[] = prototypeLists.map((item: PrototypeListItem) => ({
    id: item.id,
    href: item.slug,
    title: item.title,
    stack: item.stack,
    slug: item.slug,
    comment: item.comment,
  }));
  return (
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
          <div>{p.title}</div>
          <div className="text-sm">{p.stack}</div>
        </div>
      )}
      showGradients={false}
      displayScrollbar={false}
    />
  );
};

export default PrototypeList;
