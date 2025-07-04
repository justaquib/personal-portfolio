import AnimatedList from "@/props/creative/AnimatedList";
import React from "react";

const PrototypeList = () => {
  const items = [
    "Hello World",
    "Item 2",
    "Item 3",
    "Item 4",
    "Item 5",
    "Item 6",
    "Item 7",
    "Item 8",
    "Item 9",
    "Item 10",
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
