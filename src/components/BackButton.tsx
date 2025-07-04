import { CircleChevronLeft } from "@/props/creative/icons/CircleChevronLeft";
import React from "react";

const BackButton = () => {
  return (
    <button
      className="text-white hover:text-gray-300 transition-colors duration-200 cursor-pointer"
      onClick={() => {
        if (window.history.length > 1) {
          window.history.back();
        } else {
          window.location.href = "/";
        }
      }}
    >
      <CircleChevronLeft />
    </button>
  );
};

export default BackButton;
