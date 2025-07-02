import React from "react";

const BackButton = () => {
  return (
    <button
      className="absolute top-4 left-4 text-white hover:text-gray-300 transition-colors duration-200 cursor-pointer"
      onClick={() => {
        if (window.history.length > 1) {
          window.history.back();
        } else {
          window.location.href = "/";
        }
      }}
    >
      ← Back
    </button>
  );
};

export default BackButton;
