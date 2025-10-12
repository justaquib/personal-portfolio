"use client";

import Button from "@/components/Button";
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs"; // It's good practice to import all of tfjs
import { labels } from "@/constants/doodleLabels";

const modelJson = "/assets/model/doodleModel.json";

type Prediction = {
  label: string;
  confidence: number;
};

export default function DoodlePredictor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel(modelJson);
        setModel(loadedModel);
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "An unknown error occurred.";
        console.error("Error loading model:", errorMessage);
        setError(`Failed to load model: ${errorMessage}`);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.lineCap = "round";
    context.strokeStyle = "white";
    context.lineWidth = 12;
    contextRef.current = context;
  }, []);

  const getCoords = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if ("touches" in e.nativeEvent) {
      return {
        x: e.nativeEvent.touches[0].clientX - rect.left,
        y: e.nativeEvent.touches[0].clientY - rect.top,
      };
    }
    return { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const coords = getCoords(e);
    if (!coords) return;
    contextRef.current?.beginPath();
    contextRef.current?.moveTo(coords.x, coords.y);
    setIsDrawing(true);
  };

  const finishDrawing = () => {
    contextRef.current?.closePath();
    if (isDrawing) {
      setIsDrawing(false);
      predict();
    }
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const coords = getCoords(e);
    if (!coords) return;
    contextRef.current?.lineTo(coords.x, coords.y);
    contextRef.current?.stroke();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (canvas && context) {
      context.fillStyle = "#111827"; // bg-gray-900
      context.fillRect(0, 0, canvas.width, canvas.height);
      setPredictions([]);
    }
  };

  const predict = async () => {
    if (!model || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    let tensor = tf.browser.fromPixels(imageData, 1);
    tensor = tf.image
      .resizeBilinear(tensor, [28, 28])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(0);

    const result = model.predict(tensor) as tf.Tensor;
    const predictionsArray = await result.data();

    const top5 = Array.from(predictionsArray)
      .map((p, i) => ({
        label: labels[i],
        confidence: p as number,
      }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    setPredictions(top5);
  };

  return (
    <div className="flex flex-col md:flex-row items-center gap-8">
      <div className="flex flex-col items-center">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          // Mouse Events
          onMouseDown={startDrawing}
          onMouseUp={finishDrawing}
          onMouseLeave={finishDrawing}
          onMouseMove={draw}
          // Touch Events
          onTouchStart={startDrawing}
          onTouchEnd={finishDrawing}
          onTouchMove={draw}
          className="bg-gray-900 border-2 border-dashed border-gray-500 rounded-lg cursor-crosshair"
        />
        <Button onClick={clearCanvas} className="mt-4">Clear</Button>
      </div>
      <div className="w-full md:w-64 p-4 bg-gray-800 rounded-lg">
        <h2 className="text-2xl font-bold mb-4">Predictions</h2>
        {error && <p className="text-red-400">{error}</p>}
        {!model && !error && <p>Loading model...</p>}
        {model && predictions.length === 0 && <p>Draw something!</p>}
        <ul>
          {predictions.map((p) => (
            <li key={p.label} className="flex justify-between items-center mb-2">
              <span className="capitalize">{p.label}</span>
              <div className="w-2/4 bg-gray-700 rounded-full h-4">
                <div
                  className="bg-green-500 h-4 rounded-full"
                  style={{ width: `${Math.round(p.confidence * 100)}%` }}
                ></div>
              </div>
              <span>{`${Math.round(p.confidence * 100)}%`}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}