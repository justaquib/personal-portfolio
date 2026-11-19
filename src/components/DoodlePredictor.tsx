"use client";

import Button from "@/components/Button";
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { labels } from "@/constants/doodleLabels";

const modelJson = "/assets/model/doodleModel.json";

type Prediction = {
  label: string;
  confidence: number;
};

const doodleExamples: Record<string, string> = {
  apple: "🍎", star: "★", moon: "☾", mountain: "⛰", cup: "☕",
  tree: "🌳", house: "⌂", snowman: "☃", hat: "♧", camera: "▣",
  sun: "☼", cloud: "☁", umbrella: "☂", face: "☺", banana: "🍌",
  car: "🚗", bicycle: "🚲", fish: "🐟", flower: "✿", heart: "♡",
  book: "▤", pencil: "✎", clock: "◷", phone: "▯", key: "⚿",
};

export default function DoodlePredictor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedExample, setSelectedExample] = useState<string | null>(null);

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

    canvas.width = 1000;
    canvas.height = 560;
    context.fillStyle = "#f3f2ec";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "#35432d";
    context.lineWidth = 14;
    contextRef.current = context;
  }, []);

  const getCoords = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if ("touches" in e.nativeEvent) {
      return {
        x: ((e.nativeEvent.touches[0].clientX - rect.left) / rect.width) * canvas.width,
        y: ((e.nativeEvent.touches[0].clientY - rect.top) / rect.height) * canvas.height,
      };
    }
    return {
      x: (e.nativeEvent.offsetX / rect.width) * canvas.width,
      y: (e.nativeEvent.offsetY / rect.height) * canvas.height,
    };
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
      context.fillStyle = "#f3f2ec";
      context.fillRect(0, 0, canvas.width, canvas.height);
      setPredictions([]);
      setSelectedExample(null);
    }
  };

  const showExample = (label: string) => {
    const canvas = canvasRef.current;
    const context = contextRef.current;
    if (!canvas || !context) return;
    context.fillStyle = "#f3f2ec";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = "240px system-ui, Apple Color Emoji, Segoe UI Emoji";
    context.fillStyle = "#35432d";
    context.fillText(doodleExamples[label] || label, canvas.width / 2, canvas.height / 2);
    context.restore();
    setSelectedExample(label);
    setPredictions([]);
  };

  const predict = async () => {
    if (!model || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = tf.browser.fromPixels(imageData, 1);
    const tensor = tf.tidy(() => tf.image
      .resizeBilinear(inputTensor, [28, 28])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(0));

    const result = model.predict(tensor) as tf.Tensor;
    const predictionsArray = await result.data();
    inputTensor.dispose();
    tensor.dispose();
    result.dispose();

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
    <div className="flex min-h-0 w-full flex-1 flex-col gap-3 sm:gap-4">
      <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,1.15fr)_minmax(0,0.85fr)] gap-3 lg:grid-cols-[minmax(0,1.55fr)_minmax(280px,0.75fr)] lg:grid-rows-1">
      <section className="flex min-h-0 flex-col overflow-hidden rounded-[28px] border border-[#e4e2d9] bg-white shadow-[0_14px_45px_rgba(42,45,36,0.07)]">
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-[#eeede7] px-4 py-3 sm:px-6 sm:py-4">
          <div>
            <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-[#91ad68]" /><p className="text-sm font-semibold text-[#2c3028]">The drawing board</p></div>
            <p className="mt-1 text-[11px] text-[#85877f] sm:text-xs">{selectedExample ? `Example: ${selectedExample}. Reset to make your own drawing.` : "Use a few confident strokes. Your guess appears as soon as you lift."}</p>
          </div>
          <Button onClick={clearCanvas} className="!mt-0 !rounded-full !border !border-[#e2e0d8] !bg-white !px-4 !py-2 !text-xs !font-medium !text-[#505449] hover:!bg-[#f5f4ef]">Reset drawing</Button>
        </div>
        <canvas
          ref={canvasRef}
          width={1000}
          height={560}
          // Mouse Events
          onMouseDown={startDrawing}
          onMouseUp={finishDrawing}
          onMouseLeave={finishDrawing}
          onMouseMove={draw}
          // Touch Events
          onTouchStart={startDrawing}
          onTouchEnd={finishDrawing}
          onTouchMove={draw}
          className="doodle-canvas block min-h-0 w-full flex-1 cursor-crosshair touch-none"
          style={{ backgroundColor: "#f3f2ec", backgroundImage: "radial-gradient(#d9d8ce 0.8px, transparent 0.8px)", backgroundSize: "22px 22px" }}
        />
      </section>
      <div className="grid min-h-0 grid-cols-2 gap-2 sm:gap-3 lg:grid-cols-1 lg:grid-rows-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <section className="min-h-0 overflow-hidden rounded-2xl border border-[#e5e3dc] bg-white px-3 py-2.5 sm:px-5 sm:py-3.5">
          <div className="mb-2 flex items-center justify-between gap-2">
            <div><p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-[#84966d] sm:text-[10px]">Machine guess</p><h2 className="mt-0.5 text-sm font-semibold tracking-tight text-[#262923] sm:text-base">Top matches</h2></div>
            <span className={`rounded-full px-2.5 py-1 text-[9px] font-medium sm:text-[10px] ${model ? "bg-[#edf2e5] text-[#557736]" : "bg-[#f1f0eb] text-[#777a72]"}`}>{model ? "Live" : error ? "Offline" : "Loading"}</span>
          </div>
          {error && <p className="rounded-xl bg-red-50 p-3 text-sm text-red-700">{error}</p>}
          {!model && !error && <p className="text-sm text-[#777a72]">Loading the drawing model…</p>}
          {model && predictions.length === 0 && <p className="rounded-xl bg-[#f6f5f0] px-3 py-2.5 text-[11px] leading-4 text-[#777a72] sm:text-xs">Your best matches will show up here.</p>}
          <ul className="space-y-1 sm:space-y-1.5">
          {predictions.map((p) => (
            <li key={p.label} className="grid grid-cols-[minmax(42px,0.6fr)_minmax(0,1fr)_30px] items-center gap-1.5 text-[10px] text-[#3d4138] sm:grid-cols-[minmax(56px,0.6fr)_minmax(0,1fr)_38px] sm:gap-2 sm:text-xs">
              <span className="capitalize">{p.label}</span>
              <div className="h-2 overflow-hidden rounded-full bg-[#eeede7]">
                <div
                  className="h-full rounded-full bg-[#78964e] transition-[width] duration-300"
                  style={{ width: `${Math.round(p.confidence * 100)}%` }}
                ></div>
              </div>
              <span className="text-right tabular-nums text-[#777a72]">{`${Math.round(p.confidence * 100)}%`}</span>
            </li>
          ))}
          </ul>
        </section>
        <section className="min-h-0 overflow-hidden rounded-2xl border border-[#e5e3dc] bg-[#eeefe7] px-3 py-2.5 sm:px-5 sm:py-3.5">
          <div className="mb-2 flex items-center justify-between gap-2"><div><p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-[#78866a] sm:text-[10px]">Quick Draw vocabulary</p><h2 className="mt-0.5 text-sm font-semibold tracking-tight text-[#30352a] sm:text-base">{labels.length} things to sketch</h2><p className="mt-0.5 text-[9px] leading-3 text-[#777a72] sm:text-[10px]">Tap any item for a visual example.</p></div></div>
          <ul className="grid grid-cols-3 content-start gap-1 xl:grid-cols-4 sm:gap-1.5">
            {labels.map((label) => <li key={label}><button type="button" onClick={() => showExample(label)} aria-pressed={selectedExample === label} className={`w-full rounded-full border px-1 py-1 text-center text-[9px] capitalize leading-none transition-colors sm:px-2 sm:py-1.5 sm:text-[10px] ${selectedExample === label ? "border-[#91ad68] bg-[#dfe8d2] text-[#34452a]" : "border-[#dfe1d6] bg-white/75 text-[#4c5145] hover:border-[#91ad68] hover:bg-white"}`}>{label}</button></li>)}
          </ul>
        </section>
      </div>
      </div>
    </div>
  );
}
