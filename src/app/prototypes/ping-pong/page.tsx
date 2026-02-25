"use client";
import BackButton from '@/components/BackButton';
import DetailsModal from '@/components/DetailsModal';
import prototypeLists from '@/utils/json/prototypeList.json';
import React, { useEffect, useRef, useState } from 'react';

export default function PingPong() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [score, setScore] = useState({ player: 0, computer: 0 });
  const [gameStarted, setGameStarted] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  
  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'ping-pong');
  const [showDetails, setShowDetails] = useState(false);
  
  // Use refs to track game state inside the game loop
  const gameStartedRef = useRef(false);
  const isPausedRef = useRef(false);

  // Keep refs in sync with state
  useEffect(() => {
    gameStartedRef.current = gameStarted;
    isPausedRef.current = isPaused;
  }, [gameStarted, isPaused]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Game variables
    const paddleWidth = 10;
    const paddleHeight = 80;
    const ballSize = 10;

    let playerY = canvas.height / 2 - paddleHeight / 2;
    let computerY = canvas.height / 2 - paddleHeight / 2;
    let ballX = canvas.width / 2;
    let ballY = canvas.height / 2;
    let ballSpeedX = 5;
    let ballSpeedY = 5;

    // Draw function
    const draw = () => {
      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw paddles
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, playerY, paddleWidth, paddleHeight);
      ctx.fillRect(canvas.width - paddleWidth, computerY, paddleWidth, paddleHeight);

      // Draw ball
      ctx.beginPath();
      ctx.arc(ballX, ballY, ballSize / 2, 0, Math.PI * 2);
      ctx.fill();

      // Draw center line
      ctx.setLineDash([5, 15]);
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 0);
      ctx.lineTo(canvas.width / 2, canvas.height);
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Update game logic
    const update = () => {
      if (!gameStartedRef.current || isPausedRef.current) return;

      // Move ball
      ballX += ballSpeedX;
      ballY += ballSpeedY;

      // Ball collision with top and bottom
      if (ballY <= 0 || ballY >= canvas.height) {
        ballSpeedY = -ballSpeedY;
      }

      // Ball collision with paddles
      // Player paddle (left side) - only bounce if ball is moving left and hitting the paddle
      if (
        ballSpeedX < 0 &&
        ballX - ballSize / 2 <= paddleWidth &&
        ballX + ballSize / 2 >= 0 &&
        ballY >= playerY &&
        ballY <= playerY + paddleHeight
      ) {
        ballSpeedX = -ballSpeedX;
      }

      // Computer paddle (right side) - only bounce if ball is moving right and hitting the paddle
      if (
        ballSpeedX > 0 &&
        ballX + ballSize / 2 >= canvas.width - paddleWidth &&
        ballX - ballSize / 2 <= canvas.width &&
        ballY >= computerY &&
        ballY <= computerY + paddleHeight
      ) {
        ballSpeedX = -ballSpeedX;
      }

      // Score points - only when ball goes completely out of bounds
      if (ballX < -ballSize) {
        setScore(prev => ({ ...prev, computer: prev.computer + 1 }));
        resetBall();
      }

      if (ballX > canvas.width + ballSize) {
        setScore(prev => ({ ...prev, player: prev.player + 1 }));
        resetBall();
      }

      // Computer AI
      if (computerY + paddleHeight / 2 < ballY) {
        computerY += 3;
      } else {
        computerY -= 3;
      }

      // Keep computer paddle in bounds
      computerY = Math.max(0, Math.min(canvas.height - paddleHeight, computerY));
    };

    const resetBall = () => {
      ballX = canvas.width / 2;
      ballY = canvas.height / 2;
      ballSpeedX = -ballSpeedX;
      ballSpeedY = Math.random() > 0.5 ? 5 : -5;
    };

    // Mouse movement for player paddle
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mouseY = e.clientY - rect.top;
      playerY = mouseY - paddleHeight / 2;
      playerY = Math.max(0, Math.min(canvas.height - paddleHeight, playerY));
    };

    canvas.addEventListener('mousemove', handleMouseMove);

    // Game loop
    const gameLoop = () => {
      update();
      draw();
      requestAnimationFrame(gameLoop);
    };

    gameLoop();

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const startGame = () => {
    setGameStarted(true);
    setIsPaused(false);
    setScore({ player: 0, computer: 0 });
  };

  const pauseGame = () => {
    setIsPaused(true);
  };

  const resumeGame = () => {
    setIsPaused(false);
  };

  const endGame = () => {
    setGameStarted(false);
    setIsPaused(false);
    setScore({ player: 0, computer: 0 });
  };

  return (
    <main className="relative min-h-screen bg-black text-white flex flex-col items-center justify-center p-8">
      {/* Floating Details Button */}
      <button
        onClick={() => setShowDetails(true)}
        className="absolute top-4 right-4 z-10 p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
        title="View Details"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'Ping Pong'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />
      <div className="absolute top-4 left-4">
        <BackButton />
      </div>

      <div className="text-center mb-4">
        <h1 className="text-4xl font-bold mb-2">Ping Pong</h1>
        <div className="text-xl">
          Player: {score.player} | Computer: {score.computer}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={800}
        height={400}
        className="border border-white bg-black"
      />

      {!gameStarted && (
        <button
          onClick={startGame}
          className="mt-4 px-6 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
        >
          Play
        </button>
      )}

      {gameStarted && !isPaused && (
        <button
          onClick={pauseGame}
          className="mt-4 px-6 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
        >
          Pause
        </button>
      )}

      {gameStarted && isPaused && (
        <button
          onClick={resumeGame}
          className="mt-4 px-6 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
        >
          Resume
        </button>
      )}

      {gameStarted && (
        <button
          onClick={endGame}
          className="mt-4 px-6 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
        >
          End Game
        </button>
      )}

      <div className="mt-4 text-sm text-gray-400 text-center">
        Move your mouse to control the left paddle
      </div>
    </main>
  );
}