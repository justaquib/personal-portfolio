'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Pencil, Square, Circle, Type, MoveRight, MousePointer2,
  Download, Save, Trash2, Users, Copy, Check, X,
  Minus, Plus, Undo, Redo, ZoomIn, ZoomOut, Hand, Triangle, Eraser, Home
} from 'lucide-react';
import Link from 'next/link';
import { BASE_URL } from '@/constants';

// WebSocket connection URL - can be configured via environment variable
const WS_URL = typeof window !== 'undefined' 
  ? (process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080')
  : 'ws://localhost:8080';

// Element types
type ElementType = 'pencil' | 'rectangle' | 'circle' | 'text' | 'arrow' | 'pan' | 'triangle' | 'eraser';

interface CanvasElement {
  id: string;
  type: ElementType;
  x: number;
  y: number;
  width?: number;
  height?: number;
  points?: { x: number; y: number }[];
  text?: string;
  color: string;
  strokeWidth: number;
  createdBy: string;
}

interface CursorPosition {
  x: number;
  y: number;
}

interface UserCursor {
  userId: string;
  username: string;
  color: string;
  cursor: CursorPosition;
}

interface User {
  userId: string;
  username: string;
  color: string;
}

const COLORS = [
  '#1a1a2e', '#e94560', '#0f3460', '#16213e', '#533483',
  '#00d9ff', '#39ff14', '#ff6b35', '#f7dc6f', '#ff69b4'
];

export default function CollaborationBoard() {
  // Room state
  const [roomId, setRoomId] = useState('');
  const [username, setUsername] = useState('');
  const [isInRoom, setIsInRoom] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [userId, setUserId] = useState('');
  const [userColor, setUserColor] = useState('');
  const [users, setUsers] = useState<User[]>([]);
  const [copied, setCopied] = useState(false);

  // Canvas state
  const [elements, setElements] = useState<CanvasElement[]>([]);
  const [cursors, setCursors] = useState<Map<string, UserCursor>>(new Map());
  const [selectedTool, setSelectedTool] = useState<ElementType>('pencil');
  const [selectedColor, setSelectedColor] = useState(COLORS[0]);
  const [strokeWidth, setStrokeWidth] = useState(4);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentElement, setCurrentElement] = useState<CanvasElement | null>(null);
  
  // Zoom and pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // History for undo/redo
  const [history, setHistory] = useState<CanvasElement[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Generate unique ID
  const generateId = () => Math.random().toString(36).substring(2, 10);

  // Add to history for undo/redo
  const addToHistory = useCallback((newElements: CanvasElement[]) => {
    setHistory(prev => {
      const newHistory = prev.slice(0, historyIndex + 1);
      return [...newHistory, newElements];
    });
    setHistoryIndex(prev => prev + 1);
  }, [historyIndex]);

  // Connect to WebSocket
  const connect = useCallback((type: 'create' | 'join', room?: string) => {
    if (!username.trim()) {
      alert('Please enter your name');
      return;
    }

    setIsConnecting(true);
    
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      const payload = {
        type: type === 'create' ? 'create-room' : 'join-room',
        payload: {
          roomId: room || '',
          userId: generateId(),
          username: username.trim()
        }
      };
      ws.send(JSON.stringify(payload));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const { type: msgType, payload } = message;

      switch (msgType) {
        case 'room-created':
        case 'room-joined':
          setUserId(payload.userId);
          setUserColor(payload.color);
          setRoomId(payload.roomId);
          setIsInRoom(true);
          setElements(payload.elements || []);
          setUsers(payload.users || []);
          setHistory([payload.elements || []]);
          setHistoryIndex(0);
          break;

        case 'element-added':
          setElements(prev => {
            const newElements = [...prev, payload.element];
            addToHistory(newElements);
            return newElements;
          });
          break;

        case 'element-updated':
          setElements(prev => {
            const newElements = prev.map(el => 
              el.id === payload.element.id ? payload.element : el
            );
            addToHistory(newElements);
            return newElements;
          });
          break;

        case 'element-deleted':
          setElements(prev => {
            const newElements = prev.filter(el => el.id !== payload.elementId);
            addToHistory(newElements);
            return newElements;
          });
          break;

        case 'cursor-updated':
          setCursors(prev => {
            const newCursors = new Map(prev);
            newCursors.set(payload.userId, payload);
            return newCursors;
          });
          break;

        case 'user-joined':
          setUsers(prev => [...prev, {
            userId: payload.userId,
            username: payload.username,
            color: payload.color
          }]);
          break;

        case 'user-left':
          setUsers(prev => prev.filter(u => u.userId !== payload.userId));
          setCursors(prev => {
            const newCursors = new Map(prev);
            newCursors.delete(payload.userId);
            return newCursors;
          });
          break;

        case 'canvas-cleared':
          setElements([]);
          addToHistory([]);
          break;

        case 'error':
          alert(payload.message);
          setIsConnecting(false);
          break;
      }
    };

    ws.onerror = () => {
      alert('Connection error. Make sure the collaboration server is running.\n\nFor local development: npm run collab\nOr set NEXT_PUBLIC_WS_URL environment variable for external server.');
      setIsConnecting(false);
    };

    ws.onclose = () => {
      setIsInRoom(false);
      setUsers([]);
      setCursors(new Map());
    };
  }, [username, addToHistory]);

  // Send element to server
  const sendElement = useCallback((element: CanvasElement, action: 'add' | 'update' | 'delete') => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const type = action === 'add' ? 'add-element' : 
                 action === 'update' ? 'update-element' : 'delete-element';
    
    wsRef.current.send(JSON.stringify({
      type,
      payload: action === 'delete' 
        ? { elementId: element.id }
        : { element: action === 'add' ? element : { ...element } }
    }));
  }, []);

  // Send cursor position
  const sendCursor = useCallback((x: number, y: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    wsRef.current.send(JSON.stringify({
      type: 'cursor-move',
      payload: { cursor: { x, y } }
    }));
  }, []);

  // Draw arrow
  const drawArrow = (ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, width: number) => {
    const headLength = 15;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    
    // Draw line
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    
    // Draw arrowhead
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - headLength * Math.cos(angle - Math.PI / 6), y2 - headLength * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - headLength * Math.cos(angle + Math.PI / 6), y2 - headLength * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  // Draw on canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Apply zoom and pan
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Draw grid (larger to cover more area)
    ctx.strokeStyle = '#e5e5e5';
    ctx.lineWidth = 1 / zoom;
    const gridSize = 20;
    for (let x = -2000; x <= canvas.width / zoom + 2000; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, -2000);
      ctx.lineTo(x, canvas.height / zoom + 2000);
      ctx.stroke();
    }
    for (let y = -2000; y <= canvas.height / zoom + 2000; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(-2000, y);
      ctx.lineTo(canvas.width / zoom + 2000, y);
      ctx.stroke();
    }

    // Draw all elements
    elements.forEach(element => {
      ctx.strokeStyle = element.color;
      ctx.fillStyle = element.color;
      ctx.lineWidth = element.strokeWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      switch (element.type) {
        case 'pencil':
          if (element.points && element.points.length > 0) {
            ctx.beginPath();
            ctx.moveTo(element.points[0].x, element.points[0].y);
            element.points.forEach(point => {
              ctx.lineTo(point.x, point.y);
            });
            ctx.stroke();
          }
          break;

        case 'rectangle':
          ctx.beginPath();
          ctx.strokeRect(element.x, element.y, element.width || 0, element.height || 0);
          break;

        case 'circle':
          ctx.beginPath();
          const radius = Math.sqrt(
            Math.pow(element.width || 0, 2) + Math.pow(element.height || 0, 2)
          ) / 2;
          ctx.arc(
            element.x + (element.width || 0) / 2,
            element.y + (element.height || 0) / 2,
            Math.abs(radius),
            0,
            2 * Math.PI
          );
          ctx.stroke();
          break;

        case 'triangle':
          ctx.beginPath();
          ctx.moveTo(element.x + (element.width || 0) / 2, element.y);
          ctx.lineTo(element.x + (element.width || 0), element.y + (element.height || 0));
          ctx.lineTo(element.x, element.y + (element.height || 0));
          ctx.closePath();
          ctx.stroke();
          break;

        case 'text':
          ctx.font = `${element.strokeWidth * 4}px sans-serif`;
          ctx.fillText(element.text || '', element.x, element.y);
          break;

        case 'arrow':
          drawArrow(
            ctx,
            element.x,
            element.y,
            element.x + (element.width || 0),
            element.y + (element.height || 0),
            element.color,
            element.strokeWidth
          );
          break;
      }
    });

    // Draw current element being drawn
    if (currentElement) {
      ctx.strokeStyle = currentElement.color;
      ctx.fillStyle = currentElement.color;
      ctx.lineWidth = currentElement.strokeWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      switch (currentElement.type) {
        case 'pencil':
          if (currentElement.points && currentElement.points.length > 0) {
            ctx.beginPath();
            ctx.moveTo(currentElement.points[0].x, currentElement.points[0].y);
            currentElement.points.forEach(point => {
              ctx.lineTo(point.x, point.y);
            });
            ctx.stroke();
          }
          break;

        case 'rectangle':
          ctx.beginPath();
          ctx.strokeRect(
            currentElement.x, 
            currentElement.y, 
            currentElement.width || 0, 
            currentElement.height || 0
          );
          break;

        case 'circle':
          ctx.beginPath();
          const cr = Math.sqrt(
            Math.pow(currentElement.width || 0, 2) + Math.pow(currentElement.height || 0, 2)
          ) / 2;
          ctx.arc(
            currentElement.x + (currentElement.width || 0) / 2,
            currentElement.y + (currentElement.height || 0) / 2,
            Math.abs(cr),
            0,
            2 * Math.PI
          );
          ctx.stroke();
          break;

        case 'arrow':
          drawArrow(
            ctx,
            currentElement.x,
            currentElement.y,
            currentElement.x + (currentElement.width || 0),
            currentElement.y + (currentElement.height || 0),
            currentElement.color,
            currentElement.strokeWidth
          );
          break;

        case 'triangle':
          ctx.beginPath();
          ctx.moveTo(currentElement.x + (currentElement.width || 0) / 2, currentElement.y);
          ctx.lineTo(currentElement.x + (currentElement.width || 0), currentElement.y + (currentElement.height || 0));
          ctx.lineTo(currentElement.x, currentElement.y + (currentElement.height || 0));
          ctx.closePath();
          ctx.stroke();
          break;
      }
    }
    ctx.restore();
  }, [elements, currentElement, zoom, pan]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      drawCanvas();
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawCanvas]);

  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - pan.x) / zoom;
    const y = (e.clientY - rect.top - pan.y) / zoom;
    return { x, y };
  };

  // Zoom and pan functions
  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.1, 3));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.1, 0.3));
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      setZoom(prev => Math.max(0.3, Math.min(3, prev + delta)));
    }
  };

  const isPointInElement = (x: number, y: number, element: CanvasElement): boolean => {
    const padding = 5;
    if (element.type === 'pencil' && element.points) {
      return element.points.some(p => Math.abs(p.x - x) < padding && Math.abs(p.y - y) < padding);
    }
    const ex = element.x, ey = element.y;
    const ew = element.width || 50, eh = element.height || 50;
    return x >= ex - padding && x <= ex + ew + padding && y >= ey - padding && y <= ey + eh + padding;
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (selectedTool !== 'eraser') return;
    const pos = getMousePos(e);
    const clickedElement = [...elements].reverse().find(el => isPointInElement(pos.x, pos.y, el));
    if (clickedElement) {
      // Delete the clicked element
      setElements(prev => {
        const newElements = prev.filter(el => el.id !== clickedElement.id);
        addToHistory(newElements);
        return newElements;
      });
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'delete-element',
          payload: { elementId: clickedElement.id }
        }));
      }
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // If middle mouse button or space+click, start panning
    if (e.button === 1 || (e.button === 0 && selectedTool === 'pan')) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      return;
    }
    
    // Don't start drawing when using pan or eraser
    if (selectedTool === 'pan' || selectedTool === 'eraser') return;
    
    const pos = getMousePos(e);
    // Account for zoom and pan
    const adjustedPos = {
      x: (pos.x - pan.x) / zoom,
      y: (pos.y - pan.y) / zoom
    };
    setIsDrawing(true);

    const baseElement = {
      id: generateId(),
      color: selectedColor,
      strokeWidth,
      createdBy: userId,
      x: pos.x,
      y: pos.y
    };

    if (selectedTool === 'pencil') {
      setCurrentElement({
        ...baseElement,
        type: 'pencil',
        points: [pos]
      });
    } else if (selectedTool === 'text') {
      const text = prompt('Enter text:');
      if (text) {
        const element: CanvasElement = {
          ...baseElement,
          type: 'text',
          text
        };
        setElements(prev => {
          const newElements = [...prev, element];
          addToHistory(newElements);
          return newElements;
        });
        sendElement(element, 'add');
      }
    } else {
      setCurrentElement({
        ...baseElement,
        type: selectedTool,
        width: 0,
        height: 0
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    sendCursor((pos.x - pan.x) / zoom, (pos.y - pan.y) / zoom);

    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
      return;
    }

    if (!isDrawing || !currentElement) return;

    if (currentElement.type === 'pencil') {
      setCurrentElement(prev => prev ? {
        ...prev,
        points: [...(prev.points || []), pos]
      } : null);
    } else {
      setCurrentElement(prev => prev ? {
        ...prev,
        width: pos.x - (prev.x || 0),
        height: pos.y - (prev.y || 0)
      } : null);
    }

    drawCanvas();
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    if (!isDrawing || !currentElement) {
      setIsDrawing(false);
      return;
    }

    if (currentElement.type !== 'text') {
      setElements(prev => {
        const newElements = [...prev, currentElement];
        addToHistory(newElements);
        return newElements;
      });
      sendElement(currentElement, 'add');
    }

    setCurrentElement(null);
    setIsDrawing(false);
  };

  const undo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setElements(history[newIndex]);
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'clear-canvas', payload: {} }));
        history[newIndex].forEach(el => sendElement(el, 'add'));
      }
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setElements(history[newIndex]);
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'clear-canvas', payload: {} }));
        history[newIndex].forEach(el => sendElement(el, 'add'));
      }
    }
  };

  const clearCanvas = () => {
    if (confirm('Clear all elements? This cannot be undone.')) {
      setElements([]);
      addToHistory([]);
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'clear-canvas', payload: {} }));
      }
    }
  };

  const exportAsImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `collab-board-${roomId}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  const saveState = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'save-state', payload: {} }));
      alert('Board state saved!');
    }
  };

  const copyRoomId = () => {
    navigator.clipboard.writeText(roomId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const leaveRoom = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsInRoom(false);
    setRoomId('');
    setElements([]);
    setUsers([]);
    setCursors(new Map());
    setHistory([]);
    setHistoryIndex(-1);
  };

  if (!isInRoom) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 w-full max-w-md shadow-2xl border border-white/20">
          <h1 className="text-3xl font-bold text-white mb-2 text-center">
            Collaboration Board 🎨
          </h1>
          <p className="text-gray-300 text-center mb-8">
            Create or join a room to collaborate in real-time
          </p>

          <div className="space-y-4">
            <div>
              <label className="text-gray-300 text-sm mb-1 block">Your Name</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your name"
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => connect('create')}
                disabled={isConnecting}
                className="flex-1 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
              >
                {isConnecting ? 'Connecting...' : 'Create Room'}
              </button>
            </div>

            <div className="relative flex py-2 items-center">
              <div className="flex-grow border-t border-white/20"></div>
              <span className="flex-shrink mx-4 text-gray-400">OR</span>
              <div className="flex-grow border-t border-white/20"></div>
            </div>

            <div>
              <label className="text-gray-300 text-sm mb-1 block">Room ID</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={roomId}
                  onChange={(e) => setRoomId(e.target.value.toUpperCase())}
                  placeholder="Enter room ID"
                  className="flex-1 px-4 py-3 rounded-lg bg-white/10 border border-white/20 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 uppercase"
                />
                <button
                  onClick={() => connect('join', roomId)}
                  disabled={isConnecting || !roomId}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-all disabled:opacity-50"
                >
                  Join
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-100">
      <div className="bg-white shadow-sm border-b px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href={BASE_URL} className="flex items-center gap-2">
            <div className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
              <Home className="w-5 h-5 text-gray-600" />
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="font-semibold text-gray-800">Room: {roomId}</span>
            <button
              onClick={copyRoomId}
              className="p-1 hover:bg-gray-100 rounded transition-colors"
              title="Copy room ID"
            >
              {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} className="text-gray-500" />}
            </button>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Users size={16} />
            <span>{users.length} user{users.length !== 1 ? 's' : ''}</span>
          </div>

          <div className="flex -space-x-2">
            {users.filter(u => u.userId !== userId).map((user) => (
              <div
                key={user.userId}
                className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold border-2 border-white"
                style={{ backgroundColor: user.color }}
                title={user.username}
              >
                {user.username.charAt(0).toUpperCase()}
              </div>
            ))}
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold border-2 border-white"
              style={{ backgroundColor: userColor }}
              title="You"
            >
              {username.charAt(0).toUpperCase()}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 text-gray-400">
          <button
            onClick={undo}
            disabled={historyIndex <= 0}
            className="p-2 hover:bg-gray-100 rounded-lg disabled:opacity-50 transition-colors"
            title="Undo"
          >
            <Undo size={20} />
          </button>
          <button
            onClick={redo}
            disabled={historyIndex >= history.length - 1}
            className="p-2 hover:bg-gray-100 rounded-lg disabled:opacity-50 transition-colors"
            title="Redo"
          >
            <Redo size={20} />
          </button>
          <button
            onClick={clearCanvas}
            className="p-2 hover:bg-red-50 text-red-600 rounded-lg transition-colors"
            title="Clear Canvas"
          >
            <Trash2 size={20} />
          </button>
          <button
            onClick={saveState}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title="Save Board State"
          >
            <Save size={20} />
          </button>
          <button
            onClick={exportAsImage}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            title="Export as Image"
          >
            <Download size={20} />
          </button>
          <button
            onClick={leaveRoom}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Leave
          </button>
        </div>
      </div>

      <div className="bg-white border-b px-4 py-2 flex items-center gap-4">
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 text-gray-400">
          <button
            onClick={() => setSelectedTool('eraser')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'eraser' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Eraser"
          >
            <Eraser size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('pan')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'pan' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Pan/Move (or hold Space)"
          >
            <Hand size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('pencil')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'pencil' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Pencil"
          >
            <Pencil size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('rectangle')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'rectangle' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Rectangle"
          >
            <Square size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('circle')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'circle' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Circle"
          >
            <Circle size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('text')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'text' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Text"
          >
            <Type size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('arrow')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'arrow' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Arrow"
          >
            <MoveRight size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('triangle')}
            className={`p-2 rounded-lg transition-colors ${selectedTool === 'triangle' ? 'bg-white shadow-sm' : 'hover:bg-white/50'}`}
            title="Triangle"
          >
            <Triangle size={20} />
          </button>
        </div>

        <div className="w-px h-8 bg-gray-200"></div>

        <div className="flex items-center gap-1">
          {COLORS.map((color) => (
            <button
              key={color}
              onClick={() => setSelectedColor(color)}
              className={`w-7 h-7 rounded-full border-2 transition-all ${selectedColor === color ? 'scale-110 border-gray-800' : 'border-transparent hover:scale-105'}`}
              style={{ backgroundColor: color }}
            />
          ))}
        </div>

        <div className="w-px h-8 bg-gray-200"></div>

        <div className="flex items-center gap-2 text-gray-400">
          <button
            onClick={() => setStrokeWidth(Math.max(2, strokeWidth - 2))}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <Minus size={16} />
          </button>
          <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gray-800 transition-all"
              style={{ width: `${(strokeWidth / 12) * 100}%` }}
            />
          </div>
          <button
            onClick={() => setStrokeWidth(Math.min(12, strokeWidth + 2))}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <Plus size={16} />
          </button>
        </div>

        <div className="w-px h-8 bg-gray-200"></div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1 text-gray-400">
          <button onClick={handleZoomOut} className="p-2 hover:bg-gray-100 rounded-lg" title="Zoom Out"><ZoomOut size={20} /></button>
          <span className="text-sm text-gray-600 min-w-[50px] text-center">{Math.round(zoom * 100)}%</span>
          <button onClick={handleZoomIn} className="p-2 hover:bg-gray-100 rounded-lg" title="Zoom In"><ZoomIn size={20} /></button>
        </div>
      </div>

      <div ref={containerRef} className="flex-1 relative overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          onClick={handleCanvasClick}
          className={`absolute inset-0 ${selectedTool === 'pan' ? 'cursor-grab' : isPanning ? 'cursor-grabbing' : 'cursor-crosshair'}`}
        />

        {Array.from(cursors.values()).map((cursor) => (
          <div
            key={cursor.userId}
            className="absolute pointer-events-none transition-all duration-100"
            style={{
              left: cursor.cursor.x,
              top: cursor.cursor.y,
              transform: 'translate(-2px, -2px)'
            }}
          >
            <MousePointer2
              size={20}
              style={{ color: cursor.color }}
              fill={cursor.color}
            />
            <div
              className="absolute left-4 top-4 px-2 py-0.5 rounded text-xs text-white whitespace-nowrap"
              style={{ backgroundColor: cursor.color }}
            >
              {cursor.username}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-50 px-4 py-2 text-sm text-gray-600 flex items-center justify-between">
        <div>
          Drawing as <span className="font-semibold" style={{ color: userColor }}>{username}</span>
          {' '}with <span className="font-medium">{selectedTool}</span> tool
        </div>
        <div>
          {elements.length} element{elements.length !== 1 ? 's' : ''} on canvas
        </div>
      </div>
    </div>
  );
}
