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
  '#20352a', '#4f8057', '#557086', '#9a7045', '#b45243',
  '#64748b', '#8b6ca8', '#c17c4a', '#bd9951', '#3f8c8a'
];

export default function CollaborationBoard() {
  // Room state
  const [roomId, setRoomId] = useState('');
  const [roomName, setRoomName] = useState('');
  const [createRoomName, setCreateRoomName] = useState('');
  const [username, setUsername] = useState('');
  const [createUsername, setCreateUsername] = useState('');
  const [joinUsername, setJoinUsername] = useState('');
  const [isInRoom, setIsInRoom] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [formError, setFormError] = useState('');
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
  const connect = useCallback((type: 'create' | 'join', room: string | undefined, participantName: string, name?: string) => {
    setFormError('');
    setIsConnecting(true);

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        const payload = {
          type: type === 'create' ? 'create-room' : 'join-room',
          payload: {
            roomId: room || '',
            userId: generateId(),
            username: participantName.trim(),
            roomName: name?.trim()
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
          setIsConnecting(false);
          setUsername(payload.username || participantName);
          setUserId(payload.userId);
          setUserColor(payload.color);
          setRoomId(payload.roomId);
          setRoomName(payload.roomName || 'Shared room');
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
            setFormError(payload.message || 'Could not join that room. Check the room ID and try again.');
            setIsConnecting(false);
            ws.close();
            break;
        }
      };

      ws.onerror = () => {
        setFormError('Could not connect to the collaboration server. Check that it is running, then try again.');
        setIsConnecting(false);
      };

      ws.onclose = () => {
        setIsConnecting(false);
        setIsInRoom(false);
        setUsers([]);
        setCursors(new Map());
      };
    } catch {
      setFormError('Could not start a room connection. Check the WebSocket address and try again.');
      setIsConnecting(false);
    }
  }, [addToHistory]);

  const handleCreateRoom = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = createUsername.trim();
    const title = createRoomName.trim();
    if (!name || !title) {
      setFormError('Enter the room name and your name to create a room.');
      return;
    }
    setUsername(name);
    connect('create', undefined, name, title);
  };

  const handleJoinRoom = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = joinUsername.trim();
    const code = roomId.trim().toUpperCase();
    if (!name || !code) {
      setFormError('Enter your name and the room code to join.');
      return;
    }
    setUsername(name);
    setRoomId(code);
    connect('join', code, name);
  };

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
    setRoomName('');
    setElements([]);
    setUsers([]);
    setCursors(new Map());
    setHistory([]);
    setHistoryIndex(-1);
  };

  if (!isInRoom) {
    return (
      <main className="min-h-screen bg-[#f5f6f2] text-[#25392d]">
        <div className="mx-auto flex min-h-screen max-w-[1240px] flex-col px-4 sm:px-7 lg:px-10">
          <header className="flex h-[74px] shrink-0 items-center justify-between border-b border-[#e1e6de]">
            <Link href={BASE_URL} className="flex items-center gap-2.5 text-sm font-semibold tracking-tight text-[#2d4433]">
              <span className="grid h-9 w-9 place-items-center rounded-xl bg-[#254f36] text-white"><Users className="h-4 w-4" /></span>
              Collaboration board
            </Link>
            <Link href={BASE_URL} className="inline-flex items-center gap-2 rounded-lg border border-[#dfe6dd] bg-white px-3 py-2 text-xs font-medium text-[#66776a] transition hover:bg-[#f7f9f6]"><Home className="h-3.5 w-3.5" /><span className="hidden sm:inline">Portfolio home</span><span className="sm:hidden">Home</span></Link>
          </header>

          <div className="grid flex-1 items-center gap-10 py-8 lg:grid-cols-[.9fr_1.1fr] lg:gap-16 lg:py-12">
            <section className="mx-auto w-full max-w-lg lg:mx-0">
              <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-[#d8e5d7] bg-white px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-[#66816a]"><span className="h-1.5 w-1.5 rounded-full bg-[#77a078]" />Live collaboration</p>
              <h1 className="max-w-xl text-4xl font-semibold leading-[1.08] tracking-[-0.055em] text-[#21382b] sm:text-5xl">Make ideas visible, together.</h1>
              <p className="mt-5 max-w-md text-sm leading-7 text-[#738178] sm:text-base">Sketch, map, and shape ideas on a shared canvas. Start a room or join your team with a room code.</p>
              <div className="mt-7 flex flex-wrap gap-2 text-[11px] text-[#76857a]">
                {['Live cursors', 'Shared drawing tools', 'Room-based access'].map((feature) => <span key={feature} className="rounded-full border border-[#e0e7df] bg-white px-3 py-1.5">{feature}</span>)}
              </div>
            </section>

            <section className="mx-auto w-full max-w-2xl">
              <div className="mb-5">
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#78907a]">Get started</p>
                <h2 className="mt-1.5 text-xl font-semibold tracking-tight text-[#2a4031]">Choose how to enter</h2>
              </div>
              {formError && <div role="alert" className="mb-4 flex items-start gap-2 rounded-xl border border-[#efd2c9] bg-[#fff6f3] px-3.5 py-3 text-xs leading-5 text-[#984d3d]"><X className="mt-0.5 h-4 w-4 shrink-0" />{formError}</div>}
              <div className="grid gap-4 sm:grid-cols-2">
                <form onSubmit={handleCreateRoom} className="flex flex-col rounded-2xl border border-[#dfe7df] bg-white p-5 shadow-[0_16px_45px_-36px_rgba(30,66,42,.35)] sm:p-6">
                  <div className="mb-5 flex items-start gap-3"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#edf4ec] text-[#4f7955]"><Plus className="h-5 w-5" /></span><div><h3 className="text-sm font-semibold text-[#334a38]">Create a new room</h3><p className="mt-1 text-xs leading-5 text-[#89958c]">Name your space and start a shared canvas.</p></div></div>
                  <label htmlFor="create-room-title" className="mb-1.5 text-xs font-semibold text-[#53665a]">Room name</label>
                  <input id="create-room-title" value={createRoomName} onChange={(event) => setCreateRoomName(event.target.value)} maxLength={60} required placeholder="e.g. Product brainstorm" className="mb-4 h-11 w-full rounded-xl border border-[#dfe7df] bg-[#fcfdfb] px-3.5 text-sm text-[#33483a] outline-none placeholder:text-[#a0aaa2] focus:border-[#a5c0a7] focus:ring-4 focus:ring-[#eaf2e9]" disabled={isConnecting} />
                  <label htmlFor="create-room-name" className="mb-1.5 text-xs font-semibold text-[#53665a]">Your name</label>
                  <input id="create-room-name" value={createUsername} onChange={(event) => setCreateUsername(event.target.value)} autoComplete="name" maxLength={40} required placeholder="e.g. Alex Morgan" className="h-11 w-full rounded-xl border border-[#dfe7df] bg-[#fcfdfb] px-3.5 text-sm text-[#33483a] outline-none placeholder:text-[#a0aaa2] focus:border-[#a5c0a7] focus:ring-4 focus:ring-[#eaf2e9]" disabled={isConnecting} />
                  <p className="mt-2 text-[11px] leading-5 text-[#929d94]">Your name appears to collaborators in the room.</p>
                  <button type="submit" disabled={isConnecting} className="mt-5 inline-flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#254f36] px-4 text-sm font-semibold text-white transition hover:bg-[#1d412c] disabled:cursor-wait disabled:opacity-60">{isConnecting ? 'Connecting…' : 'Create room'}{!isConnecting && <MoveRight className="h-4 w-4" />}</button>
                </form>

                <form onSubmit={handleJoinRoom} className="flex flex-col rounded-2xl border border-[#dfe7df] bg-white p-5 shadow-[0_16px_45px_-36px_rgba(30,66,42,.35)] sm:p-6">
                  <div className="mb-5 flex items-start gap-3"><span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[#eef2ed] text-[#5b7460]"><MoveRight className="h-5 w-5" /></span><div><h3 className="text-sm font-semibold text-[#334a38]">Join an existing room</h3><p className="mt-1 text-xs leading-5 text-[#89958c]">Enter your name and the room code.</p></div></div>
                  <label htmlFor="join-room-name" className="mb-1.5 text-xs font-semibold text-[#53665a]">Your name</label>
                  <input id="join-room-name" value={joinUsername} onChange={(event) => setJoinUsername(event.target.value)} autoComplete="name" maxLength={40} required placeholder="e.g. Alex Morgan" className="mb-4 h-11 w-full rounded-xl border border-[#dfe7df] bg-[#fcfdfb] px-3.5 text-sm text-[#33483a] outline-none placeholder:text-[#a0aaa2] focus:border-[#a5c0a7] focus:ring-4 focus:ring-[#eaf2e9]" disabled={isConnecting} />
                  <label htmlFor="join-room-code" className="mb-1.5 text-xs font-semibold text-[#53665a]">Room code</label>
                  <input id="join-room-code" value={roomId} onChange={(event) => setRoomId(event.target.value.toUpperCase().replace(/\s/g, ''))} autoComplete="off" maxLength={12} required placeholder="e.g. A1B2C3" className="h-11 w-full rounded-xl border border-[#dfe7df] bg-[#fcfdfb] px-3.5 font-mono text-sm uppercase tracking-[0.12em] text-[#33483a] outline-none placeholder:font-sans placeholder:normal-case placeholder:tracking-normal placeholder:text-[#a0aaa2] focus:border-[#a5c0a7] focus:ring-4 focus:ring-[#eaf2e9]" disabled={isConnecting} />
                  <button type="submit" disabled={isConnecting} className="mt-5 inline-flex h-11 w-full items-center justify-center gap-2 rounded-xl border border-[#d8e4d7] bg-[#f5f8f4] px-4 text-sm font-semibold text-[#3f6647] transition hover:bg-[#edf4ec] disabled:cursor-wait disabled:opacity-60">{isConnecting ? 'Connecting…' : 'Join room'}{!isConnecting && <MoveRight className="h-4 w-4" />}</button>
                </form>
              </div>
              <p className="mt-4 text-center text-[10px] leading-5 text-[#9aa49b]">Rooms are temporary and stay available while someone is connected.</p>
            </section>
          </div>

          <footer className="hidden min-h-10 items-center justify-between border-t border-[#e1e6de] text-[10px] text-[#9aa49b] sm:flex"><span>Shared canvas · Real-time updates</span><span>Draw with care</span></footer>
        </div>
      </main>
    );
  }

  return (
    <div className="flex h-[100dvh] min-h-0 flex-col bg-[#f1f4ef] text-[#2d4033]">
      <div className="z-20 flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-[#e1e7df] bg-white px-3 py-2.5 shadow-[0_4px_14px_-12px_rgba(30,66,42,.3)] sm:px-5">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2 sm:gap-4">
          <Link href={BASE_URL} className="flex items-center gap-2">
            <div className="rounded-lg p-2 transition-colors hover:bg-[#f1f5f0]">
              <Home className="h-4 w-4 text-[#718176]" />
            </div>
          </Link>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 animate-pulse rounded-full bg-[#65a06f]"></div>
            <span className="max-w-[32vw] truncate text-xs font-semibold text-[#38513c] sm:max-w-[220px] sm:text-sm">{roomName || 'Shared room'}</span>
            <span className="rounded-md bg-[#f1f5f0] px-1.5 py-1 font-mono text-[10px] tracking-[0.08em] text-[#718176] sm:text-[11px]">{roomId}</span>
            <button
              onClick={copyRoomId}
              className="rounded p-1 text-[#839087] transition-colors hover:bg-[#f1f5f0]"
              title="Copy room ID"
            >
              {copied ? <Check size={15} className="text-[#52835b]" /> : <Copy size={15} />}
            </button>
          </div>
          
          <div className="flex items-center gap-1.5 text-[11px] text-[#78867c] sm:text-xs">
            <Users size={14} />
            <span>{users.length} here</span>
          </div>

          <div className="flex -space-x-2">
            {users.filter(u => u.userId !== userId).map((user) => (
              <div
                key={user.userId}
                className="flex h-7 w-7 items-center justify-center rounded-full border-2 border-white text-xs font-bold text-white"
                style={{ backgroundColor: user.color }}
                title={user.username}
              >
                {user.username.charAt(0).toUpperCase()}
              </div>
            ))}
            <div
              className="flex h-7 w-7 items-center justify-center rounded-full border-2 border-white text-xs font-bold text-white"
              style={{ backgroundColor: userColor }}
              title="You"
            >
              {username.charAt(0).toUpperCase()}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-0.5 text-[#7b897e] sm:gap-1">
          <button
            onClick={undo}
            disabled={historyIndex <= 0}
            className="rounded-lg p-2 transition-colors hover:bg-[#f1f5f0] disabled:opacity-40"
            title="Undo"
          >
            <Undo size={20} />
          </button>
          <button
            onClick={redo}
            disabled={historyIndex >= history.length - 1}
            className="rounded-lg p-2 transition-colors hover:bg-[#f1f5f0] disabled:opacity-40"
            title="Redo"
          >
            <Redo size={20} />
          </button>
          <button
            onClick={clearCanvas}
            className="rounded-lg p-2 text-[#a64f40] transition-colors hover:bg-[#fff2ef]"
            title="Clear Canvas"
          >
            <Trash2 size={20} />
          </button>
          <button
            onClick={saveState}
            className="rounded-lg p-2 transition-colors hover:bg-[#f1f5f0]"
            title="Save Board State"
          >
            <Save size={20} />
          </button>
          <button
            onClick={exportAsImage}
            className="rounded-lg p-2 transition-colors hover:bg-[#f1f5f0]"
            title="Export as Image"
          >
            <Download size={20} />
          </button>
          <button
            onClick={leaveRoom}
            className="rounded-lg bg-[#a64f40] px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#914433] sm:px-4 sm:text-sm"
          >
            Leave
          </button>
        </div>
      </div>

      <div className="z-10 flex shrink-0 items-center gap-3 overflow-x-auto border-b border-[#e1e7df] bg-white px-3 py-2 text-[#728075] sm:gap-4 sm:px-5">
        <div className="flex shrink-0 items-center gap-0.5 rounded-xl bg-[#f1f4ef] p-1">
          <button
            onClick={() => setSelectedTool('eraser')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'eraser' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Eraser"
          >
            <Eraser size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('pan')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'pan' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Pan/Move (or hold Space)"
          >
            <Hand size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('pencil')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'pencil' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Pencil"
          >
            <Pencil size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('rectangle')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'rectangle' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Rectangle"
          >
            <Square size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('circle')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'circle' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Circle"
          >
            <Circle size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('text')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'text' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Text"
          >
            <Type size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('arrow')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'arrow' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Arrow"
          >
            <MoveRight size={20} />
          </button>
          <button
            onClick={() => setSelectedTool('triangle')}
            className={`rounded-lg p-2 transition-colors ${selectedTool === 'triangle' ? 'bg-white text-[#365d3d] shadow-sm' : 'hover:bg-white/70'}`}
            title="Triangle"
          >
            <Triangle size={20} />
          </button>
        </div>

        <div className="h-8 w-px shrink-0 bg-[#e4eae3]"></div>

        <div className="flex shrink-0 items-center gap-1">
          {COLORS.map((color) => (
            <button
              key={color}
              onClick={() => setSelectedColor(color)}
              aria-label={`Select drawing color ${color}`}
              className={`h-6 w-6 rounded-full border-2 transition-all ${selectedColor === color ? 'scale-110 border-[#304a35] ring-2 ring-[#dce8dc]' : 'border-transparent hover:scale-105'}`}
              style={{ backgroundColor: color }}
            />
          ))}
        </div>

        <div className="h-8 w-px shrink-0 bg-[#e4eae3]"></div>

        <div className="flex shrink-0 items-center gap-2 text-[#78867c]">
          <button
            onClick={() => setStrokeWidth(Math.max(2, strokeWidth - 2))}
            aria-label="Decrease brush size"
            className="rounded p-1 hover:bg-[#f1f5f0]"
          >
            <Minus size={16} />
          </button>
          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-[#e4eae3] sm:w-20">
            <div
              className="h-full bg-[#4d7853] transition-all"
              style={{ width: `${(strokeWidth / 12) * 100}%` }}
            />
          </div>
          <button
            onClick={() => setStrokeWidth(Math.min(12, strokeWidth + 2))}
            aria-label="Increase brush size"
            className="rounded p-1 hover:bg-[#f1f5f0]"
          >
            <Plus size={16} />
          </button>
        </div>

        <div className="h-8 w-px shrink-0 bg-[#e4eae3]"></div>

        {/* Zoom controls */}
        <div className="flex shrink-0 items-center gap-1 text-[#78867c]">
          <button onClick={handleZoomOut} className="rounded-lg p-2 hover:bg-[#f1f5f0]" title="Zoom out" aria-label="Zoom out"><ZoomOut size={18} /></button>
          <span className="min-w-[44px] text-center text-xs text-[#637368]">{Math.round(zoom * 100)}%</span>
          <button onClick={handleZoomIn} className="rounded-lg p-2 hover:bg-[#f1f5f0]" title="Zoom in" aria-label="Zoom in"><ZoomIn size={18} /></button>
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

      <div className="flex shrink-0 flex-wrap items-center justify-between gap-1 border-t border-[#e1e7df] bg-white px-3 py-2 text-[10px] text-[#738178] sm:px-5 sm:text-xs">
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
