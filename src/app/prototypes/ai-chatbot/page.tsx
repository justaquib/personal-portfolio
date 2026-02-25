"use client";

import BackButton from "@/components/BackButton";
import DetailsModal from "@/components/DetailsModal";
import prototypeLists from "@/utils/json/prototypeList.json";
import React, { useState, useRef, useEffect, useCallback } from "react";
import { 
  Send, 
  Loader2, 
  Bot, 
  User, 
  Trash2, 
  Settings, 
  Sparkles, 
  MessageSquare, 
  Brain,
  Zap,
  History,
  X,
  Copy,
  Check,
  RefreshCw
} from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  createdAt: Date;
}

interface NLPIntent {
  name: string;
  confidence: number;
  entities: Record<string, string>;
}

// Predefined intents for NLP processing
const INTENT_PATTERNS = {
  greeting: /^(hi|hello|hey|howdy|good morning|good evening|good afternoon|greetings)/i,
  help: /^(help|what can you do|commands|features|assist|support)/i,
  weather: /(?:weather|temperature|forecast|rain|sunny|cloudy)/i,
  time: /(?:time|what time|clock|date|today)/i,
  search: /(?:search|find|look up|google)/i,
  math: /(?:calculate|compute|math|sum|plus|minus|multiply|divide|equation)/i,
  code: /(?:code|program|function|script|debug|error)/i,
  explain: /(?:explain|what is|how does|define|meaning)/i,
  summary: /(?:summarize|summary|recap|brief)/i,
  translate: /(?:translate|language|spanish|french|german|chinese)/i,
};

export default function AIChatbot() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [copied, setCopied] = useState<string | null>(null);
  const [showNLPAnalysis, setShowNLPAnalysis] = useState(false);
  const [lastIntent, setLastIntent] = useState<NLPIntent | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'ai-chatbot');
  const [showDetails, setShowDetails] = useState(false);

  // Analyze user intent using NLP
  const analyzeIntent = useCallback((text: string): NLPIntent => {
    let bestIntent: NLPIntent = { name: "general", confidence: 0, entities: {} };
    
    for (const [intentName, pattern] of Object.entries(INTENT_PATTERNS)) {
      const match = text.match(pattern);
      if (match) {
        const confidence = match[0].length / text.length;
        if (confidence > bestIntent.confidence) {
          bestIntent = { 
            name: intentName, 
            confidence, 
            entities: extractEntities(text, intentName) 
          };
        }
      }
    }
    
    return bestIntent;
  }, []);

  // Extract entities from text
  const extractEntities = (text: string, intent: string): Record<string, string> => {
    const entities: Record<string, string> = {};
    
    // Extract numbers
    const numbers = text.match(/\d+/g);
    if (numbers) entities.numbers = numbers.join(", ");
    
    // Extract quoted text
    const quoted = text.match(/"([^"]+)"/g);
    if (quoted) entities.quoted = quoted.join(", ");
    
    // Extract question words
    if (/\b(who|what|where|when|why|how|which)\b/i.test(text)) {
      entities.questionType = text.match(/\b(who|what|where|when|why|how|which)\b/i)?.[0] || "";
    }
    
    return entities;
  };

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  // Create a new chat session
  const createNewSession = useCallback(() => {
    const newSession: ChatSession = {
      id: Date.now().toString(),
      name: `Chat ${sessions.length + 1}`,
      messages: [
        {
          role: "assistant",
          content: "Hello! I'm your AI-powered chat assistant with NLP capabilities. I can understand natural language, recognize your intent, and provide contextual responses. How can I help you today?",
          timestamp: new Date(),
        },
      ],
      createdAt: new Date(),
    };
    
    setSessions(prev => [newSession, ...prev]);
    setCurrentSession(newSession);
    setError(null);
    setLastIntent(null);
  }, [sessions.length]);

  // Initialize with a new session on first load
  useEffect(() => {
    if (sessions.length === 0) {
      createNewSession();
    }
  }, []);

  // Send message to the chat
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading || !currentSession) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    // Analyze intent
    const intent = analyzeIntent(userMessage.content);
    setLastIntent(intent);

    // Update current session with user message
    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
    };
    setCurrentSession(updatedSession);
    
    // Update sessions list
    setSessions(prev => prev.map(s => s.id === currentSession.id ? updatedSession : s));
    
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "chat",
          content: "",
          question: userMessage.content,
          history: updatedSession.messages.map(m => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to get response");
      }

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        timestamp: new Date(),
      };

      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, assistantMessage],
      };
      
      setCurrentSession(finalSession);
      setSessions(prev => prev.map(s => s.id === currentSession.id ? finalSession : s));
    } catch (err) {
      console.error("Chat error:", err);
      setError("Failed to get response. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle keyboard input
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Delete a chat session
  const deleteSession = (sessionId: string) => {
    const newSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(newSessions);
    
    if (currentSession?.id === sessionId) {
      if (newSessions.length > 0) {
        setCurrentSession(newSessions[0]);
      } else {
        createNewSession();
      }
    }
  };

  // Copy message to clipboard
  const copyToClipboard = async (content: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(messageId);
      setTimeout(() => setCopied(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Clear current chat
  const clearCurrentChat = () => {
    if (!currentSession) return;
    
    const clearedSession: ChatSession = {
      ...currentSession,
      messages: [
        {
          role: "assistant",
          content: "Chat cleared. How can I help you now?",
          timestamp: new Date(),
        },
      ],
    };
    
    setCurrentSession(clearedSession);
    setSessions(prev => prev.map(s => s.id === currentSession.id ? clearedSession : s));
    setLastIntent(null);
  };

  // Format timestamp
  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Get intent display info
  const getIntentInfo = (intentName: string) => {
    const intentColors: Record<string, string> = {
      greeting: "bg-green-500/20 text-green-400 border-green-500/30",
      help: "bg-blue-500/20 text-blue-400 border-blue-500/30",
      weather: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
      time: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
      search: "bg-purple-500/20 text-purple-400 border-purple-500/30",
      math: "bg-orange-500/20 text-orange-400 border-orange-500/30",
      code: "bg-pink-500/20 text-pink-400 border-pink-500/30",
      explain: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
      summary: "bg-teal-500/20 text-teal-400 border-teal-500/30",
      translate: "bg-rose-500/20 text-rose-400 border-rose-500/30",
    };
    
    const intentIcons: Record<string, string> = {
      greeting: "👋",
      help: "🆘",
      weather: "🌤️",
      time: "⏰",
      search: "🔍",
      math: "🔢",
      code: "💻",
      explain: "📚",
      summary: "📝",
      translate: "🌍",
    };

    return {
      color: intentColors[intentName] || "bg-gray-500/20 text-gray-400 border-gray-500/30",
      icon: intentIcons[intentName] || "💬",
      label: intentName.charAt(0).toUpperCase() + intentName.slice(1),
    };
  };

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <BackButton />
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
            title={showSidebar ? "Hide sidebar" : "Show sidebar"}
          >
            <MessageSquare className="w-5 h-5" />
          </button>
        </div>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent flex items-center gap-2">
          <Bot className="w-6 h-6 text-emerald-400" />
          AI Chatbot
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNLPAnalysis(!showNLPAnalysis)}
            className={`p-2 rounded-full transition-colors ${
              showNLPAnalysis ? "bg-purple-500/20 text-purple-400" : "bg-gray-800 hover:bg-gray-700"
            }`}
            title="Toggle NLP Analysis"
          >
            <Brain className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowDetails(true)}
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors"
            title="View Details"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
      </div>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'AI Chatbot'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      <div className="pt-20 h-screen flex">
        {/* Sidebar - Chat Sessions */}
        <div 
          className={`${
            showSidebar ? "w-72" : "w-0"
          } transition-all duration-300 overflow-hidden flex flex-col border-r border-gray-700/50`}
        >
          <div className="p-4 border-b border-gray-700/50">
            <button
              onClick={createNewSession}
              className="w-full py-2.5 px-4 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 rounded-xl font-medium transition-all flex items-center justify-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              New Chat
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`group flex items-center gap-2 p-3 rounded-xl cursor-pointer transition-all ${
                  currentSession?.id === session.id
                    ? "bg-gray-700/50 border border-gray-600/50"
                    : "hover:bg-gray-800/50 border border-transparent"
                }`}
                onClick={() => setCurrentSession(session)}
              >
                <History className="w-4 h-4 text-gray-400 flex-shrink-0" />
                <span className="flex-1 text-sm truncate">{session.name}</span>
                <span className="text-xs text-gray-500">{formatTime(session.createdAt)}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded-lg transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5 text-red-400" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Chat Header */}
          <div className="px-6 py-3 border-b border-gray-700/50 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-sm text-gray-400">AI Assistant</span>
              {lastIntent && (
                <span className={`px-2 py-0.5 rounded-full text-xs border ${getIntentInfo(lastIntent.name).color}`}>
                  {getIntentInfo(lastIntent.name).icon} {getIntentInfo(lastIntent.name).label}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={clearCurrentChat}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                title="Clear chat"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <button
                onClick={createNewSession}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                title="New chat"
              >
                <Zap className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {currentSession?.messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`flex items-start gap-3 max-w-[75%] ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.role === "user" 
                      ? "bg-blue-500/20" 
                      : "bg-gradient-to-br from-emerald-500/20 to-cyan-500/20"
                  }`}>
                    {message.role === "user" ? (
                      <User className="w-4 h-4 text-blue-400" />
                    ) : (
                      <Bot className="w-4 h-4 text-emerald-400" />
                    )}
                  </div>
                  
                  {/* Message Bubble */}
                  <div className={`relative group ${
                    message.role === "user"
                      ? "bg-blue-500 text-white rounded-2xl rounded-br-md"
                      : "bg-gray-800/50 border border-gray-700/50 text-gray-100 rounded-2xl rounded-bl-md"
                  }`}>
                    <div className="p-4 text-sm leading-relaxed whitespace-pre-wrap">
                      {message.content}
                    </div>
                    <div className={`absolute bottom-2 right-2 flex items-center gap-1 ${
                      message.role === "user" ? "" : "opacity-0 group-hover:opacity-100"
                    } transition-opacity`}>
                      <button
                        onClick={() => copyToClipboard(message.content, `msg-${index}`)}
                        className="p-1.5 hover:bg-gray-700/50 rounded-lg transition-colors"
                        title="Copy"
                      >
                        {copied === `msg-${index}` ? (
                          <Check className="w-3.5 h-3.5 text-green-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5 text-gray-400" />
                        )}
                      </button>
                      <span className="text-xs text-gray-500 px-1">
                        {formatTime(message.timestamp)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center">
                    <Bot className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div className="bg-gray-800/50 border border-gray-700/50 rounded-2xl rounded-bl-md px-4 py-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
                      <span className="text-sm text-gray-400">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* NLP Analysis Panel */}
          {showNLPAnalysis && lastIntent && (
            <div className="mx-6 mb-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-xl">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-purple-400">NLP Analysis</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-500">Intent:</span>
                  <span className="ml-2 text-white">{getIntentInfo(lastIntent.name).label}</span>
                </div>
                <div>
                  <span className="text-gray-500">Confidence:</span>
                  <span className="ml-2 text-white">{(lastIntent.confidence * 100).toFixed(1)}%</span>
                </div>
                {Object.keys(lastIntent.entities).length > 0 && (
                  <div className="col-span-2">
                    <span className="text-gray-500">Entities:</span>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {Object.entries(lastIntent.entities).map(([key, value]) => (
                        <span key={key} className="px-2 py-1 bg-gray-700/50 rounded text-gray-300">
                          {key}: {value}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mx-6 mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-2">
              <span className="text-red-400 text-sm">{error}</span>
              <button
                onClick={() => setError(null)}
                className="ml-auto p-1 hover:bg-red-500/20 rounded"
              >
                <X className="w-4 h-4 text-red-400" />
              </button>
            </div>
          )}

          {/* Input Area */}
          <div className="p-4 border-t border-gray-700/50">
            <div className="flex items-end gap-3 max-w-4xl mx-auto">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message..."
                  className="w-full bg-gray-800/50 border border-gray-700/50 text-white rounded-2xl px-5 py-3.5 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 placeholder-gray-500 resize-none"
                  rows={1}
                  style={{ minHeight: '52px', maxHeight: '120px' }}
                  disabled={isLoading}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                className="p-3.5 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed rounded-xl transition-all flex items-center justify-center"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            <p className="text-center text-xs text-gray-500 mt-3">
              AI can make mistakes. Consider checking important information.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
