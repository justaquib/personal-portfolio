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
  const initializedRef = useRef(false);
  
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
          content: "Hello! I'm Aquib's website assistant. I can answer questions about this website, Aquib Shahbaz, and all the projects. What would you like to know?",
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
    if (sessions.length === 0 && !initializedRef.current) {
      initializedRef.current = true;
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
      const response = await fetch("/api/local-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
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
      greeting: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      help: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      weather: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      time: "border-[#e7dfc8] bg-[#f6f1e3] text-[#806b35]",
      search: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      math: "border-[#e7dfc8] bg-[#f6f1e3] text-[#806b35]",
      code: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      explain: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      summary: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
      translate: "border-[#d9e5cd] bg-[#edf2e5] text-[#557736]",
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
      color: intentColors[intentName] || "border-[#e5e3dc] bg-white text-[#777a72]",
      icon: intentIcons[intentName] || "💬",
      label: intentName.charAt(0).toUpperCase() + intentName.slice(1),
    };
  };

  return (
    <main className="relative flex h-dvh flex-col overflow-hidden bg-[#f5f4ef] text-[#252821]">
      {/* Header */}
      <div className="z-30 flex shrink-0 items-center justify-between gap-2 border-b border-[#e5e3dc] bg-[#fbfaf7] px-3 py-2.5 sm:px-5">
        <div className="flex items-center gap-2">
          <BackButton />
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="rounded-full border border-[#e1dfd6] bg-white p-2 text-[#62665c] transition-colors hover:bg-[#eeede7]"
            title={showSidebar ? "Hide sidebar" : "Show sidebar"}
          >
            <MessageSquare className="w-5 h-5" />
          </button>
        </div>
        <h1 className="flex items-center gap-2 text-base font-semibold tracking-tight text-[#292d25] sm:text-lg">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-[#e6eddb] text-[#688746]"><Bot className="h-4 w-4" /></div>
          <span className="hidden sm:inline">AI Chatbot</span><span className="sm:hidden">AI Chat</span>
        </h1>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNLPAnalysis(!showNLPAnalysis)}
            className={`rounded-full border p-2 transition-colors ${
              showNLPAnalysis ? "border-[#cad6b9] bg-[#edf2e5] text-[#557736]" : "border-[#e1dfd6] bg-white text-[#62665c] hover:bg-[#eeede7]"
            }`}
            title="Toggle NLP Analysis"
          >
            <Brain className="w-5 h-5" />
          </button>
          <button
            onClick={() => setShowDetails(true)}
            className="rounded-full border border-[#e1dfd6] bg-white p-2 text-[#62665c] transition-colors hover:bg-[#eeede7]"
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

      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        {/* Sidebar - Chat Sessions */}
        <div 
          className={`absolute inset-y-0 left-0 z-20 flex flex-col overflow-hidden border-r border-[#e5e3dc] bg-[#fbfaf7] shadow-lg transition-all duration-300 md:relative md:z-0 md:shadow-none ${
            showSidebar ? "w-72 translate-x-0" : "w-0 -translate-x-full md:translate-x-0"
          }`}
        >
          <div className="border-b border-[#e5e3dc] p-4">
            <button
              onClick={createNewSession}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#252821] px-4 py-2.5 font-medium text-white transition-colors hover:bg-[#3a4033]"
            >
              <Sparkles className="w-4 h-4" />
              New Chat
            </button>
          </div>
          
          <div className="flex-1 space-y-2 overflow-y-auto p-2">
            {sessions.map((session) => (
              <div
                key={session.id}
                className={`group flex items-center gap-2 p-3 rounded-xl cursor-pointer transition-all ${
                  currentSession?.id === session.id
                    ? "border border-[#dfe5d5] bg-[#edf2e5]"
                    : "border border-transparent hover:bg-[#f0efe9]"
                }`}
                onClick={() => setCurrentSession(session)}
              >
                <History className="h-4 w-4 shrink-0 text-[#85877f]" />
                <span className="flex-1 truncate text-sm text-[#454940]">{session.name}</span>
                <span className="text-xs text-[#85877f]">{formatTime(session.createdAt)}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.id);
                  }}
                  className="rounded-lg p-1 opacity-0 transition-all hover:bg-red-50 group-hover:opacity-100"
                >
                  <Trash2 className="h-3.5 w-3.5 text-red-600" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex min-w-0 flex-1 flex-col bg-[#f8f7f3]">
          {/* Chat Header */}
          <div className="flex items-center justify-between border-b border-[#e5e3dc] bg-[#fbfaf7] px-3 py-2.5 sm:px-5">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 animate-pulse rounded-full bg-[#86a55e]" />
              <span className="text-sm text-[#65695f]">AI Assistant</span>
              {lastIntent && (
                <span className={`px-2 py-0.5 rounded-full text-xs border ${getIntentInfo(lastIntent.name).color}`}>
                  {getIntentInfo(lastIntent.name).icon} {getIntentInfo(lastIntent.name).label}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={clearCurrentChat}
                className="rounded-lg p-2 text-[#71756c] transition-colors hover:bg-[#eeede7]"
                title="Clear chat"
              >
                <Trash2 className="w-4 h-4" />
              </button>
              <button
                onClick={createNewSession}
                className="rounded-lg p-2 text-[#71756c] transition-colors hover:bg-[#eeede7]"
                title="New chat"
              >
                <Zap className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Messages Area */}
          <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-3 sm:p-6">
            {currentSession?.messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`flex items-start gap-3 max-w-[75%] ${message.role === "user" ? "flex-row-reverse" : ""}`}>
                  {/* Avatar */}
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.role === "user" 
                    ? "bg-[#e7edf7]"
                      : "bg-[#e6eddb]"
                  }`}>
                    {message.role === "user" ? (
                      <User className="h-4 w-4 text-[#6581a8]" />
                    ) : (
                      <Bot className="h-4 w-4 text-[#688746]" />
                    )}
                  </div>
                  
                  {/* Message Bubble */}
                  <div className={`relative group ${
                    message.role === "user"
                      ? "rounded-2xl rounded-br-md bg-[#343a30] text-white"
                      : "rounded-2xl rounded-bl-md border border-[#e5e3dc] bg-white text-[#30342c] shadow-sm"
                  }`}>
                    <div className="p-4 text-sm leading-relaxed whitespace-pre-wrap">
                      {message.content}
                    </div>
                    <div className={`absolute bottom-2 right-2 flex items-center gap-1 ${
                      message.role === "user" ? "" : "opacity-0 group-hover:opacity-100"
                    } transition-opacity`}>
                      <button
                        onClick={() => copyToClipboard(message.content, `msg-${index}`)}
                        className="rounded-lg p-1.5 transition-colors hover:bg-black/5"
                        title="Copy"
                      >
                        {copied === `msg-${index}` ? (
                          <Check className="h-3.5 w-3.5 text-[#a8c58a]" />
                        ) : (
                          <Copy className="h-3.5 w-3.5 text-[#85877f]" />
                        )}
                      </button>
                      <span className="px-1 text-xs text-[#85877f]">
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
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#e6eddb]">
                    <Bot className="h-4 w-4 text-[#688746]" />
                  </div>
                  <div className="rounded-2xl rounded-bl-md border border-[#e5e3dc] bg-white px-4 py-3 shadow-sm">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-[#688746]" />
                      <span className="text-sm text-[#777a72]">Thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* NLP Analysis Panel */}
          {showNLPAnalysis && lastIntent && (
            <div className="mx-3 mb-3 rounded-xl border border-[#dedccf] bg-[#eeefe7] p-3 sm:mx-5">
              <div className="flex items-center gap-2 mb-2">
                <Brain className="h-4 w-4 text-[#688746]" />
                <span className="text-sm font-medium text-[#526b39]">NLP Analysis</span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-[#85877f]">Intent:</span>
                  <span className="ml-2 text-[#30342c]">{getIntentInfo(lastIntent.name).label}</span>
                </div>
                <div>
                  <span className="text-[#85877f]">Confidence:</span>
                  <span className="ml-2 text-[#30342c]">{(lastIntent.confidence * 100).toFixed(1)}%</span>
                </div>
                {Object.keys(lastIntent.entities).length > 0 && (
                  <div className="col-span-2">
                    <span className="text-[#85877f]">Entities:</span>
                    <div className="mt-1 flex flex-wrap gap-2">
                      {Object.entries(lastIntent.entities).map(([key, value]) => (
                        <span key={key} className="rounded bg-white px-2 py-1 text-[#53564f]">
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
            <div className="mx-3 mb-3 flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 p-3 sm:mx-5">
              <span className="text-sm text-red-700">{error}</span>
              <button
                onClick={() => setError(null)}
                className="ml-auto p-1 hover:bg-red-500/20 rounded"
              >
                <X className="h-4 w-4 text-red-600" />
              </button>
            </div>
          )}

          {/* Input Area */}
          <div className="border-t border-[#e5e3dc] bg-[#fbfaf7] p-3 sm:p-4">
            <div className="mx-auto flex max-w-4xl items-end gap-2 sm:gap-3">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message..."
                  className="w-full resize-none rounded-2xl border border-[#deddd5] bg-white px-4 py-3 text-[#30342c] placeholder-[#9a9c95] focus:border-[#91a676] focus:outline-none focus:ring-2 focus:ring-[#dfe8d2] sm:px-5"
                  rows={1}
                  style={{ minHeight: '52px', maxHeight: '120px' }}
                  disabled={isLoading}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading}
                className="flex items-center justify-center rounded-xl bg-[#252821] p-3.5 text-white transition-colors hover:bg-[#3a4033] disabled:cursor-not-allowed disabled:bg-[#c9c9c2]"
              >
                {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
              <Send className="h-5 w-5" />
                )}
              </button>
            </div>
            <p className="mt-2 text-center text-[10px] text-[#85877f] sm:text-xs">
              AI can make mistakes. Consider checking important information.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
