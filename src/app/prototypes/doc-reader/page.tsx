"use client";

import BackButton from "@/components/BackButton";
import DetailsModal from "@/components/DetailsModal";
import prototypeLists from "@/utils/json/prototypeList.json";
import React, { useState, useRef, useCallback, useEffect } from "react";
import { Upload, FileText, Send, Loader2, X, Sparkles, MessageCircle, FileQuestion, AlertCircle, Copy, Check, Eye, ChevronDown, FileSpreadsheet } from "lucide-react";
import * as XLSX from "xlsx";
import mammoth from "mammoth";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface DocumentInfo {
  name: string;
  url: string;
  type: string;
  content: string;
  fileType: "pdf" | "txt" | "docx" | "xlsx" | "csv" | "unknown";
}

// Dynamic import for PDF.js to avoid SSR issues
let pdfjsLib: typeof import("pdfjs-dist") | null = null;

const getPdfLib = async () => {
  if (!pdfjsLib) {
    pdfjsLib = await import("pdfjs-dist");
    pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
  }
  return pdfjsLib;
};

// Get file extension
const getFileExtension = (filename: string): string => {
  return filename.split(".").pop()?.toLowerCase() || "";
};

// Check if file is supported
const isFileSupported = (file: File): boolean => {
  const ext = getFileExtension(file.name);
  const supportedExtensions = ["pdf", "txt", "docx", "xlsx", "xls", "doc", "csv"];
  return supportedExtensions.includes(ext);
};

export default function DocReader() {
  const [document, setDocument] = useState<DocumentInfo | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [visibleLines, setVisibleLines] = useState(50);
  const [showAllContent, setShowAllContent] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const contentEndRef = useRef<HTMLDivElement>(null);

  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'doc-reader');
  const [showDetails, setShowDetails] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const extractTextFromPDF = async (file: File): Promise<string> => {
    const lib = await getPdfLib();
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await lib.getDocument({ data: arrayBuffer }).promise;
    let fullText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: unknown) => (item as { str: string }).str)
        .join(" ");
      fullText += `--- Page ${i} ---\n${pageText}\n\n`;
    }

    return fullText;
  };

  const extractTextFromDOCX = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
  };

  const extractTextFromExcel = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: "array" });
    let fullText = "";

    workbook.SheetNames.forEach((sheetName) => {
      const sheet = workbook.Sheets[sheetName];
      const csv = XLSX.utils.sheet_to_csv(sheet);
      fullText += `--- Sheet: ${sheetName} ---\n${csv}\n\n`;
    });

    return fullText;
  };

  const extractTextFromFile = async (file: File): Promise<{ content: string; fileType: DocumentInfo["fileType"] }> => {
    const ext = getFileExtension(file.name);

    switch (ext) {
      case "pdf":
        return { content: await extractTextFromPDF(file), fileType: "pdf" };
      case "docx":
      case "doc":
        return { content: await extractTextFromDOCX(file), fileType: "docx" };
      case "xlsx":
      case "xls":
        return { content: await extractTextFromExcel(file), fileType: "xlsx" };
      case "csv":
        return { content: await file.text(), fileType: "csv" };
      case "txt":
        return { content: await file.text(), fileType: "txt" };
      default:
        if (file.type.startsWith("text/")) {
          return { content: await file.text(), fileType: "txt" };
        }
        return { content: "", fileType: "unknown" };
    }
  };

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!isFileSupported(file)) {
        setError("Unsupported file type. Please upload PDF, TXT, DOCX, or XLSX files.");
        return;
      }

      setError(null);
      setSummary(null);
      setMessages([]);
      setIsLoading(true);
      setVisibleLines(50);
      setShowAllContent(false);

      try {
        const url = URL.createObjectURL(file);
        const { content, fileType } = await extractTextFromFile(file);

        setDocument({
          name: file.name,
          url: url,
          type: file.type,
          content: content,
          fileType: fileType,
        });

        // Auto-summarize on upload
        if (content) {
          await generateSummary(content);
        }
      } catch (err) {
        console.error("Error processing file:", err);
        setError("Failed to process the document. Please try again.");
      } finally {
        setIsLoading(false);
      }
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      if (!isFileSupported(file)) {
        setError("Unsupported file type. Please upload PDF, TXT, DOCX, or XLSX files.");
        return;
      }

      setError(null);
      setSummary(null);
      setMessages([]);
      setIsLoading(true);
      setVisibleLines(50);
      setShowAllContent(false);

      try {
        const url = URL.createObjectURL(file);
        const { content, fileType } = await extractTextFromFile(file);

        setDocument({
          name: file.name,
          url: url,
          type: file.type,
          content: content,
          fileType: fileType,
        });

        // Auto-summarize on upload
        if (content) {
          await generateSummary(content);
        }
      } catch (err) {
        console.error("Error processing file:", err);
        setError("Failed to process the document. Please try again.");
      } finally {
        setIsLoading(false);
      }
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const generateSummary = async (content: string) => {
    setIsSummarizing(true);
    setError(null);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "summarize",
          content: content.substring(0, 30000), // Limit content length
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate summary");
      }

      setSummary(data.summary);
    } catch (err) {
      console.error("Summary error:", err);
      setError("Failed to generate summary. Please check your API key and try again.");
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading || !document?.content) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/gemini", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "chat",
          content: document.content.substring(0, 30000),
          question: userMessage,
          history: messages,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to get response");
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    } catch (err) {
      console.error("Chat error:", err);
      setError("Failed to get response. Please try again.");
      setMessages((prev) => prev.slice(0, -1)); // Remove the user message on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearDocument = () => {
    if (document?.url.startsWith("blob:")) {
      URL.revokeObjectURL(document.url);
    }
    setDocument(null);
    setMessages([]);
    setSummary(null);
    setError(null);
    setVisibleLines(50);
    setShowAllContent(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const contentLines = document?.content.split("\n") || [];
  const displayedContent = showAllContent
    ? document?.content
    : contentLines.slice(0, visibleLines).join("\n");
  const hasMoreContent = !showAllContent && contentLines.length > visibleLines;

  const getFileIcon = (fileType: DocumentInfo["fileType"]) => {
    switch (fileType) {
      case "xlsx":
      case "csv":
        return <FileSpreadsheet className="w-5 h-5 text-green-400" />;
      default:
        return <FileText className="w-5 h-5 text-blue-400" />;
    }
  };

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <BackButton />
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          Doc Reader
        </h1>
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

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'Doc Reader'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      {/* Main Content */}
      <div className="pt-20 px-4 pb-4 h-screen flex gap-4">
        {!document ? (
          /* Upload Area */
          <div
            className="flex-1 flex items-center justify-center"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <div
              className="w-full max-w-xl p-12 border-2 border-dashed border-gray-600 rounded-2xl bg-gray-800/50 backdrop-blur-sm hover:border-blue-500 hover:bg-gray-800/70 transition-all duration-300 cursor-pointer group"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="flex flex-col items-center gap-6">
                <div className="p-6 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full group-hover:scale-110 transition-transform duration-300">
                  <Upload className="w-12 h-12 text-blue-400" />
                </div>
                <div className="text-center">
                  <h2 className="text-xl font-semibold mb-2">Upload a Document</h2>
                  <p className="text-gray-400">
                    Drag and drop your file here, or click to browse
                  </p>
                </div>
                <div className="flex flex-wrap gap-2 justify-center text-sm text-gray-500">
                  <span className="px-3 py-1 bg-gray-700/50 rounded-full">PDF</span>
                  <span className="px-3 py-1 bg-gray-700/50 rounded-full">TXT</span>
                  <span className="px-3 py-1 bg-gray-700/50 rounded-full">DOCX</span>
                  <span className="px-3 py-1 bg-gray-700/50 rounded-full">XLSX</span>
                  <span className="px-3 py-1 bg-gray-700/50 rounded-full">CSV</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Powered by Google Gemini AI
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.doc,.docx,.xlsx,.xls,.csv"
                onChange={handleFileUpload}
                className="hidden"
              />
            </div>
          </div>
        ) : (
          <>
            {/* Left Column - Document Preview */}
            <div className="w-1/3 min-w-[300px] flex flex-col gap-3 overflow-hidden">
              <div className="flex items-center justify-between bg-gray-800/50 backdrop-blur-sm rounded-xl p-3">
                <div className="flex items-center gap-2">
                  {getFileIcon(document.fileType)}
                  <span className="font-medium text-sm truncate max-w-[140px]">{document.name}</span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setShowPreview(true)}
                    className="p-1.5 hover:bg-gray-700 rounded-lg transition-colors flex items-center gap-1 text-xs text-blue-400"
                    title="Preview document"
                  >
                    <Eye className="w-4 h-4" />
                    <span className="hidden sm:inline">Preview</span>
                  </button>
                  <button
                    onClick={clearDocument}
                    className="p-1.5 hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Document Content Preview */}
              <div className="flex-1 bg-gray-800/30 backdrop-blur-sm rounded-xl overflow-hidden flex flex-col">
                <div className="p-3 border-b border-gray-700/50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getFileIcon(document.fileType)}
                    <span className="text-sm text-gray-400">Document Content</span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {contentLines.length} lines
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto p-4">
                  <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
                    {displayedContent || "No text content could be extracted from this document."}
                  </pre>
                  <div ref={contentEndRef} />
                </div>
                {hasMoreContent && (
                  <div className="p-3 border-t border-gray-700/50">
                    <button
                      onClick={() => setShowAllContent(true)}
                      className="w-full py-2 px-4 bg-gray-700/50 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors flex items-center justify-center gap-2"
                    >
                      <span>Load all content ({contentLines.length - visibleLines} more lines)</span>
                      <ChevronDown className="w-4 h-4" />
                    </button>
                  </div>
                )}
                {showAllContent && contentLines.length > 50 && (
                  <div className="p-3 border-t border-gray-700/50">
                    <button
                      onClick={() => {
                        setShowAllContent(false);
                        setVisibleLines(50);
                      }}
                      className="w-full py-2 px-4 bg-gray-700/50 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors"
                    >
                      Show less
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Center Column - Summary */}
            <div className="w-1/3 min-w-[300px] flex flex-col gap-3 overflow-hidden">
              <div className="flex items-center gap-2 bg-gray-800/50 backdrop-blur-sm rounded-xl p-3">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <span className="font-medium text-sm">AI Summary</span>
                {summary && (
                  <button
                    onClick={() => copyToClipboard(summary)}
                    className="ml-auto p-1.5 hover:bg-gray-700 rounded-lg transition-colors"
                    title="Copy summary"
                  >
                    {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                  </button>
                )}
              </div>

              <div className="flex-1 bg-gray-800/30 backdrop-blur-sm rounded-xl overflow-hidden flex flex-col">
                {isLoading && !summary ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4">
                    <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
                    <p className="text-gray-400 text-sm">Analyzing document...</p>
                  </div>
                ) : isSummarizing ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4">
                    <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
                    <p className="text-gray-400 text-sm">Generating summary...</p>
                  </div>
                ) : error ? (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4 p-6">
                    <AlertCircle className="w-10 h-10 text-red-400" />
                    <p className="text-red-400 text-sm text-center">{error}</p>
                    <button
                      onClick={() => document.content && generateSummary(document.content)}
                      className="px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-lg text-sm transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                ) : summary ? (
                  <div className="flex-1 overflow-y-auto p-4">
                    <div className="prose prose-invert prose-sm max-w-none">
                      <div className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
                        {summary}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center gap-4 p-6">
                    <FileQuestion className="w-10 h-10 text-gray-500" />
                    <p className="text-gray-400 text-sm text-center">
                      Upload a document to generate an AI summary
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column - Chat */}
            <div className="w-1/3 min-w-[300px] flex flex-col gap-3 overflow-hidden bg-gray-800/30 backdrop-blur-sm rounded-xl border border-gray-700/50">
              <div className="flex items-center gap-2 p-3 border-b border-gray-700/50">
                <MessageCircle className="w-5 h-5 text-blue-400" />
                <span className="font-medium text-sm">Ask Questions</span>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {messages.length === 0 ? (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center">
                      <MessageCircle className="w-10 h-10 text-gray-500 mx-auto mb-3" />
                      <p className="text-gray-400 text-sm mb-4">
                        Ask questions about your document
                      </p>
                      <div className="space-y-2">
                        <p className="text-xs text-gray-500">Try asking:</p>
                        <div className="flex flex-wrap gap-2 justify-center">
                          {["What is the main topic?", "Key takeaways?", "Summarize section 1"].map((q) => (
                            <button
                              key={q}
                              onClick={() => setInput(q)}
                              className="px-3 py-1.5 bg-gray-700/50 hover:bg-gray-700 rounded-full text-xs text-gray-300 transition-colors"
                            >
                              {q}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${
                        message.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[90%] p-3 rounded-2xl ${
                          message.role === "user"
                            ? "bg-blue-500 text-white rounded-br-md"
                            : "bg-gray-700 text-gray-100 rounded-bl-md"
                        }`}
                      >
                        <p className="text-xs whitespace-pre-wrap">{message.content}</p>
                      </div>
                    </div>
                  ))
                )}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-700 p-3 rounded-2xl rounded-bl-md">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="p-3 border-t border-gray-700/50">
                {error && (
                  <div className="mb-2 p-2 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    <p className="text-xs text-red-400">{error}</p>
                  </div>
                )}
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question..."
                    className="flex-1 bg-gray-700 text-white rounded-xl px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400 text-sm"
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!input.trim() || isLoading}
                    className="p-2.5 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-xl transition-colors"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Document Preview Modal */}
      {showPreview && document && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="relative w-[90vw] h-[90vh] bg-gray-900 rounded-2xl overflow-hidden shadow-2xl border border-gray-700">
            {/* Modal Header */}
            <div className="absolute top-0 left-0 right-0 flex items-center justify-between p-4 bg-gray-900/90 backdrop-blur-sm border-b border-gray-700 z-10">
              <div className="flex items-center gap-2">
                {getFileIcon(document.fileType)}
                <span className="font-medium truncate max-w-md">{document.name}</span>
              </div>
              <button
                onClick={() => setShowPreview(false)}
                className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="pt-16 h-full">
              {document.fileType === "pdf" ? (
                <iframe
                  src={document.url}
                  className="w-full h-full"
                  title={document.name}
                />
              ) : document.fileType === "xlsx" || document.fileType === "csv" ? (
                <div className="h-full overflow-auto p-6">
                  <ExcelPreview content={document.content} isCsv={document.fileType === "csv"} />
                </div>
              ) : (
                <div className="h-full overflow-y-auto p-6">
                  <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
                    {document.content}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

// Excel/CSV Preview Component with table formatting
function ExcelPreview({ content, isCsv = false }: { content: string; isCsv?: boolean }) {
  if (isCsv) {
    // For CSV files, directly parse the content
    const lines = content.trim().split("\n");
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-gray-600 text-sm">
          <tbody>
            {lines.map((line, lineIndex) => (
              <tr key={lineIndex} className={lineIndex === 0 ? "bg-gray-700 font-semibold" : ""}>
                {parseCSVLine(line).map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border border-gray-600 px-3 py-2 text-gray-200"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // For Excel files, parse by sheets
  const sheets = content.split(/--- Sheet: (.+?) ---/).filter(Boolean);
  
  return (
    <div className="space-y-6">
      {sheets.map((sheet, index) => {
        if (sheet.includes("\n") && !sheet.startsWith("---")) {
          const lines = sheet.trim().split("\n");
          return (
            <div key={index} className="overflow-x-auto">
              <table className="min-w-full border-collapse border border-gray-600 text-sm">
                <tbody>
                  {lines.map((line, lineIndex) => (
                    <tr key={lineIndex} className={lineIndex === 0 ? "bg-gray-700 font-semibold" : ""}>
                      {line.split(",").map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className="border border-gray-600 px-3 py-2 text-gray-200"
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

// Helper function to parse CSV lines (handles quoted fields)
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  
  result.push(current.trim());
  return result;
}
