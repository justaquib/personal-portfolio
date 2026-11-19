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
    <main className="min-h-screen bg-[#f3f6f4] text-[#182b27]">
      <header className="sticky top-0 z-30 border-b border-[#dce5df] bg-white/95 backdrop-blur-xl">
        <div className="mx-auto flex h-[72px] max-w-[1680px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="rounded-xl bg-[#183b33] p-2.5 shadow-sm">
              <FileText className="h-5 w-5 text-[#d5f1df]" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-base font-semibold tracking-tight text-[#18332d] sm:text-lg">RAG Document Studio</h1>
                <span className="rounded-full border border-[#b8d8c7] bg-[#e9f5ed] px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.14em] text-[#27634d]">RAG</span>
              </div>
              <p className="hidden text-xs text-[#71827b] sm:block">Summarize and ask questions across your files</p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <div className="rounded-full bg-[#183b33]"><BackButton /></div>
            <button
              onClick={() => setShowDetails(true)}
              className="rounded-xl border border-[#dce5df] bg-white p-2.5 text-[#5e746c] transition hover:border-[#afcbb9] hover:bg-[#f3f8f4] hover:text-[#214f3e]"
              title="About this prototype"
              aria-label="About this prototype"
            >
              <Sparkles className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      <DetailsModal
        isOpen={showDetails}
        onClose={() => setShowDetails(false)}
        title={prototype?.title || 'RAG Document Studio'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      <div className={`mx-auto flex w-full max-w-[1680px] px-4 sm:px-6 lg:px-8 ${document ? "h-[calc(100dvh-72px)] min-h-0 flex-col overflow-hidden py-3 sm:py-4" : "flex-col py-6 lg:py-8"}`}>
        {!document ? (
          <div className="grid min-h-[calc(100vh-136px)] items-center gap-10 py-8 lg:grid-cols-[1.02fr_.98fr] lg:gap-16">
            <section className="mx-auto w-full max-w-2xl">
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[#cee4d5] bg-white px-3 py-1.5 text-xs font-medium text-[#3b7055] shadow-sm">
                <span className="h-2 w-2 rounded-full bg-[#4e9a70]" />
                Private document workspace
              </div>
              <h2 className="max-w-xl text-4xl font-semibold leading-[1.08] tracking-[-0.055em] text-[#18332d] sm:text-5xl lg:text-6xl">
                Make your documents
                <span className="block text-[#4c8a68]">easy to understand.</span>
              </h2>
              <p className="mt-6 max-w-xl text-base leading-7 text-[#63756e] sm:text-lg">
                Bring in a file, get a clear summary, then ask questions with the document as context.
              </p>
              <div className="mt-8 grid max-w-xl grid-cols-1 gap-3 sm:grid-cols-3">
                {[
                  [FileText, "Read", "PDFs, docs, sheets"],
                  [Sparkles, "Summarize", "Find the main ideas"],
                  [MessageCircle, "Ask", "Explore the details"],
                ].map(([Icon, title, description]) => {
                  const StepIcon = Icon as typeof FileText;
                  return (
                    <div key={title as string} className="rounded-2xl border border-[#dfe8e2] bg-white/75 p-4">
                      <StepIcon className="h-4 w-4 text-[#4d8d68]" />
                      <p className="mt-3 text-sm font-semibold text-[#27453a]">{title as string}</p>
                      <p className="mt-1 text-xs leading-5 text-[#788981]">{description as string}</p>
                    </div>
                  );
                })}
              </div>
            </section>

            <section
              className="mx-auto w-full max-w-xl"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.doc,.docx,.xlsx,.xls,.csv"
                onChange={handleFileUpload}
                className="hidden"
              />
              <div className="relative overflow-hidden rounded-[28px] border border-[#cfe1d4] bg-white p-5 shadow-[0_24px_80px_-40px_rgba(33,83,59,0.35)] sm:p-7">
                <div className="absolute -right-14 -top-20 h-56 w-56 rounded-full bg-[#e4f2e6] blur-2xl" />
                <div className="relative rounded-[22px] border-2 border-dashed border-[#b8d5c2] bg-[#f8fbf8] px-5 py-10 text-center transition hover:border-[#67a67e] hover:bg-[#f3faf4] sm:px-9 sm:py-14">
                  <div className="mx-auto grid h-16 w-16 place-items-center rounded-2xl bg-[#e1f0e5] text-[#397b54] shadow-inner">
                    {isLoading ? <Loader2 className="h-7 w-7 animate-spin" /> : <Upload className="h-7 w-7" />}
                  </div>
                  <h3 className="mt-6 text-xl font-semibold tracking-tight text-[#203b31]">
                    {isLoading ? "Preparing your document…" : "Drop a document to get started"}
                  </h3>
                  <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-[#71827a]">
                    {isLoading ? "Extracting text and preparing your workspace." : "We’ll extract its text, create a summary, and open a grounded Q&A workspace."}
                  </p>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isLoading}
                    className="mt-6 inline-flex items-center gap-2 rounded-xl bg-[#1f5b43] px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#174a36] disabled:cursor-wait disabled:opacity-60"
                  >
                    <Upload className="h-4 w-4" /> Choose a file
                  </button>
                  <div className="mt-7 flex flex-wrap justify-center gap-2">
                    {["PDF", "DOCX", "TXT", "XLSX", "CSV"].map((format) => (
                      <span key={format} className="rounded-lg border border-[#e0e9e2] bg-white px-2.5 py-1 text-[10px] font-semibold tracking-wide text-[#6d7d75]">{format}</span>
                    ))}
                  </div>
                  <p className="mt-5 text-[11px] text-[#8a9991]">Text is extracted in your browser · Gemini helps summarize and answer</p>
                </div>
                {error && (
                  <div className="relative mt-4 flex items-start gap-2 rounded-xl border border-[#f1c9c3] bg-[#fff4f2] p-3 text-sm text-[#a34338]">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                    <p>{error}</p>
                  </div>
                )}
              </div>

              <div className="mt-5 flex items-center justify-center gap-2 text-xs text-[#7d8e85]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#73a785]" />
                Built for focused, file-based exploration
              </div>
            </section>
          </div>
        ) : (
          <>
            <div className="mb-3 flex shrink-0 flex-wrap items-center justify-between gap-3 sm:mb-4">
              <div className="min-w-0">
                <p className="text-[10px] font-bold uppercase tracking-[0.17em] text-[#759084]">Document workspace</p>
                <h2 className="mt-1 truncate text-lg font-semibold tracking-tight text-[#203a30]">{document.name}</h2>
              </div>
              <div className="flex items-center gap-2">
                <span className="rounded-full border border-[#cde2d2] bg-[#eaf5ed] px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.12em] text-[#3a7553]">RAG · Ready</span>
                <button onClick={clearDocument} className="inline-flex items-center gap-2 rounded-xl border border-[#d6e2da] bg-white px-3 py-2 text-xs font-medium text-[#61746a] transition hover:border-[#c4d7ca] hover:bg-[#f9fcf9] hover:text-[#28533d]">
                  <X className="h-3.5 w-3.5" /> Replace file
                </button>
                <input ref={fileInputRef} type="file" accept=".pdf,.txt,.doc,.docx,.xlsx,.xls,.csv" onChange={handleFileUpload} className="hidden" />
              </div>
            </div>

            <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto pb-2 xl:grid-cols-[.94fr_1fr_1.08fr] xl:overflow-hidden xl:pb-0">
              <section className="flex h-[min(62dvh,560px)] min-h-[360px] flex-col overflow-hidden rounded-2xl border border-[#dce6df] bg-white shadow-[0_8px_28px_-24px_rgba(22,56,40,.38)] xl:h-full xl:min-h-0">
                <div className="flex items-center justify-between border-b border-[#e6ece8] px-4 py-3.5">
                  <div className="flex min-w-0 items-center gap-2.5">
                    <div className="rounded-lg bg-[#e9f3ec] p-2">{getFileIcon(document.fileType)}</div>
                    <div className="min-w-0"><h3 className="text-sm font-semibold text-[#29473a]">Source document</h3><p className="truncate text-[11px] text-[#8a9991]">{contentLines.length} lines</p></div>
                  </div>
                  <button onClick={() => setShowPreview(true)} className="inline-flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-2 text-xs font-medium text-[#46775b] transition hover:bg-[#f0f7f1]" title="Preview document">
                    <Eye className="h-4 w-4" /><span>Preview</span>
                  </button>
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto bg-[#fbfcfb] p-4 sm:p-5">
                  <pre className="whitespace-pre-wrap font-mono text-[11px] leading-6 text-[#586c62]">{displayedContent || "No text content could be extracted from this document."}</pre>
                  <div ref={contentEndRef} />
                </div>
                {hasMoreContent && <div className="border-t border-[#e6ece8] p-3"><button onClick={() => setShowAllContent(true)} className="flex w-full items-center justify-center gap-2 rounded-lg bg-[#f2f7f3] px-4 py-2.5 text-xs font-medium text-[#4a7159] transition hover:bg-[#e8f2ea]">Load all content ({contentLines.length - visibleLines} more lines)<ChevronDown className="h-4 w-4" /></button></div>}
                {showAllContent && contentLines.length > 50 && <div className="border-t border-[#e6ece8] p-3"><button onClick={() => { setShowAllContent(false); setVisibleLines(50); }} className="w-full rounded-lg bg-[#f2f7f3] px-4 py-2.5 text-xs font-medium text-[#4a7159] transition hover:bg-[#e8f2ea]">Show less</button></div>}
              </section>

              <section className="flex h-[min(62dvh,560px)] min-h-[360px] flex-col overflow-hidden rounded-2xl border border-[#dce6df] bg-white shadow-[0_8px_28px_-24px_rgba(22,56,40,.38)] xl:h-full xl:min-h-0">
                <div className="flex items-center justify-between border-b border-[#e6ece8] px-4 py-3.5">
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-[#eff3df] p-2"><Sparkles className="h-4 w-4 text-[#8a8c47]" /></div>
                    <div><h3 className="text-sm font-semibold text-[#29473a]">AI summary</h3><p className="text-[11px] text-[#8a9991]">Key ideas from this file</p></div>
                  </div>
                  {summary && <button onClick={() => copyToClipboard(summary)} className="rounded-lg p-2 text-[#70857a] transition hover:bg-[#f2f7f3] hover:text-[#356348]" title="Copy summary">{copied ? <Check className="h-4 w-4 text-[#4b9565]" /> : <Copy className="h-4 w-4" />}</button>}
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto bg-white p-5">
                  {isLoading && !summary ? (
                    <div className="flex h-full min-h-[250px] flex-col items-center justify-center gap-3"><Loader2 className="h-7 w-7 animate-spin text-[#57916a]" /><p className="text-sm text-[#74867d]">Analyzing document…</p></div>
                  ) : isSummarizing ? (
                    <div className="flex h-full min-h-[250px] flex-col items-center justify-center gap-3"><Loader2 className="h-7 w-7 animate-spin text-[#57916a]" /><p className="text-sm text-[#74867d]">Preparing your summary…</p></div>
                  ) : error ? (
                    <div className="flex h-full min-h-[250px] flex-col items-center justify-center gap-3 px-5 text-center"><AlertCircle className="h-8 w-8 text-[#bd6758]" /><p className="text-sm leading-6 text-[#9b5145]">{error}</p><button onClick={() => document.content && generateSummary(document.content)} className="rounded-lg bg-[#1f5b43] px-4 py-2 text-xs font-semibold text-white transition hover:bg-[#174a36]">Try again</button></div>
                  ) : summary ? (
                    <div className="whitespace-pre-wrap text-sm leading-7 text-[#445b50]">{summary}</div>
                  ) : (
                    <div className="flex h-full min-h-[250px] flex-col items-center justify-center gap-3 px-5 text-center"><FileQuestion className="h-8 w-8 text-[#9aac9f]" /><p className="text-sm text-[#71847a]">Your summary will appear here.</p></div>
                  )}
                </div>
              </section>

              <section className="flex h-[min(68dvh,600px)] min-h-[400px] flex-col overflow-hidden rounded-2xl border border-[#d5e2d9] bg-white shadow-[0_8px_28px_-24px_rgba(22,56,40,.38)] xl:h-full xl:min-h-0">
                <div className="flex items-center gap-2.5 border-b border-[#e6ece8] px-4 py-3.5">
                  <div className="rounded-lg bg-[#e9f3ec] p-2"><MessageCircle className="h-4 w-4 text-[#4a8460]" /></div>
                  <div><h3 className="text-sm font-semibold text-[#29473a]">Ask this document</h3><p className="text-[11px] text-[#8a9991]">Answers use the uploaded content</p></div>
                </div>
                <div className="min-h-0 flex-1 space-y-3 overflow-y-auto bg-[#fbfcfb] p-4 sm:p-5">
                  {messages.length === 0 ? (
                    <div className="flex h-full min-h-[300px] items-center justify-center">
                      <div className="max-w-sm text-center">
                        <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-[#eaf4ec]"><MessageCircle className="h-5 w-5 text-[#4a8460]" /></div>
                        <p className="mt-4 text-sm font-medium text-[#486054]">What would you like to know?</p>
                        <p className="mt-1 text-xs leading-5 text-[#8a9991]">Start with one of these or write your own question.</p>
                        <div className="mt-5 flex flex-wrap justify-center gap-2">
                          {["What is the main topic?", "What are the key takeaways?", "Summarize section 1"].map((question) => <button key={question} onClick={() => setInput(question)} className="rounded-full border border-[#dce7df] bg-white px-3 py-2 text-[11px] text-[#5a7565] transition hover:border-[#b5d3be] hover:bg-[#f1f8f2]">{question}</button>)}
                        </div>
                      </div>
                    </div>
                  ) : messages.map((message, index) => (
                    <div key={`${message.role}-${index}`} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[90%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-xs leading-6 ${message.role === "user" ? "rounded-br-md bg-[#1f5b43] text-white" : "rounded-bl-md border border-[#e2eae4] bg-white text-[#4b6255] shadow-sm"}`}>{message.content}</div>
                    </div>
                  ))}
                  {isLoading && <div className="flex justify-start"><div className="rounded-2xl rounded-bl-md border border-[#e2eae4] bg-white p-3"><Loader2 className="h-4 w-4 animate-spin text-[#57916a]" /></div></div>}
                  <div ref={messagesEndRef} />
                </div>
                <div className="border-t border-[#e6ece8] bg-white p-3 sm:p-4">
                  {error && <div className="mb-2 flex items-start gap-2 rounded-lg border border-[#f1c9c3] bg-[#fff4f2] p-2.5 text-xs text-[#a34338]"><AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" /><p>{error}</p></div>}
                  <div className="flex items-center gap-2 rounded-xl border border-[#dce6df] bg-[#fbfcfb] p-1.5 pl-3 transition focus-within:border-[#8eb99a] focus-within:ring-2 focus-within:ring-[#d9ecdd]">
                    <input type="text" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask a question…" className="min-w-0 flex-1 bg-transparent py-2 text-sm text-[#29473a] outline-none placeholder:text-[#9aa9a1]" disabled={isLoading} />
                    <button onClick={handleSendMessage} disabled={!input.trim() || isLoading} className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[#1f5b43] text-white transition hover:bg-[#174a36] disabled:cursor-not-allowed disabled:bg-[#c7d5cb]" aria-label="Send question"><Send className="h-4 w-4" /></button>
                  </div>
                </div>
              </section>
            </div>
          </>
        )}
      </div>

      {showPreview && document && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#10221d]/55 p-3 backdrop-blur-sm sm:p-6">
          <div className="relative flex h-[92vh] w-[96vw] max-w-6xl flex-col overflow-hidden rounded-2xl border border-[#dce6df] bg-white shadow-2xl">
            <div className="flex shrink-0 items-center justify-between border-b border-[#e5ece7] px-4 py-3">
              <div className="flex min-w-0 items-center gap-2.5">{getFileIcon(document.fileType)}<span className="truncate text-sm font-semibold text-[#29473a]">{document.name}</span></div>
              <button onClick={() => setShowPreview(false)} className="rounded-lg p-2 text-[#75877d] transition hover:bg-[#f0f6f1] hover:text-[#345c43]" aria-label="Close preview"><X className="h-5 w-5" /></button>
            </div>
            <div className="min-h-0 flex-1">
              {document.fileType === "pdf" ? <iframe src={document.url} className="h-full w-full" title={document.name} /> : document.fileType === "xlsx" || document.fileType === "csv" ? <div className="h-full overflow-auto p-4 sm:p-6"><ExcelPreview content={document.content} isCsv={document.fileType === "csv"} /></div> : <div className="h-full overflow-y-auto p-5 sm:p-7"><pre className="whitespace-pre-wrap font-mono text-sm leading-7 text-[#52675c]">{document.content}</pre></div>}
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
        <table className="min-w-full border-collapse border border-[#dce6df] text-sm">
          <tbody>
            {lines.map((line, lineIndex) => (
              <tr key={lineIndex} className={lineIndex === 0 ? "bg-[#eaf3ec] font-semibold" : "odd:bg-white even:bg-[#f8faf8]"}>
                {parseCSVLine(line).map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="border border-[#e2eae4] px-3 py-2 text-[#4b6255]"
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
              <table className="min-w-full border-collapse border border-[#dce6df] text-sm">
                <tbody>
                  {lines.map((line, lineIndex) => (
                    <tr key={lineIndex} className={lineIndex === 0 ? "bg-[#eaf3ec] font-semibold" : "odd:bg-white even:bg-[#f8faf8]"}>
                      {line.split(",").map((cell, cellIndex) => (
                        <td
                          key={cellIndex}
                          className="border border-[#e2eae4] px-3 py-2 text-[#4b6255]"
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
