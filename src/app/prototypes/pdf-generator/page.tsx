"use client";

import BackButton from "@/components/BackButton";
import DetailsModal from "@/components/DetailsModal";
import prototypeLists from "@/utils/json/prototypeList.json";
import React, { useState, useRef, useCallback } from "react";
import {
  Upload,
  FileText,
  Loader2,
  X,
  Download,
  Image as ImageIcon,
  FileSpreadsheet,
  File,
  AlertCircle,
  CheckCircle,
  Trash2,
  Plus,
} from "lucide-react";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import { jsPDF } from "jspdf";

// Types
interface ConvertedFile {
  id: string;
  name: string;
  type: "image" | "docx" | "xlsx" | "csv";
  originalFile: File;
  status: "pending" | "converting" | "ready" | "error";
  error?: string;
  preview?: string;
}

// A4 dimensions in mm
const A4_WIDTH = 210;
const A4_HEIGHT = 297;
const MARGIN = 10;

// Get file extension
const getFileExtension = (filename: string): string => {
  return filename.split(".").pop()?.toLowerCase() || "";
};

// Get file type
const getFileType = (file: File): ConvertedFile["type"] | null => {
  const ext = getFileExtension(file.name);
  const imageExtensions = ["jpg", "jpeg", "png", "gif", "webp", "bmp"];

  if (imageExtensions.includes(ext)) return "image";
  if (ext === "docx") return "docx";
  if (ext === "xlsx" || ext === "xls") return "xlsx";
  if (ext === "csv") return "csv";

  return null;
};

// Get file icon
const FileIcon = ({ type }: { type: ConvertedFile["type"] }) => {
  switch (type) {
    case "image":
      return <ImageIcon className="h-5 w-5 text-[#718455]" />;
    case "xlsx":
    case "csv":
      return <FileSpreadsheet className="h-5 w-5 text-[#688746]" />;
    default:
      return <FileText className="h-5 w-5 text-[#75839a]" />;
  }
};

export default function PDFGenerator() {
  const [files, setFiles] = useState<ConvertedFile[]>([]);
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Get prototype details from JSON
  const prototype = prototypeLists.find(p => p.slug === 'pdf-generator');
  const [showDetails, setShowDetails] = useState(false);

  // Handle file selection
  const handleFileSelect = useCallback((selectedFiles: FileList | null) => {
    if (!selectedFiles) return;

    const newFiles: ConvertedFile[] = [];
    let hasError = false;

    Array.from(selectedFiles).forEach((file) => {
      const fileType = getFileType(file);
      if (fileType) {
        const newFile: ConvertedFile = {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          name: file.name,
          type: fileType,
          originalFile: file,
          status: "pending",
        };

        // Create preview for images
        if (fileType === "image") {
          newFile.preview = URL.createObjectURL(file);
        }

        newFiles.push(newFile);
      } else {
        hasError = true;
      }
    });

    if (hasError) {
      setError(
        "Some files were skipped. Supported formats: Images (JPG, PNG, GIF, WebP), DOCX, XLSX, CSV"
      );
    }

    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  // Handle file upload
  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFileSelect(e.target.files);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [handleFileSelect]
  );

  // Handle drag and drop
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      handleFileSelect(e.dataTransfer.files);
    },
    [handleFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  // Remove file
  const removeFile = (id: string) => {
    setFiles((prev) => {
      const file = prev.find((f) => f.id === id);
      if (file?.preview) {
        URL.revokeObjectURL(file.preview);
      }
      return prev.filter((f) => f.id !== id);
    });
  };

  // Clear all files
  const clearFiles = () => {
    files.forEach((file) => {
      if (file.preview) {
        URL.revokeObjectURL(file.preview);
      }
    });
    setFiles([]);
    setError(null);
  };

  // Convert image to PDF
  const convertImageToPDF = async (file: File): Promise<jsPDF> => {
    return new Promise((resolve, reject) => {
      const img = new window.Image();
      img.onload = () => {
        try {
          const pdf = new jsPDF({
            orientation: img.width > img.height ? "landscape" : "portrait",
            unit: "mm",
            format: "a4",
          });

          const pageWidth = pdf.internal.pageSize.getWidth();
          const pageHeight = pdf.internal.pageSize.getHeight();

          // Calculate dimensions to fit image on page with margins
          const maxWidth = pageWidth - MARGIN * 2;
          const maxHeight = pageHeight - MARGIN * 2;

          const imgRatio = img.width / img.height;
          let imgWidth = maxWidth;
          let imgHeight = imgWidth / imgRatio;

          if (imgHeight > maxHeight) {
            imgHeight = maxHeight;
            imgWidth = imgHeight * imgRatio;
          }

          // Center image on page
          const x = (pageWidth - imgWidth) / 2;
          const y = (pageHeight - imgHeight) / 2;

          // Add image to PDF
          const canvas = document.createElement("canvas");
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          ctx?.drawImage(img, 0, 0);

          const imgData = canvas.toDataURL("image/jpeg", 0.95);
          pdf.addImage(imgData, "JPEG", x, y, imgWidth, imgHeight);

          resolve(pdf);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = URL.createObjectURL(file);
    });
  };

  // Convert DOCX to PDF
  const convertDOCXToPDF = async (file: File): Promise<jsPDF> => {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    const text = result.value;

    const pdf = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    });

    // Set font
    pdf.setFont("helvetica");
    pdf.setFontSize(12);

    // Add text with word wrap
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const maxWidth = pageWidth - MARGIN * 2;
    const lineHeight = 6;

    const lines = pdf.splitTextToSize(text, maxWidth);
    let y = MARGIN;

    lines.forEach((line: string) => {
      if (y + lineHeight > pageHeight - MARGIN) {
        pdf.addPage();
        y = MARGIN;
      }
      pdf.text(line, MARGIN, y);
      y += lineHeight;
    });

    return pdf;
  };

  // Convert XLSX to PDF
  const convertXLSXToPDF = async (file: File): Promise<jsPDF> => {
    const arrayBuffer = await file.arrayBuffer();
    const workbook = XLSX.read(arrayBuffer, { type: "array" });

    const pdf = new jsPDF({
      orientation: "landscape",
      unit: "mm",
      format: "a4",
    });

    pdf.setFont("helvetica");
    pdf.setFontSize(10);

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();

    let isFirstSheet = true;

    workbook.SheetNames.forEach((sheetName) => {
      if (!isFirstSheet) {
        pdf.addPage();
      }
      isFirstSheet = false;

      // Add sheet title
      pdf.setFontSize(14);
      pdf.setFont("helvetica", "bold");
      pdf.text(`Sheet: ${sheetName}`, MARGIN, MARGIN);
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(9);

      const sheet = workbook.Sheets[sheetName];
      const data = XLSX.utils.sheet_to_json(sheet, { header: 1 }) as string[][];

      let y = MARGIN + 10;
      const cellWidth = 30;
      const cellHeight = 6;
      const maxCellsPerRow = Math.floor((pageWidth - MARGIN * 2) / cellWidth);

      data.forEach((row, rowIndex) => {
        if (y + cellHeight > pageHeight - MARGIN) {
          pdf.addPage();
          y = MARGIN;
        }

        let x = MARGIN;
        const cellsToShow = row.slice(0, maxCellsPerRow);

        cellsToShow.forEach((cell) => {
          const cellText = String(cell || "").substring(0, 15);
          pdf.text(cellText, x, y);
          x += cellWidth;
        });

        y += cellHeight;
      });
    });

    return pdf;
  };

  // Convert CSV to PDF
  const convertCSVToPDF = async (file: File): Promise<jsPDF> => {
    const text = await file.text();
    const lines = text.split("\n").filter((line) => line.trim());

    const pdf = new jsPDF({
      orientation: "landscape",
      unit: "mm",
      format: "a4",
    });

    pdf.setFont("helvetica");
    pdf.setFontSize(10);

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();

    // Add title
    pdf.setFontSize(14);
    pdf.setFont("helvetica", "bold");
    pdf.text(`CSV: ${file.name}`, MARGIN, MARGIN);
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(9);

    let y = MARGIN + 10;
    const cellWidth = 30;
    const cellHeight = 6;
    const maxCellsPerRow = Math.floor((pageWidth - MARGIN * 2) / cellWidth);

    // Parse CSV
    const parseCSVLine = (line: string): string[] => {
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
    };

    lines.forEach((line) => {
      if (y + cellHeight > pageHeight - MARGIN) {
        pdf.addPage();
        y = MARGIN;
      }

      let x = MARGIN;
      const cells = parseCSVLine(line).slice(0, maxCellsPerRow);

      cells.forEach((cell) => {
        const cellText = cell.substring(0, 15);
        pdf.text(cellText, x, y);
        x += cellWidth;
      });

      y += cellHeight;
    });

    return pdf;
  };

  // Convert single file to PDF
  const convertFileToPDF = async (file: ConvertedFile): Promise<jsPDF> => {
    switch (file.type) {
      case "image":
        return convertImageToPDF(file.originalFile);
      case "docx":
        return convertDOCXToPDF(file.originalFile);
      case "xlsx":
        return convertXLSXToPDF(file.originalFile);
      case "csv":
        return convertCSVToPDF(file.originalFile);
      default:
        throw new Error(`Unsupported file type: ${file.type}`);
    }
  };

  // Convert all files
  const convertAllFiles = async () => {
    if (files.length === 0) return;

    setIsConverting(true);
    setError(null);

    try {
      // Update status to converting
      setFiles((prev) =>
        prev.map((f) => ({ ...f, status: "converting" as const }))
      );

      if (files.length === 1) {
        // Single file - convert and download
        const pdf = await convertFileToPDF(files[0]);
        pdf.save(`${files[0].name.replace(/\.[^/.]+$/, "")}.pdf`);

        setFiles((prev) =>
          prev.map((f) => ({ ...f, status: "ready" as const }))
        );
      } else {
        // Multiple files - merge into single PDF
        const pdf = new jsPDF({
          orientation: "portrait",
          unit: "mm",
          format: "a4",
        });

        let isFirst = true;

        for (const file of files) {
          try {
            const filePdf = await convertFileToPDF(file);

            if (!isFirst) {
              pdf.addPage();
            }
            isFirst = false;

            // Copy content from filePdf to main pdf
            const pageCount = filePdf.getNumberOfPages();
            for (let i = 1; i <= pageCount; i++) {
              if (i > 1 || !isFirst) {
                pdf.addPage();
              }
              // Note: jsPDF doesn't have a direct way to merge PDFs
              // For a production app, you'd use pdf-lib or similar
            }

            setFiles((prev) =>
              prev.map((f) =>
                f.id === file.id ? { ...f, status: "ready" as const } : f
              )
            );
          } catch (err) {
            setFiles((prev) =>
              prev.map((f) =>
                f.id === file.id
                  ? {
                      ...f,
                      status: "error" as const,
                      error: "Conversion failed",
                    }
                  : f
              )
            );
          }
        }

        // For multiple files, download each separately
        for (const file of files) {
          if (file.status === "ready") {
            const filePdf = await convertFileToPDF(file);
            filePdf.save(`${file.name.replace(/\.[^/.]+$/, "")}.pdf`);
          }
        }
      }
    } catch (err) {
      console.error("Conversion error:", err);
      setError("Failed to convert files. Please try again.");
    } finally {
      setIsConverting(false);
    }
  };

  // Download single file
  const downloadFile = async (file: ConvertedFile) => {
    try {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id ? { ...f, status: "converting" as const } : f
        )
      );

      const pdf = await convertFileToPDF(file);
      pdf.save(`${file.name.replace(/\.[^/.]+$/, "")}.pdf`);

      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id ? { ...f, status: "ready" as const } : f
        )
      );
    } catch (err) {
      console.error("Download error:", err);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id
            ? { ...f, status: "error" as const, error: "Download failed" }
            : f
        )
      );
    }
  };

  return (
    <main className="relative flex h-dvh flex-col overflow-hidden bg-[#f5f4ef] text-[#252821]">
      {/* Header */}
      <div className="z-10 flex shrink-0 items-center justify-between border-b border-[#e5e3dc] bg-[#fbfaf7] px-4 py-3 sm:px-6">
        <BackButton />
        <h1 className="text-lg font-semibold tracking-tight text-[#292d25] sm:text-xl">
          PDF Generator
        </h1>
        <button
          onClick={() => setShowDetails(true)}
          className="rounded-full border border-[#e1dfd6] bg-white p-2.5 text-[#62665c] transition-colors hover:bg-[#eeede7]"
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
        title={prototype?.title || 'PDF Generator'}
        details={{
          problem: prototype?.problem || '',
          approach: prototype?.approach || '',
          challenges: prototype?.challenges || '',
          optimizations: prototype?.optimizations || '',
          improvements: prototype?.improvements || '',
        }}
      />

      {/* Main Content */}
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-3 sm:gap-4 sm:p-5">
        {/* Upload Area */}
        <div
          className={`group flex shrink-0 cursor-pointer items-center justify-center rounded-2xl border border-dashed px-4 py-4 transition-all duration-300 sm:py-5 ${files.length ? "border-[#deddd5] bg-white" : "border-[#c9cdbf] bg-[#eeefe7] hover:border-[#91a676] hover:bg-[#e9eddf]"}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <div className="flex flex-wrap items-center justify-center gap-4">
            <div className="rounded-2xl bg-white p-3 text-[#688746] shadow-sm transition-transform duration-300 group-hover:scale-105">
              <Upload className="h-6 w-6" />
            </div>
            <div className="text-center">
              <h2 className="mb-1 text-sm font-semibold text-[#30342c] sm:text-base">
                Drop files here or click to upload
              </h2>
              <p className="text-xs text-[#777a72] sm:text-sm">
                Convert images, DOCX, XLSX, and CSV to PDF
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-1.5 text-[10px] text-[#6e7267] sm:gap-2 sm:text-xs">
              <span className="flex items-center gap-1 rounded-full border border-[#e4e2d9] bg-white px-2.5 py-1">
                <ImageIcon className="w-3 h-3" /> Images
              </span>
              <span className="flex items-center gap-1 rounded-full border border-[#e4e2d9] bg-white px-2.5 py-1">
                <FileText className="w-3 h-3" /> DOCX
              </span>
              <span className="flex items-center gap-1 rounded-full border border-[#e4e2d9] bg-white px-2.5 py-1">
                <FileSpreadsheet className="w-3 h-3" /> XLSX
              </span>
              <span className="flex items-center gap-1 rounded-full border border-[#e4e2d9] bg-white px-2.5 py-1">
                <File className="w-3 h-3" /> CSV
              </span>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.docx,.xlsx,.xls,.csv"
            multiple
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="flex shrink-0 items-center gap-2 rounded-xl border border-red-200 bg-red-50 p-3">
            <AlertCircle className="h-5 w-5 shrink-0 text-red-600" />
            <p className="text-sm text-red-700">{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto rounded p-1 hover:bg-red-100"
            >
              <X className="h-4 w-4 text-red-600" />
            </button>
          </div>
        )}

        {/* Files List */}
        {files.length > 0 && (
          <div className="flex-1 flex flex-col gap-3 overflow-hidden">
            {/* Actions Bar */}
              <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 rounded-xl border border-[#e5e3dc] bg-white p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-[#777a72]">
                  {files.length} file{files.length !== 1 ? "s" : ""} selected
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1 rounded-lg border border-[#e5e3dc] bg-white px-3 py-1.5 text-sm text-[#53564f] transition-colors hover:bg-[#f1f0eb]"
                >
                  <Plus className="w-4 h-4" />
                  Add More
                </button>
                <button
                  onClick={clearFiles}
                  className="flex items-center gap-1 rounded-lg border border-[#e5e3dc] bg-white px-3 py-1.5 text-sm text-[#53564f] transition-colors hover:border-red-200 hover:bg-red-50 hover:text-red-700"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All
                </button>
                <button
                  onClick={convertAllFiles}
                  disabled={isConverting || files.length === 0}
                  className="flex items-center gap-1 rounded-lg bg-[#252821] px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-[#3a4033] disabled:bg-[#c9c9c2]"
                >
                  {isConverting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Converting...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      Convert All to PDF
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Files Grid */}
            <div className="min-h-0 flex-1 overflow-y-auto pr-1">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="overflow-hidden rounded-xl border border-[#e5e3dc] bg-white transition-colors hover:border-[#a9b895]"
                  >
                    {/* Preview */}
                    <div className="relative flex aspect-video items-center justify-center overflow-hidden bg-[#f0efe9]">
                      {file.type === "image" && file.preview ? (
                        <img
                          src={file.preview}
                          alt={file.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="rounded-2xl bg-white p-4">
                          <FileIcon type={file.type} />
                        </div>
                      )}

                      {/* Status Overlay */}
                      {file.status === "converting" && (
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                          <Loader2 className="h-8 w-8 animate-spin text-[#688746]" />
                        </div>
                      )}
                      {file.status === "ready" && (
                        <div className="absolute right-2 top-2 rounded-full bg-[#edf2e5] p-1">
                          <CheckCircle className="h-4 w-4 text-[#688746]" />
                        </div>
                      )}
                      {file.status === "error" && (
                        <div className="absolute right-2 top-2 rounded-full bg-red-50 p-1">
                          <AlertCircle className="h-4 w-4 text-red-600" />
                        </div>
                      )}
                    </div>

                    {/* Info */}
                    <div className="p-3 text-[#30342c]">
                      <div className="flex items-center gap-2 mb-2">
                        <FileIcon type={file.type} />
                        <span className="text-sm font-medium truncate flex-1">
                          {file.name}
                        </span>
                      </div>

                      {file.error && (
                        <p className="mb-2 text-xs text-red-700">{file.error}</p>
                      )}

                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => downloadFile(file)}
                          disabled={file.status === "converting"}
                          className="flex flex-1 items-center justify-center gap-1 rounded-lg bg-[#edf2e5] px-3 py-1.5 text-xs text-[#557736] transition-colors hover:bg-[#e1eacf] disabled:bg-[#eeede7]"
                        >
                          {file.status === "converting" ? (
                            <Loader2 className="w-3 h-3 animate-spin" />
                          ) : (
                            <Download className="w-3 h-3" />
                          )}
                          Download PDF
                        </button>
                        <button
                          onClick={() => removeFile(file.id)}
                          className="rounded-lg p-1.5 transition-colors hover:bg-red-50"
                        >
                          <X className="h-4 w-4 text-[#85877f] hover:text-red-600" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {files.length === 0 && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="mb-4 inline-block rounded-3xl bg-[#eeefe7] p-6">
                <File className="h-12 w-12 text-[#82936e]" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-[#53564f]">
                No files selected
              </h3>
              <p className="text-sm text-[#85877f]">
                Upload images, DOCX, XLSX, or CSV files to convert them to PDF
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
