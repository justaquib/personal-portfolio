"use client";

import BackButton from "@/components/BackButton";
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
      return <ImageIcon className="w-5 h-5 text-purple-400" />;
    case "xlsx":
    case "csv":
      return <FileSpreadsheet className="w-5 h-5 text-green-400" />;
    default:
      return <FileText className="w-5 h-5 text-blue-400" />;
  }
};

export default function PDFGenerator() {
  const [files, setFiles] = useState<ConvertedFile[]>([]);
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    <main className="relative min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
        <BackButton />
        <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
          PDF Generator
        </h1>
        <div className="w-10" />
      </div>

      {/* Main Content */}
      <div className="pt-20 px-4 pb-4 h-screen flex flex-col gap-4">
        {/* Upload Area */}
        <div
          className="flex-shrink-0 p-8 border-2 border-dashed border-gray-600 rounded-2xl bg-gray-800/50 backdrop-blur-sm hover:border-purple-500 hover:bg-gray-800/70 transition-all duration-300 cursor-pointer group"
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full group-hover:scale-110 transition-transform duration-300">
              <Upload className="w-8 h-8 text-purple-400" />
            </div>
            <div className="text-center">
              <h2 className="text-lg font-semibold mb-1">
                Drop files here or click to upload
              </h2>
              <p className="text-gray-400 text-sm">
                Convert images, DOCX, XLSX, and CSV to PDF
              </p>
            </div>
            <div className="flex flex-wrap gap-2 justify-center text-xs text-gray-500">
              <span className="px-2 py-1 bg-gray-700/50 rounded-full flex items-center gap-1">
                <ImageIcon className="w-3 h-3" /> Images
              </span>
              <span className="px-2 py-1 bg-gray-700/50 rounded-full flex items-center gap-1">
                <FileText className="w-3 h-3" /> DOCX
              </span>
              <span className="px-2 py-1 bg-gray-700/50 rounded-full flex items-center gap-1">
                <FileSpreadsheet className="w-3 h-3" /> XLSX
              </span>
              <span className="px-2 py-1 bg-gray-700/50 rounded-full flex items-center gap-1">
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
          <div className="flex-shrink-0 p-3 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <p className="text-red-400 text-sm">{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto p-1 hover:bg-red-500/20 rounded"
            >
              <X className="w-4 h-4 text-red-400" />
            </button>
          </div>
        )}

        {/* Files List */}
        {files.length > 0 && (
          <div className="flex-1 flex flex-col gap-3 overflow-hidden">
            {/* Actions Bar */}
            <div className="flex-shrink-0 flex items-center justify-between bg-gray-800/50 backdrop-blur-sm rounded-xl p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-400">
                  {files.length} file{files.length !== 1 ? "s" : ""} selected
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Add More
                </button>
                <button
                  onClick={clearFiles}
                  className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-red-600 rounded-lg text-sm transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear All
                </button>
                <button
                  onClick={convertAllFiles}
                  disabled={isConverting || files.length === 0}
                  className="flex items-center gap-1 px-4 py-1.5 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-gray-600 disabled:to-gray-600 rounded-lg text-sm font-medium transition-all"
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
            <div className="flex-1 overflow-y-auto">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700/50 hover:border-purple-500/50 transition-colors"
                  >
                    {/* Preview */}
                    <div className="aspect-video bg-gray-900/50 flex items-center justify-center relative overflow-hidden">
                      {file.type === "image" && file.preview ? (
                        <img
                          src={file.preview}
                          alt={file.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="p-4 bg-gray-800/50 rounded-full">
                          <FileIcon type={file.type} />
                        </div>
                      )}

                      {/* Status Overlay */}
                      {file.status === "converting" && (
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                          <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
                        </div>
                      )}
                      {file.status === "ready" && (
                        <div className="absolute top-2 right-2 p-1 bg-green-500/20 rounded-full">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        </div>
                      )}
                      {file.status === "error" && (
                        <div className="absolute top-2 right-2 p-1 bg-red-500/20 rounded-full">
                          <AlertCircle className="w-4 h-4 text-red-400" />
                        </div>
                      )}
                    </div>

                    {/* Info */}
                    <div className="p-3">
                      <div className="flex items-center gap-2 mb-2">
                        <FileIcon type={file.type} />
                        <span className="text-sm font-medium truncate flex-1">
                          {file.name}
                        </span>
                      </div>

                      {file.error && (
                        <p className="text-xs text-red-400 mb-2">{file.error}</p>
                      )}

                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => downloadFile(file)}
                          disabled={file.status === "converting"}
                          className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 bg-purple-500/20 hover:bg-purple-500/30 disabled:bg-gray-700 rounded-lg text-xs text-purple-300 transition-colors"
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
                          className="p-1.5 hover:bg-red-500/20 rounded-lg transition-colors"
                        >
                          <X className="w-4 h-4 text-gray-400 hover:text-red-400" />
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
              <div className="p-6 bg-gray-800/30 rounded-full inline-block mb-4">
                <File className="w-12 h-12 text-gray-500" />
              </div>
              <h3 className="text-lg font-medium text-gray-400 mb-2">
                No files selected
              </h3>
              <p className="text-sm text-gray-500">
                Upload images, DOCX, XLSX, or CSV files to convert them to PDF
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
