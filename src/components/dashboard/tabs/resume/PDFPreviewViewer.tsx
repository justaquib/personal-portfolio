'use client'

import { useEffect, useState, useRef } from 'react'
import { ResumeData } from './types'
import { generateResumePDF } from './pdfGenerator'
import { Download, FileDown, ZoomIn, ZoomOut, ChevronLeft, ChevronRight } from 'lucide-react'

interface PDFPreviewViewerProps {
  resumeData: ResumeData
}

export function PDFPreviewViewer({ resumeData }: PDFPreviewViewerProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(true)
  const [scale, setScale] = useState(1)
  const iframeRef = useRef<HTMLIFrameElement>(null)

  useEffect(() => {
    const generatePDF = async () => {
      setIsGenerating(true)
      try {
        // Generate the PDF using jsPDF
        const doc = generateResumePDF(resumeData)
        
        // Get the PDF as a blob
        const pdfBlob = doc.output('blob')
        
        // Create a URL for the blob
        const url = URL.createObjectURL(pdfBlob)
        setPdfUrl(url)
      } catch (error) {
        console.error('Error generating PDF:', error)
      } finally {
        setIsGenerating(false)
      }
    }

    generatePDF()

    // Cleanup the URL when component unmounts or resumeData changes
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
    }
  }, [resumeData])

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.25, 2))
  }

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.25, 0.5))
  }

  const handleDownload = () => {
    const doc = generateResumePDF(resumeData)
    const fileName = resumeData.personalInfo.name 
      ? `${resumeData.personalInfo.name.replace(/\s+/g, '_')}_Resume.pdf`
      : 'Resume.pdf'
    doc.save(fileName)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-gray-100 border-b rounded-t-xl">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">PDF Preview</span>
          {isGenerating && (
            <span className="text-xs text-gray-500">Generating...</span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {/* Zoom Controls */}
          <div className="flex items-center gap-1 mr-4">
            <button
              onClick={handleZoomOut}
              disabled={scale <= 0.5}
              className="p-1.5 hover:bg-gray-200 rounded disabled:opacity-50"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-600 min-w-[50px] text-center">
              {Math.round(scale * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              disabled={scale >= 2}
              className="p-1.5 hover:bg-gray-200 rounded disabled:opacity-50"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </div>

          {/* Download Button */}
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 text-white text-sm rounded-lg hover:bg-gray-700"
          >
            <Download className="w-4 h-4" />
            Download
          </button>
        </div>
      </div>

      {/* PDF Viewer */}
      <div className="flex-1 bg-gray-200 p-4 overflow-auto rounded-b-xl min-h-[600px]">
        {isGenerating ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-8 h-8 border-2 border-gray-400 border-t-gray-800 rounded-full animate-spin mx-auto mb-3"></div>
              <p className="text-gray-600">Generating PDF preview...</p>
            </div>
          </div>
        ) : pdfUrl ? (
          <div className="flex justify-center">
            <iframe
              ref={iframeRef}
              src={`${pdfUrl}#toolbar=0&navpanes=0&scrollbar=0&zoom=${scale * 100}`}
              className="w-full max-w-[210mm] bg-white shadow-lg"
              style={{ 
                height: '297mm',
                transform: `scale(${scale})`,
                transformOrigin: 'top center'
              }}
              title="PDF Preview"
            />
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-600">Failed to generate PDF preview</p>
          </div>
        )}
      </div>
    </div>
  )
}
