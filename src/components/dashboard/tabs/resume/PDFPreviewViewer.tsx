'use client'

import { useState } from 'react'
import { ResumeData } from './types'
import { PDFPreview, downloadResumePDF, templateColors } from './templates'
import { Download, ZoomIn, ZoomOut } from 'lucide-react'

type TemplateType = 'modern' | 'classic' | 'minimal' | 'creative'

interface PDFPreviewViewerProps {
  resumeData: ResumeData
}

export function PDFPreviewViewer({ resumeData }: PDFPreviewViewerProps) {
  const [scale, setScale] = useState(1)
  const [template, setTemplate] = useState<TemplateType>(
    (resumeData.template as TemplateType) || 'modern'
  )

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.25, 2))
  }

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.25, 0.5))
  }

  const handleDownload = async () => {
    await downloadResumePDF(resumeData, template)
  }

  const handleTemplateChange = (newTemplate: TemplateType) => {
    setTemplate(newTemplate)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between p-3 bg-gray-100 border-b rounded-t-xl">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-700">PDF Preview</span>
          <span className="text-xs text-gray-500">|</span>
          <span className="text-xs text-gray-500 capitalize">{template} template</span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Template Selector */}
          <div className="flex items-center gap-1 mr-4">
            {(['modern', 'classic', 'minimal', 'creative'] as TemplateType[]).map((t) => (
              <button
                key={t}
                onClick={() => handleTemplateChange(t)}
                className={`px-2 py-1 text-xs rounded transition-colors ${
                  template === t 
                    ? 'bg-gray-800 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                style={template === t ? { backgroundColor: templateColors[t]?.accent || '#333' } : {}}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

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
        <div 
          className="flex justify-center"
          style={{ transform: `scale(${scale})`, transformOrigin: 'top center' }}
        >
          <div className="w-full bg-white shadow-lg" style={{ height: '297mm' }}>
            <PDFPreview 
              data={resumeData} 
              template={template} 
            />
          </div>
        </div>
      </div>
    </div>
  )
}
