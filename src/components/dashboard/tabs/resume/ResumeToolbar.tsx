'use client'

import { useRef, ChangeEvent } from 'react'
import { 
  Upload, Layout, Save, FileText, Settings, Edit3, Eye, RefreshCw, X, FileDown
} from 'lucide-react'
import { ResumeData } from './types'
import { TemplateBuilder } from './TemplateBuilder'
import { Tooltip } from '@/components/Tooltip'
import { Badge } from '@/components/Badge'

interface ResumeToolbarProps {
  resumeData: ResumeData
  savedResumes: ResumeData[]
  showPreview: boolean
  showPDFPreview: boolean
  showTemplates: boolean
  showTemplateBuilder: boolean
  activeTool: string | null
  isSuperAdmin: boolean
  isAdmin: boolean
  isImportingPDF: boolean
  importProgress: string
  selectedTemplate: { id: string; name: string }
  resumeName: string
  isSaving: boolean
  hasResumeContent: () => boolean
  fileInputRef: React.RefObject<HTMLInputElement | null>
  showResumesDropdown: boolean
  onImportPDF: (e: React.ChangeEvent<HTMLInputElement>) => void
  onToggleTemplates: () => void
  onTogglePreview: (preview: boolean) => void
  onTogglePDFPreview: (preview: boolean) => void
  onToggleTemplateBuilder: () => void
  onResetSectionOrder: () => void
  onSave: () => void
  onUpdateResume: () => void
  onLoadResume: (resume: ResumeData) => void
  onSelectTemplate: (templateId: string) => void
  onCloseTemplates: () => void
  onCloseTemplateBuilder: () => void
  onToggleResumesDropdown: () => void
}

export function ResumeToolbar({
  resumeData,
  savedResumes,
  showPreview,
  showPDFPreview,
  showTemplates,
  showTemplateBuilder,
  activeTool,
  isSuperAdmin,
  isAdmin,
  isImportingPDF,
  importProgress,
  selectedTemplate,
  resumeName,
  isSaving,
  hasResumeContent,
  fileInputRef,
  showResumesDropdown,
  onImportPDF,
  onToggleTemplates,
  onTogglePreview,
  onTogglePDFPreview,
  onToggleTemplateBuilder,
  onResetSectionOrder,
  onSave,
  onUpdateResume,
  onLoadResume,
  onSelectTemplate,
  onCloseTemplates,
  onCloseTemplateBuilder,
  onToggleResumesDropdown,
}: ResumeToolbarProps) {
  const getButtonStyle = (isActive: boolean) => ({
    backgroundColor: isActive ? '#212529' : '#e9ecef',
    color: isActive ? '#ffffff' : '#212529',
  })

  const TEMPLATES = [
    { id: 'modern', name: 'Modern' },
    { id: 'classic', name: 'Classic' },
    { id: 'minimal', name: 'Minimal' },
    { id: 'creative', name: 'Creative' },
  ]

  return (
    <>
      {/* File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        onChange={onImportPDF}
        className="hidden"
      />
      
      <div className="flex flex-wrap items-center gap-2 overflow-visible">
        {/* Import PDF */}
        <Tooltip content={isImportingPDF ? 'Importing...' : 'Import from PDF'} position="bottom">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isImportingPDF}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(false)}
          >
            {isImportingPDF ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <Upload className="w-5 h-5" />
            )}
          </button>
        </Tooltip>

        {/* Template Selector */}
        <Tooltip content={`Template: ${selectedTemplate.name}`} position="bottom">
          <button
            onClick={onToggleTemplates}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(showTemplates)}
          >
            <Layout className="w-5 h-5" />
          </button>
        </Tooltip>

        {/* Save */}
        <Tooltip content={resumeData.id ? 'Update Resume' : 'Save Resume'} position="bottom">
          <button
            onClick={resumeData.id ? onUpdateResume : onSave}
            disabled={isSaving || !hasResumeContent()}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(false)}
          >
            <Save className="w-5 h-5" />
          </button>
        </Tooltip>

        {/* My Resumes Dropdown */}
        {savedResumes.length > 0 && (
          <div className="relative">
            <Tooltip content={`My Resumes (${savedResumes.length})`} position="bottom">
              <button
                onClick={onToggleResumesDropdown}
                className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors relative"
                style={getButtonStyle(false)}
              >
                <FileText className="w-5 h-5" />
                <Badge 
                  variant="primary" 
                  size="sm" 
                  className="absolute -top-1 -right-1"
                >
                  {savedResumes.length}
                </Badge>
              </button>
            </Tooltip>
            {/* Saved Resumes Dropdown */}
            {showResumesDropdown && (
              <div className="absolute top-full left-0 mt-2 w-64 bg-white rounded-xl shadow-lg border border-gray-200 z-50">
                <div className="p-2">
                  <div className="text-xs font-semibold text-gray-500 px-3 py-2">
                    Your Saved Resumes
                  </div>
                  {savedResumes.map(resume => (
                    <div 
                      key={resume.id} 
                      className={`flex items-center justify-between p-3 rounded-lg hover:bg-gray-100 cursor-pointer ${
                        resumeData.id === resume.id ? 'bg-blue-50 border border-blue-200' : ''
                      }`}
                      onClick={() => {
                        onLoadResume(resume)
                        onToggleResumesDropdown()
                      }}
                    >
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 truncate">{resume.name}</p>
                        <p className="text-xs text-gray-500">{resume.template} template</p>
                      </div>
                      <div className="flex items-center gap-1 ml-2">
                        {resumeData.id === resume.id && (
                          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                            Active
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Template Builder (Admin) */}
        {(isSuperAdmin || isAdmin) && (
          <Tooltip content="Template Builder" position="bottom">
            <button
              onClick={onToggleTemplateBuilder}
              className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
              style={getButtonStyle(showTemplateBuilder)}
            >
              <Settings className="w-5 h-5" />
            </button>
          </Tooltip>
        )}

        {/* Edit */}
        <Tooltip content="Edit Mode" position="bottom">
          <button
            onClick={() => {
              onTogglePreview(false)
              onTogglePDFPreview(false)
            }}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(!showPreview && !showPDFPreview && activeTool !== 'templates')}
          >
            <Edit3 className="w-5 h-5" />
          </button>
        </Tooltip>

        {/* Preview */}
        <Tooltip content="Preview Mode" position="bottom">
          <button
            onClick={() => onTogglePreview(true)}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(showPreview && !showPDFPreview)}
          >
            <Eye className="w-5 h-5" />
          </button>
        </Tooltip>

        {/* PDF Preview */}
        <Tooltip content="PDF Preview" position="bottom">
          <button
            onClick={() => onTogglePDFPreview(true)}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(showPDFPreview)}
          >
            <FileDown className="w-5 h-5" />
          </button>
        </Tooltip>

        {/* Reset */}
        <Tooltip content="Reset Section Order" position="bottom">
          <button
            onClick={onResetSectionOrder}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(false)}
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </Tooltip>
      </div>

      {/* Import Progress */}
      {isImportingPDF && (
        <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg text-blue-700">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm">{importProgress}</span>
        </div>
      )}

      {/* Template Selector Dropdown */}
      {showTemplates && (
        <div className="absolute top-full right-0 mt-2 p-4 bg-white rounded-xl shadow-lg border border-gray-200 z-50 w-80">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-medium">Choose a Template</h3>
            <button onClick={onCloseTemplates} className="p-1 hover:bg-gray-200 rounded">
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {TEMPLATES.map(template => (
              <button
                key={template.id}
                onClick={() => onSelectTemplate(template.id)}
                className={`p-3 rounded-lg border-2 transition-all ${
                  resumeData.template === template.id ? 'border-gray-800 bg-gray-50' : 'border-gray-200'
                }`}
              >
                <span className="font-medium text-sm">{template.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Template Builder Modal */}
      {showTemplateBuilder && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b">
              <h2 className="text-xl font-semibold">Template Builder</h2>
              <button onClick={onCloseTemplateBuilder} className="p-1 hover:bg-gray-100 rounded">
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
              <TemplateBuilder onClose={onCloseTemplateBuilder} />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
