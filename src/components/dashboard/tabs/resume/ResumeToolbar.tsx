'use client'

import { useRef } from 'react'
import { 
  Upload, Layout, Save, FileText, Settings, Edit3, Eye, RefreshCw, X
} from 'lucide-react'
import { ResumeData } from './types'
import { TemplateBuilder } from './TemplateBuilder'

interface ResumeToolbarProps {
  resumeData: ResumeData
  savedResumes: ResumeData[]
  showPreview: boolean
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
  onImportPDF: () => void
  onToggleTemplates: () => void
  onTogglePreview: (preview: boolean) => void
  onToggleTemplateBuilder: () => void
  onResetSectionOrder: () => void
  onSave: () => void
  onUpdateResume: () => void
  onLoadResume: (resume: ResumeData) => void
  onSelectTemplate: (templateId: string) => void
  onCloseTemplates: () => void
  onCloseTemplateBuilder: () => void
}

export function ResumeToolbar({
  resumeData,
  savedResumes,
  showPreview,
  showTemplates,
  showTemplateBuilder,
  activeTool,
  isSuperAdmin,
  isAdmin,
  isImportingPDF,
  importProgress,
  selectedTemplate,
  isSaving,
  hasResumeContent,
  onImportPDF,
  onToggleTemplates,
  onTogglePreview,
  onToggleTemplateBuilder,
  onResetSectionOrder,
  onSave,
  onUpdateResume,
  onLoadResume,
  onSelectTemplate,
  onCloseTemplates,
  onCloseTemplateBuilder,
}: ResumeToolbarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

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
      <div className="flex flex-wrap items-center gap-2">
        {/* Import PDF */}
        <button
          onClick={onImportPDF}
          disabled={isImportingPDF}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(false)}
          title={isImportingPDF ? 'Importing...' : 'Import from PDF'}
        >
          {isImportingPDF ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Upload className="w-5 h-5" />
          )}
        </button>

        {/* Template Selector */}
        <button
          onClick={onToggleTemplates}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(showTemplates)}
          title={`Template: ${selectedTemplate.name}`}
        >
          <Layout className="w-5 h-5" />
        </button>

        {/* Save */}
        <button
          onClick={resumeData.id ? onUpdateResume : onSave}
          disabled={isSaving || !hasResumeContent()}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(false)}
          title={resumeData.id ? 'Update Resume' : 'Save Resume'}
        >
          <Save className="w-5 h-5" />
        </button>

        {/* My Resumes Dropdown */}
        {savedResumes.length > 0 && (
          <div className="relative">
            <button
              className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
              style={getButtonStyle(false)}
              title={`My Resumes (${savedResumes.length})`}
            >
              <FileText className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Template Builder (Admin) */}
        {(isSuperAdmin || isAdmin) && (
          <button
            onClick={onToggleTemplateBuilder}
            className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
            style={getButtonStyle(showTemplateBuilder)}
            title="Template Builder"
          >
            <Settings className="w-5 h-5" />
          </button>
        )}

        {/* Edit */}
        <button
          onClick={() => onTogglePreview(false)}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(!showPreview && activeTool !== 'templates')}
          title="Edit Mode"
        >
          <Edit3 className="w-5 h-5" />
        </button>

        {/* Preview */}
        <button
          onClick={() => onTogglePreview(true)}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(showPreview)}
          title="Preview Mode"
        >
          <Eye className="w-5 h-5" />
        </button>

        {/* Reset */}
        <button
          onClick={onResetSectionOrder}
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-colors"
          style={getButtonStyle(false)}
          title="Reset Section Order"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
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
