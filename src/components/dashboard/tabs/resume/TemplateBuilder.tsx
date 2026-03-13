'use client'

import { useState, useEffect } from 'react'
import { Card, Button } from '../../ui'
import { 
  Palette, Save, Trash2, Plus, Eye, Edit3, 
  Check, X, ChevronDown, ChevronUp, Layout
} from 'lucide-react'
import { useAuth } from '@/context/AuthContext'

interface TemplateData {
  id?: number
  name: string
  description: string
  headerBg: string
  headerText: string
  accentColor: string
  primaryColor: string
  fontFamily: string
  layout: string
  sectionOrder?: string[]
  badgeStyle?: 'rounded' | 'square' | 'pill'
}

interface ResumeTemplate {
  id: number
  name: string
  description: string
  template_data: string
  is_active: number
  created_at: string
}

const DEFAULT_SECTIONS = [
  'summary',
  'experience', 
  'education',
  'skills',
  'projects',
  'certifications',
  'languages',
  'websites'
]

const COLOR_PRESETS = [
  { name: 'Purple Modern', headerBg: '#9333ea', accentColor: '#9333ea', primaryColor: '#808080' },
  { name: 'Blue Minimal', headerBg: '#3b82f6', accentColor: '#3b82f6', primaryColor: '#3c3c3c' },
  { name: 'Orange Creative', headerBg: '#ea580c', accentColor: '#ea580c', primaryColor: '#dc2626' },
  { name: 'Green Nature', headerBg: '#059669', accentColor: '#059669', primaryColor: '#065f46' },
  { name: 'Pink Fashion', headerBg: '#db2777', accentColor: '#db2777', primaryColor: '#831843' },
  { name: 'Teal Ocean', headerBg: '#0d9488', accentColor: '#0d9488', primaryColor: '#115e59' },
  { name: 'Indigo Royal', headerBg: '#4f46e5', accentColor: '#4f46e5', primaryColor: '#3730a3' },
  { name: 'Red Passion', headerBg: '#dc2626', accentColor: '#dc2626', primaryColor: '#991b1b' },
]

const FONT_OPTIONS = [
  { value: 'helvetica', label: 'Helvetica / Arial' },
  { value: 'times', label: 'Times New Roman' },
  { value: 'courier', label: 'Courier New' },
  { value: 'georgia', label: 'Georgia' },
  { value: 'verdana', label: 'Verdana' }
]

const LAYOUT_OPTIONS = [
  { value: 'modern', label: 'Modern' },
  { value: 'classic', label: 'Classic' },
  { value: 'minimal', label: 'Minimal' },
  { value: 'creative', label: 'Creative' }
]

export function TemplateBuilder({ onClose }: { onClose?: () => void }) {
  const { user, isAdmin, isSuperAdmin } = useAuth()
  const [templates, setTemplates] = useState<ResumeTemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<TemplateData | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [previewData, setPreviewData] = useState<TemplateData | null>(null)

  // Load templates on mount
  useEffect(() => {
    loadTemplates()
  }, [])

  const loadTemplates = async () => {
    try {
      const response = await fetch('/api/resume-templates')
      if (response.ok) {
        const data = await response.json()
        setTemplates(data)
      }
    } catch (error) {
      console.error('Error loading templates:', error)
    }
  }

  const handleCreateNew = () => {
    setSelectedTemplate({
      name: '',
      description: '',
      headerBg: '#9333ea',
      headerText: '#ffffff',
      accentColor: '#9333ea',
      primaryColor: '#808080',
      fontFamily: 'helvetica',
      layout: 'modern',
      sectionOrder: DEFAULT_SECTIONS,
      badgeStyle: 'rounded'
    })
    setIsEditing(true)
  }

  const handleEdit = (template: ResumeTemplate) => {
    const parsedData = JSON.parse(template.template_data)
    setSelectedTemplate({
      id: template.id,
      name: template.name,
      description: template.description || '',
      ...parsedData
    })
    setIsEditing(true)
  }

  const handleSave = async () => {
    if (!selectedTemplate) return
    
    if (!selectedTemplate.name) {
      alert('Please enter a template name')
      return
    }

    setIsSaving(true)
    try {
      const payload = {
        id: selectedTemplate.id,
        name: selectedTemplate.name,
        description: selectedTemplate.description,
        templateData: {
          headerBg: selectedTemplate.headerBg,
          headerText: selectedTemplate.headerText,
          accentColor: selectedTemplate.accentColor,
          primaryColor: selectedTemplate.primaryColor,
          fontFamily: selectedTemplate.fontFamily,
          layout: selectedTemplate.layout,
          sectionOrder: selectedTemplate.sectionOrder,
          badgeStyle: selectedTemplate.badgeStyle
        },
        userId: user?.id
      }

      const response = await fetch('/api/resume-templates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        await loadTemplates()
        setIsEditing(false)
        setSelectedTemplate(null)
        alert(selectedTemplate.id ? 'Template updated successfully!' : 'Template created successfully!')
      } else {
        const error = await response.json()
        alert('Error: ' + error.error)
      }
    } catch (error) {
      console.error('Error saving template:', error)
      alert('Failed to save template')
    } finally {
      setIsSaving(false)
    }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this template?')) return

    try {
      const response = await fetch(`/api/resume-templates?id=${id}`, { method: 'DELETE' })
      if (response.ok) {
        await loadTemplates()
        alert('Template deleted successfully!')
      }
    } catch (error) {
      console.error('Error deleting template:', error)
      alert('Failed to delete template')
    }
  }

  const applyColorPreset = (preset: typeof COLOR_PRESETS[0]) => {
    if (!selectedTemplate) return
    setSelectedTemplate({
      ...selectedTemplate,
      headerBg: preset.headerBg,
      accentColor: preset.accentColor,
      primaryColor: preset.primaryColor
    })
  }

  // Only show for admins
  if (!isAdmin && !isSuperAdmin) {
    return (
      <Card title="Resume Templates">
        <div className="text-center py-8 text-gray-500">
          <Palette className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>You don't have permission to manage templates.</p>
          <p className="text-sm mt-2">Contact an administrator for access.</p>
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card title="Resume Template Builder">
        {/* Header */}
        <div className="flex flex-wrap gap-3 mb-6">
          <Button
            onClick={handleCreateNew}
            className="flex items-center gap-2"
            style={{ backgroundColor: '#212529', color: '#ffffff' }}
          >
            <Plus className="w-4 h-4" />
            Create New Template
          </Button>
          
          {selectedTemplate && (
            <Button
              onClick={() => setShowPreview(!showPreview)}
              variant="secondary"
              className="flex items-center gap-2"
            >
              <Eye className="w-4 h-4" />
              {showPreview ? 'Hide Preview' : 'Preview'}
            </Button>
          )}
        </div>

        {/* Template List */}
        {!isEditing && !selectedTemplate && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {templates.map(template => (
              <div
                key={template.id}
                className="border rounded-xl p-4 hover:shadow-lg transition-shadow"
              >
                <div 
                  className="h-16 rounded-lg mb-3"
                  style={{ backgroundColor: JSON.parse(template.template_data).headerBg }}
                />
                <h3 className="font-semibold text-gray-900">{template.name}</h3>
                <p className="text-sm text-gray-500 mt-1">{template.description}</p>
                <div className="flex gap-2 mt-4">
                  <button
                    onClick={() => handleEdit(template)}
                    className="flex-1 flex items-center justify-center gap-1 px-3 py-2 bg-gray-100 rounded-lg text-sm hover:bg-gray-200"
                  >
                    <Edit3 className="w-4 h-4" />
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(template.id)}
                    className="flex items-center justify-center px-3 py-2 text-red-500 hover:bg-red-50 rounded-lg"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
            
            {templates.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-500">
                <Layout className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No templates yet. Create your first template!</p>
              </div>
            )}
          </div>
        )}

        {/* Editor */}
        {isEditing && selectedTemplate && (
          <div className="space-y-6">
            {/* Template Name */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Template Name *
                </label>
                <input
                  type="text"
                  value={selectedTemplate.name}
                  onChange={(e) => setSelectedTemplate({ ...selectedTemplate, name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500"
                  placeholder="e.g., Modern Purple"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  value={selectedTemplate.description}
                  onChange={(e) => setSelectedTemplate({ ...selectedTemplate, description: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500"
                  placeholder="Brief description of the template"
                />
              </div>
            </div>

            {/* Color Presets */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Quick Color Presets
              </label>
              <div className="flex flex-wrap gap-2">
                {COLOR_PRESETS.map((preset, index) => (
                  <button
                    key={index}
                    onClick={() => applyColorPreset(preset)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg border hover:shadow-md transition-shadow"
                  >
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: preset.headerBg }}
                    />
                    <span className="text-sm">{preset.name}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Color Customization */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Header Background
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={selectedTemplate.headerBg}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, headerBg: e.target.value })}
                    className="w-10 h-10 rounded border cursor-pointer"
                  />
                  <input
                    type="text"
                    value={selectedTemplate.headerBg}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, headerBg: e.target.value })}
                    className="flex-1 px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Header Text
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={selectedTemplate.headerText}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, headerText: e.target.value })}
                    className="w-10 h-10 rounded border cursor-pointer"
                  />
                  <input
                    type="text"
                    value={selectedTemplate.headerText}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, headerText: e.target.value })}
                    className="flex-1 px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Accent Color
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={selectedTemplate.accentColor}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, accentColor: e.target.value })}
                    className="w-10 h-10 rounded border cursor-pointer"
                  />
                  <input
                    type="text"
                    value={selectedTemplate.accentColor}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, accentColor: e.target.value })}
                    className="flex-1 px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Primary Color
                </label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={selectedTemplate.primaryColor}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, primaryColor: e.target.value })}
                    className="w-10 h-10 rounded border cursor-pointer"
                  />
                  <input
                    type="text"
                    value={selectedTemplate.primaryColor}
                    onChange={(e) => setSelectedTemplate({ ...selectedTemplate, primaryColor: e.target.value })}
                    className="flex-1 px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
              </div>
            </div>

            {/* Font and Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Font Family
                </label>
                <select
                  value={selectedTemplate.fontFamily}
                  onChange={(e) => setSelectedTemplate({ ...selectedTemplate, fontFamily: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500"
                >
                  {FONT_OPTIONS.map(font => (
                    <option key={font.value} value={font.value}>
                      {font.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Layout Style
                </label>
                <select
                  value={selectedTemplate.layout}
                  onChange={(e) => setSelectedTemplate({ ...selectedTemplate, layout: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500"
                >
                  {LAYOUT_OPTIONS.map(layout => (
                    <option key={layout.value} value={layout.value}>
                      {layout.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Badge Style */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Badge Style
              </label>
              <div className="flex gap-4">
                {(['rounded', 'square', 'pill'] as const).map(style => (
                  <label key={style} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="badgeStyle"
                      value={style}
                      checked={selectedTemplate.badgeStyle === style}
                      onChange={() => setSelectedTemplate({ ...selectedTemplate, badgeStyle: style })}
                      className="w-4 h-4"
                    />
                    <span className="capitalize">{style}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Preview Section */}
            {showPreview && (
              <div className="border rounded-xl p-6 bg-gray-50">
                <h4 className="font-medium mb-4">Live Preview</h4>
                <div 
                  className="bg-white rounded-lg shadow-lg p-6 max-w-lg mx-auto"
                  style={{ fontFamily: selectedTemplate.fontFamily }}
                >
                  {/* Header Preview */}
                  <div 
                    className="text-center pb-4 mb-4"
                    style={{ backgroundColor: selectedTemplate.headerBg, margin: '-1.5rem -1.5rem 1.5rem -1.5rem', padding: '1.5rem' }}
                  >
                    <h2 
                      className="text-2xl font-bold"
                      style={{ color: selectedTemplate.headerText }}
                    >
                      John Doe
                    </h2>
                    <p style={{ color: selectedTemplate.headerText, opacity: 0.8 }}>
                      john@example.com | Software Engineer
                    </p>
                  </div>
                  
                  {/* Section Preview */}
                  <div className="space-y-4">
                    <div>
                      <h3 
                        className="font-bold border-b pb-1 mb-2"
                        style={{ 
                          color: selectedTemplate.primaryColor,
                          borderColor: selectedTemplate.primaryColor
                        }}
                      >
                        Skills
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {['JavaScript', 'React', 'Node.js'].map(skill => (
                          <span
                            key={skill}
                            className="px-3 py-1 text-sm text-white"
                            style={{ 
                              backgroundColor: selectedTemplate.accentColor,
                              borderRadius: selectedTemplate.badgeStyle === 'rounded' ? '0.5rem' : 
                                selectedTemplate.badgeStyle === 'pill' ? '9999px' : '0.25rem'
                            }}
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3 pt-4 border-t">
              <Button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center gap-2"
                style={{ backgroundColor: '#212529', color: '#ffffff' }}
              >
                <Save className="w-4 h-4" />
                {isSaving ? 'Saving...' : 'Save Template'}
              </Button>
              <Button
                onClick={() => { 
                  setIsEditing(false); 
                  setSelectedTemplate(null); 
                  if (onClose) onClose();
                }}
                variant="secondary"
                className="flex items-center gap-2"
              >
                <X className="w-4 h-4" />
                Cancel
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
