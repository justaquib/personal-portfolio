'use client'

import { Card, Button } from '../ui'
import { useState, useRef, useEffect } from 'react'
import { jsPDF } from 'jspdf'
import { useAuth } from '@/context/AuthContext'
import { 
  User, Mail, Phone, MapPin, Linkedin, Globe, 
  Plus, Trash2, Download, FileText, Briefcase, 
  GraduationCap, Code, Folder, ChevronDown, ChevronUp,
  Save, Edit3, Eye, Sparkles, Layout, Trash, RefreshCw,
  Check, Copy, X
} from 'lucide-react'

// Types
interface Experience {
  id: string
  company: string
  role: string
  startDate: string
  endDate: string
  current: boolean
  description: string
}

interface Education {
  id: string
  institution: string
  degree: string
  field: string
  graduationDate: string
}

interface Project {
  id: string
  name: string
  description: string
  technologies: string
}

interface ResumeData {
  id?: number
  name: string
  template: string
  personalInfo: {
    name: string
    email: string
    phone: string
    location: string
    linkedin: string
    portfolio: string
  }
  summary: string
  experience: Experience[]
  education: Education[]
  skills: string
  projects: Project[]
  isDefault?: boolean
}

// Template definitions
const TEMPLATES = [
  {
    id: 'modern',
    name: 'Modern',
    description: 'Clean and professional with a contemporary look',
    preview: 'bg-gradient-to-r from-purple-600 to-pink-600'
  },
  {
    id: 'classic',
    name: 'Classic',
    description: 'Traditional resume format, perfect for corporate jobs',
    preview: 'bg-gradient-to-r from-gray-600 to-gray-800'
  },
  {
    id: 'minimal',
    name: 'Minimal',
    description: 'Simple and elegant with ample white space',
    preview: 'bg-gradient-to-r from-blue-500 to-cyan-500'
  },
  {
    id: 'creative',
    name: 'Creative',
    description: 'Stand out with a unique and memorable design',
    preview: 'bg-gradient-to-r from-orange-500 to-red-500'
  }
]

// Default resume data
const defaultResumeData: ResumeData = {
  name: 'My Resume',
  template: 'modern',
  personalInfo: {
    name: '',
    email: '',
    phone: '',
    location: '',
    linkedin: '',
    portfolio: ''
  },
  summary: '',
  experience: [],
  education: [],
  skills: '',
  projects: []
}

export function ResumeBuilderTab() {
  const { user } = useAuth()
  const [resumeData, setResumeData] = useState<ResumeData>(defaultResumeData)
  const [savedResumes, setSavedResumes] = useState<ResumeData[]>([])
  const [activeSection, setActiveSection] = useState<string | null>('personal')
  const [isGenerating, setIsGenerating] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [resumeName, setResumeName] = useState('My Resume')
  const [isSaving, setIsSaving] = useState(false)
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([])
  const resumeRef = useRef<HTMLDivElement>(null)

  // Load saved resumes on mount
  useEffect(() => {
    if (user?.id) {
      loadResumes()
    }
  }, [user?.id])

  const loadResumes = async () => {
    try {
      const response = await fetch(`/api/resumes?userId=${user?.id}`)
      if (response.ok) {
        const data = await response.json()
        const parsed = data.map((r: any) => ({
          ...r,
          personalInfo: typeof r.personal_info === 'string' ? JSON.parse(r.personal_info) : r.personal_info,
          experience: typeof r.experience === 'string' ? JSON.parse(r.experience) : r.experience,
          education: typeof r.education === 'string' ? JSON.parse(r.education) : r.education,
          projects: typeof r.projects === 'string' ? JSON.parse(r.projects) : r.projects,
          isDefault: r.is_default === 1
        }))
        setSavedResumes(parsed)
        
        // Load default resume if exists
        const defaultResume = parsed.find((r: ResumeData) => r.isDefault)
        if (defaultResume) {
          setResumeData(defaultResume)
          setResumeName(defaultResume.name)
        }
      }
    } catch (error) {
      console.error('Error loading resumes:', error)
    }
  }

  const updatePersonalInfo = (field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      personalInfo: { ...prev.personalInfo, [field]: value }
    }))
  }

  const addExperience = () => {
    setResumeData(prev => ({
      ...prev,
      experience: [
        ...prev.experience,
        { id: Date.now().toString(), company: '', role: '', startDate: '', endDate: '', current: false, description: '' }
      ]
    }))
  }

  const updateExperience = (id: string, field: string, value: string | boolean) => {
    setResumeData(prev => ({
      ...prev,
      experience: prev.experience.map(exp => 
        exp.id === id ? { ...exp, [field]: value } : exp
      )
    }))
  }

  const removeExperience = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      experience: prev.experience.filter(exp => exp.id !== id)
    }))
  }

  const addEducation = () => {
    setResumeData(prev => ({
      ...prev,
      education: [
        ...prev.education,
        { id: Date.now().toString(), institution: '', degree: '', field: '', graduationDate: '' }
      ]
    }))
  }

  const updateEducation = (id: string, field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      education: prev.education.map(edu => 
        edu.id === id ? { ...edu, [field]: value } : edu
      )
    }))
  }

  const removeEducation = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      education: prev.education.filter(edu => edu.id !== id)
    }))
  }

  const addProject = () => {
    setResumeData(prev => ({
      ...prev,
      projects: [
        ...prev.projects,
        { id: Date.now().toString(), name: '', description: '', technologies: '' }
      ]
    }))
  }

  const updateProject = (id: string, field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      projects: prev.projects.map(proj => 
        proj.id === id ? { ...proj, [field]: value } : proj
      )
    }))
  }

  const removeProject = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      projects: prev.projects.filter(proj => proj.id !== id)
    }))
  }

  const selectTemplate = (templateId: string) => {
    setResumeData(prev => ({ ...prev, template: templateId }))
    setShowTemplates(false)
  }

  // Generate AI summary
  const generateAISummary = async () => {
    setIsLoadingAI(true)
    try {
      const response = await fetch('/api/ai-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experience: resumeData.experience,
          education: resumeData.education,
          skills: resumeData.skills,
          jobTitle: resumeData.experience[0]?.role,
          industry: 'technology'
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.summary) {
          setResumeData(prev => ({ ...prev, summary: data.summary }))
          setAiSuggestions([data.summary])
        }
      }
    } catch (error) {
      console.error('Error generating AI summary:', error)
    } finally {
      setIsLoadingAI(false)
    }
  }

  // Save resume to database
  const saveResume = async () => {
    if (!user?.id) {
      alert('Please sign in to save your resume')
      return
    }

    setIsSaving(true)
    try {
      const payload = {
        id: resumeData.id,
        userId: user.id,
        name: resumeName,
        template: resumeData.template,
        personalInfo: resumeData.personalInfo,
        summary: resumeData.summary,
        experience: resumeData.experience,
        education: resumeData.education,
        skills: resumeData.skills,
        projects: resumeData.projects,
        isDefault: savedResumes.length === 0
      }

      const response = await fetch('/api/resumes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        const saved = await response.json()
        setResumeData({ ...resumeData, id: saved.id, name: resumeName })
        await loadResumes()
        setShowSaveModal(false)
        alert('Resume saved successfully!')
      }
    } catch (error) {
      console.error('Error saving resume:', error)
      alert('Failed to save resume')
    } finally {
      setIsSaving(false)
    }
  }

  // Load selected resume
  const loadResume = (resume: ResumeData) => {
    setResumeData(resume)
    setResumeName(resume.name)
    setShowTemplates(false)
  }

  // Delete resume
  const deleteResume = async (id: number) => {
    if (!confirm('Are you sure you want to delete this resume?')) return

    try {
      const response = await fetch(`/api/resumes?id=${id}`, { method: 'DELETE' })
      if (response.ok) {
        await loadResumes()
        if (resumeData.id === id) {
          setResumeData(defaultResumeData)
          setResumeName('My Resume')
        }
      }
    } catch (error) {
      console.error('Error deleting resume:', error)
    }
  }

  // Generate PDF with template
  const generatePDF = () => {
    setIsGenerating(true)
    try {
      const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      })

      const pageWidth = doc.internal.pageSize.getWidth()
      const pageHeight = doc.internal.pageSize.getHeight()
      const margin = 15
      const maxWidth = pageWidth - margin * 2
      let y = margin

      // Template-specific colors
      const colors: Record<string, [number, number, number]> = {
        modern: [80, 80, 80],
        classic: [40, 40, 40],
        minimal: [60, 60, 60],
        creative: [220, 80, 80]
      }
      
      const accentColors: Record<string, [number, number, number]> = {
        modern: [147, 51, 234], // Purple
        classic: [80, 80, 80], // Gray
        minimal: [59, 130, 246], // Blue
        creative: [234, 88, 12] // Orange
      }

      const primaryColor = colors[resumeData.template] || colors.modern
      const accentColor = accentColors[resumeData.template] || accentColors.modern

      // Template-specific header
      if (resumeData.template === 'modern') {
        // Modern: Gradient-like header
        doc.setFillColor(...accentColor)
        doc.rect(0, 0, pageWidth, 35, 'F')
        doc.setTextColor(255, 255, 255)
        doc.setFontSize(24)
        doc.setFont('helvetica', 'bold')
        doc.text(resumeData.personalInfo.name || 'Your Name', margin, 22)
        y = 45
      } else if (resumeData.template === 'classic') {
        // Classic: Centered header with underline
        doc.setFontSize(22)
        doc.setFont('times', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text(resumeData.personalInfo.name || 'Your Name', pageWidth / 2, y, { align: 'center' })
        y += 2
        doc.setDrawColor(...primaryColor)
        doc.setLineWidth(0.5)
        doc.line(margin, y, pageWidth - margin, y)
        y += 10
      } else if (resumeData.template === 'minimal') {
        // Minimal: Large name, minimal decoration
        doc.setFontSize(28)
        doc.setFont('helvetica', 'light')
        doc.setTextColor(...primaryColor)
        doc.text(resumeData.personalInfo.name || 'Your Name', margin, y + 5)
        y += 18
      } else if (resumeData.template === 'creative') {
        // Creative: Bold colored name
        doc.setFontSize(26)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...accentColor)
        doc.text(resumeData.personalInfo.name || 'Your Name', margin, y + 5)
        y += 15
      }

      // Contact info
      doc.setFontSize(9)
      doc.setFont('helvetica', 'normal')
      
      if (resumeData.template === 'modern') {
        doc.setTextColor(255, 255, 255)
      } else {
        doc.setTextColor(100, 100, 100)
      }
      
      const contactParts = []
      if (resumeData.personalInfo.email) contactParts.push(resumeData.personalInfo.email)
      if (resumeData.personalInfo.phone) contactParts.push(resumeData.personalInfo.phone)
      if (resumeData.personalInfo.location) contactParts.push(resumeData.personalInfo.location)
      
      if (contactParts.length > 0) {
        doc.text(contactParts.join(' | '), margin, y)
        y += 4
      }

      if (resumeData.personalInfo.linkedin || resumeData.personalInfo.portfolio) {
        const links = []
        if (resumeData.personalInfo.linkedin) links.push(`LinkedIn: ${resumeData.personalInfo.linkedin}`)
        if (resumeData.personalInfo.portfolio) links.push(`Portfolio: ${resumeData.personalInfo.portfolio}`)
        doc.text(links.join(' | '), margin, y)
        y += 8
      }

      // Divider
      if (resumeData.template !== 'modern') {
        doc.setDrawColor(200, 200, 200)
        doc.line(margin, y, pageWidth - margin, y)
        y += 8
      } else {
        y += 5
      }

      // Summary
      if (resumeData.summary) {
        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('PROFESSIONAL SUMMARY', margin, y)
        y += 5
        
        doc.setFontSize(10)
        doc.setFont('helvetica', 'normal')
        doc.setTextColor(80, 80, 80)
        const summaryLines = doc.splitTextToSize(resumeData.summary, maxWidth)
        doc.text(summaryLines, margin, y)
        y += summaryLines.length * 5 + 8
      }

      // Experience
      if (resumeData.experience.length > 0) {
        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('WORK EXPERIENCE', margin, y)
        y += 5

        resumeData.experience.forEach(exp => {
          if (y > pageHeight - 40) {
            doc.addPage()
            y = margin
          }

          doc.setFontSize(10)
          doc.setFont('helvetica', 'bold')
          doc.setTextColor(...primaryColor)
          doc.text(exp.role || 'Job Title', margin, y)
          
          const dateText = exp.current ? `${exp.startDate} - Present` : `${exp.startDate} - ${exp.endDate}`
          const dateWidth = doc.getTextWidth(dateText)
          doc.text(dateText, pageWidth - margin - dateWidth, y)
          y += 4

          doc.setFontSize(9)
          doc.setFont('helvetica', 'italic')
          doc.setTextColor(100, 100, 100)
          doc.text(exp.company || 'Company Name', margin, y)
          y += 4

          doc.setFont('helvetica', 'normal')
          const descLines = doc.splitTextToSize(exp.description, maxWidth)
          doc.text(descLines, margin, y)
          y += descLines.length * 4 + 6
        })
      }

      // Education
      if (resumeData.education.length > 0) {
        if (y > pageHeight - 40) {
          doc.addPage()
          y = margin
        }

        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('EDUCATION', margin, y)
        y += 5

        resumeData.education.forEach(edu => {
          doc.setFontSize(10)
          doc.setFont('helvetica', 'bold')
          doc.setTextColor(...primaryColor)
          doc.text(`${edu.degree} in ${edu.field}`, margin, y)
          
          const dateWidth = doc.getTextWidth(edu.graduationDate)
          doc.text(edu.graduationDate, pageWidth - margin - dateWidth, y)
          y += 4

          doc.setFontSize(9)
          doc.setFont('helvetica', 'italic')
          doc.setTextColor(100, 100, 100)
          doc.text(edu.institution, margin, y)
          y += 6
        })
      }

      // Skills
      if (resumeData.skills) {
        if (y > pageHeight - 40) {
          doc.addPage()
          y = margin
        }

        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('SKILLS', margin, y)
        y += 5

        doc.setFontSize(9)
        doc.setFont('helvetica', 'normal')
        doc.setTextColor(80, 80, 80)
        const skillsLines = doc.splitTextToSize(resumeData.skills, maxWidth)
        doc.text(skillsLines, margin, y)
        y += skillsLines.length * 4 + 6
      }

      // Projects
      if (resumeData.projects.length > 0) {
        if (y > pageHeight - 40) {
          doc.addPage()
          y = margin
        }

        doc.setFontSize(11)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('PROJECTS', margin, y)
        y += 5

        resumeData.projects.forEach(proj => {
          doc.setFontSize(10)
          doc.setFont('helvetica', 'bold')
          doc.setTextColor(...primaryColor)
          doc.text(proj.name, margin, y)
          y += 4

          doc.setFontSize(9)
          doc.setFont('helvetica', 'normal')
          doc.setTextColor(80, 80, 80)
          const descLines = doc.splitTextToSize(proj.description, maxWidth)
          doc.text(descLines, margin, y)
          y += descLines.length * 4 + 2

          if (proj.technologies) {
            doc.setFont('helvetica', 'italic')
            doc.text(`Technologies: ${proj.technologies}`, margin, y)
            y += 6
          }
        })
      }

      // Save
      const fileName = resumeData.personalInfo.name 
        ? `${resumeData.personalInfo.name.replace(/\s+/g, '_')}_Resume.pdf`
        : 'Resume.pdf'
      doc.save(fileName)
    } catch (error) {
      console.error('Error generating PDF:', error)
      alert('Failed to generate PDF. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  // Generate DOCX using HTML
  const generateDOCX = () => {
    setIsGenerating(true)
    try {
      const templateStyles: Record<string, string> = {
        modern: 'color: #9333ea;',
        classic: 'color: #333; font-family: Times New Roman;',
        minimal: 'color: #374151;',
        creative: 'color: #ea580c;'
      }

      const htmlContent = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
    h1 { ${templateStyles[resumeData.template] || templateStyles.modern } border-bottom: 2px solid #333; padding-bottom: 10px; }
    h2 { color: #555; margin-top: 30px; }
    .contact { color: #666; margin-bottom: 20px; }
    .section { margin-bottom: 25px; }
    .job-title { font-weight: bold; }
    .company { font-style: italic; color: #666; }
    .date { float: right; }
  </style>
</head>
<body>
  <h1>${resumeData.personalInfo.name || 'Your Name'}</h1>
  <div class="contact">
    ${resumeData.personalInfo.email} | ${resumeData.personalInfo.phone} | ${resumeData.personalInfo.location}
    ${resumeData.personalInfo.linkedin ? `<br>LinkedIn: ${resumeData.personalInfo.linkedin}` : ''}
    ${resumeData.personalInfo.portfolio ? `<br>Portfolio: ${resumeData.personalInfo.portfolio}` : ''}
  </div>
  
  ${resumeData.summary ? `<div class="section"><h2>PROFESSIONAL SUMMARY</h2><p>${resumeData.summary}</p></div>` : ''}
  
  ${resumeData.experience.length > 0 ? `
  <div class="section">
    <h2>WORK EXPERIENCE</h2>
    ${resumeData.experience.map(exp => `
      <div>
        <span class="job-title">${exp.role}</span>
        <span class="date">${exp.current ? exp.startDate + ' - Present' : exp.startDate + ' - ' + exp.endDate}</span>
        <div class="company">${exp.company}</div>
        <p>${exp.description}</p>
      </div>
    `).join('')}
  </div>
  ` : ''}
  
  ${resumeData.education.length > 0 ? `
  <div class="section">
    <h2>EDUCATION</h2>
    ${resumeData.education.map(edu => `
      <div>
        <span class="job-title">${edu.degree} in ${edu.field}</span>
        <span class="date">${edu.graduationDate}</span>
        <div class="company">${edu.institution}</div>
      </div>
    `).join('')}
  </div>
  ` : ''}
  
  ${resumeData.skills ? `
  <div class="section">
    <h2>SKILLS</h2>
    <p>${resumeData.skills}</p>
  </div>
  ` : ''}
  
  ${resumeData.projects.length > 0 ? `
  <div class="section">
    <h2>PROJECTS</h2>
    ${resumeData.projects.map(proj => `
      <div>
        <span class="job-title">${proj.name}</span>
        ${proj.technologies ? `<div><em>Technologies: ${proj.technologies}</em></div>` : ''}
        <p>${proj.description}</p>
      </div>
    `).join('')}
  </div>
  ` : ''}
</body>
</html>`

      const blob = new Blob([htmlContent], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      const fileName = resumeData.personalInfo.name 
        ? `${resumeData.personalInfo.name.replace(/\s+/g, '_')}_Resume.html`
        : 'Resume.html'
      a.download = fileName.replace('.html', '.doc')
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      alert('DOCX file downloaded! Note: The file is saved as .doc which can be opened in Microsoft Word.')
    } catch (error) {
      console.error('Error generating DOCX:', error)
      alert('Failed to generate DOCX. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  const toggleSection = (section: string) => {
    setActiveSection(activeSection === section ? null : section)
  }

  const isResumeComplete = () => {
    return resumeData.personalInfo.name && 
           (resumeData.experience.length > 0 || resumeData.education.length > 0 || resumeData.skills)
  }

  const selectedTemplate = TEMPLATES.find(t => t.id === resumeData.template) || TEMPLATES[0]

  return (
    <div className="space-y-6">
      <Card title="Resume Builder">
        {/* Header with template selector and saved resumes */}
        <div className="flex flex-wrap gap-3 mb-6">
          <button
            onClick={() => setShowTemplates(!showTemplates)}
            className="flex items-center gap-2 px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
          >
            <Layout className="w-4 h-4" />
            {selectedTemplate.name} Template
          </button>
          
          <button
            onClick={() => setShowSaveModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition-colors"
          >
            <Save className="w-4 h-4" />
            Save
          </button>

          {savedResumes.length > 0 && (
            <div className="relative">
              <button className="flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors">
                <FileText className="w-4 h-4" />
                My Resumes ({savedResumes.length})
              </button>
            </div>
          )}

          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setShowPreview(false)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                !showPreview ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Edit3 className="w-4 h-4" />
              Edit
            </button>
            <button
              onClick={() => setShowPreview(true)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                showPreview ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <Eye className="w-4 h-4" />
              Preview
            </button>
          </div>
        </div>

        {/* Template Selector Dropdown */}
        {showTemplates && (
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <h3 className="font-medium mb-3">Choose a Template</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {TEMPLATES.map(template => (
                <button
                  key={template.id}
                  onClick={() => selectTemplate(template.id)}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    resumeData.template === template.id 
                      ? 'border-purple-500 bg-white shadow-md' 
                      : 'border-gray-200 bg-white hover:border-purple-300'
                  }`}
                >
                  <div className={`h-8 rounded-lg mb-3 ${template.preview}`}></div>
                  <h4 className="font-medium text-gray-900">{template.name}</h4>
                  <p className="text-xs text-gray-500 mt-1">{template.description}</p>
                  {resumeData.template === template.id && (
                    <div className="mt-2 flex items-center gap-1 text-purple-600 text-sm">
                      <Check className="w-4 h-4" /> Selected
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Save Modal */}
        {showSaveModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-2xl p-6 w-full max-w-md mx-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Save Resume</h3>
                <button onClick={() => setShowSaveModal(false)} className="p-1 hover:bg-gray-100 rounded">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <input
                type="text"
                value={resumeName}
                onChange={(e) => setResumeName(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg mb-4"
                placeholder="Resume name"
              />
              <div className="flex gap-3">
                <Button onClick={saveResume} disabled={isSaving} className="flex-1">
                  {isSaving ? 'Saving...' : 'Save Resume'}
                </Button>
                <Button variant="secondary" onClick={() => setShowSaveModal(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        )}

        {!showPreview ? (
          <div className="space-y-4">
            {/* Personal Information Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('personal')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <User className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Personal Information</span>
                </div>
                {activeSection === 'personal' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'personal' && (
                <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
                    <input
                      type="text"
                      value={resumeData.personalInfo.name}
                      onChange={(e) => updatePersonalInfo('name', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="John Doe"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
                    <input
                      type="email"
                      value={resumeData.personalInfo.email}
                      onChange={(e) => updatePersonalInfo('email', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="john@example.com"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                    <input
                      type="tel"
                      value={resumeData.personalInfo.phone}
                      onChange={(e) => updatePersonalInfo('phone', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="+1 (555) 123-4567"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                    <input
                      type="text"
                      value={resumeData.personalInfo.location}
                      onChange={(e) => updatePersonalInfo('location', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="New York, NY"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">LinkedIn</label>
                    <input
                      type="text"
                      value={resumeData.personalInfo.linkedin}
                      onChange={(e) => updatePersonalInfo('linkedin', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="linkedin.com/in/johndoe"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Portfolio</label>
                    <input
                      type="text"
                      value={resumeData.personalInfo.portfolio}
                      onChange={(e) => updatePersonalInfo('portfolio', e.target.value)}
                      className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      placeholder="johndoe.com"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Summary Section with AI */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('summary')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Professional Summary</span>
                </div>
                {activeSection === 'summary' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'summary' && (
                <div className="p-4">
                  <div className="flex items-start gap-3 mb-3">
                    <textarea
                      value={resumeData.summary}
                      onChange={(e) => setResumeData(prev => ({ ...prev, summary: e.target.value }))}
                      className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                      rows={4}
                      placeholder="Write a brief summary of your professional background and career goals..."
                    />
                    <button
                      onClick={generateAISummary}
                      disabled={isLoadingAI || (!resumeData.experience.length && !resumeData.skills)}
                      className="flex flex-col items-center gap-1 px-3 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:opacity-90 disabled:opacity-50 transition-all"
                    >
                      {isLoadingAI ? (
                        <RefreshCw className="w-5 h-5 animate-spin" />
                      ) : (
                        <Sparkles className="w-5 h-5" />
                      )}
                      <span className="text-xs">AI Summary</span>
                    </button>
                  </div>
                  {aiSuggestions.length > 0 && (
                    <div className="mt-3 p-3 bg-purple-50 rounded-lg">
                      <p className="text-sm text-purple-700 mb-2">AI Suggestion:</p>
                      <p className="text-sm text-gray-700">{aiSuggestions[0]}</p>
                      <button
                        onClick={() => {
                          setResumeData(prev => ({ ...prev, summary: aiSuggestions[0] }))
                          setAiSuggestions([])
                        }}
                        className="mt-2 text-sm text-purple-600 hover:text-purple-700 flex items-center gap-1"
                      >
                        <Copy className="w-4 h-4" /> Use this
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Experience Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('experience')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Briefcase className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Work Experience</span>
                  <span className="text-sm text-gray-500">({resumeData.experience.length})</span>
                </div>
                {activeSection === 'experience' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'experience' && (
                <div className="p-4 space-y-4">
                  {resumeData.experience.map((exp, index) => (
                    <div key={exp.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Experience {index + 1}</span>
                        <button
                          onClick={() => removeExperience(exp.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={exp.company}
                          onChange={(e) => updateExperience(exp.id, 'company', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Company Name"
                        />
                        <input
                          type="text"
                          value={exp.role}
                          onChange={(e) => updateExperience(exp.id, 'role', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Job Title"
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <input
                          type="text"
                          value={exp.startDate}
                          onChange={(e) => updateExperience(exp.id, 'startDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Start Date (e.g., Jan 2020)"
                        />
                        <input
                          type="text"
                          value={exp.endDate}
                          onChange={(e) => updateExperience(exp.id, 'endDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="End Date (e.g., Dec 2023)"
                          disabled={exp.current}
                        />
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={exp.current}
                            onChange={(e) => updateExperience(exp.id, 'current', e.target.checked)}
                            className="w-4 h-4 text-purple-600 rounded"
                          />
                          <span className="text-sm text-gray-700">Currently working</span>
                        </label>
                      </div>
                      <textarea
                        value={exp.description}
                        onChange={(e) => updateExperience(exp.id, 'description', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        rows={3}
                        placeholder="Describe your responsibilities and achievements..."
                      />
                    </div>
                  ))}
                  <button
                    onClick={addExperience}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-purple-500 hover:text-purple-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Experience
                  </button>
                </div>
              )}
            </div>

            {/* Education Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('education')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <GraduationCap className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Education</span>
                  <span className="text-sm text-gray-500">({resumeData.education.length})</span>
                </div>
                {activeSection === 'education' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'education' && (
                <div className="p-4 space-y-4">
                  {resumeData.education.map((edu, index) => (
                    <div key={edu.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Education {index + 1}</span>
                        <button
                          onClick={() => removeEducation(edu.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={edu.institution}
                          onChange={(e) => updateEducation(edu.id, 'institution', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="University/College"
                        />
                        <input
                          type="text"
                          value={edu.graduationDate}
                          onChange={(e) => updateEducation(edu.id, 'graduationDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Graduation Date (e.g., May 2020)"
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={edu.degree}
                          onChange={(e) => updateEducation(edu.id, 'degree', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Degree (e.g., Bachelor of Science)"
                        />
                        <input
                          type="text"
                          value={edu.field}
                          onChange={(e) => updateEducation(edu.id, 'field', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                          placeholder="Field of Study (e.g., Computer Science)"
                        />
                      </div>
                    </div>
                  ))}
                  <button
                    onClick={addEducation}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-purple-500 hover:text-purple-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Education
                  </button>
                </div>
              )}
            </div>

            {/* Skills Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('skills')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Code className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Skills</span>
                </div>
                {activeSection === 'skills' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'skills' && (
                <div className="p-4">
                  <textarea
                    value={resumeData.skills}
                    onChange={(e) => setResumeData(prev => ({ ...prev, skills: e.target.value }))}
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    rows={4}
                    placeholder="List your skills (e.g., JavaScript, Python, React, Node.js, AWS, Docker...)"
                  />
                  <p className="text-sm text-gray-500 mt-2">Separate skills with commas</p>
                </div>
              )}
            </div>

            {/* Projects Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('projects')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Folder className="w-5 h-5 text-purple-600" />
                  <span className="font-medium">Projects</span>
                  <span className="text-sm text-gray-500">({resumeData.projects.length})</span>
                </div>
                {activeSection === 'projects' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'projects' && (
                <div className="p-4 space-y-4">
                  {resumeData.projects.map((proj, index) => (
                    <div key={proj.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Project {index + 1}</span>
                        <button
                          onClick={() => removeProject(proj.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <input
                        type="text"
                        value={proj.name}
                        onChange={(e) => updateProject(proj.id, 'name', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        placeholder="Project Name"
                      />
                      <textarea
                        value={proj.description}
                        onChange={(e) => updateProject(proj.id, 'description', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        rows={2}
                        placeholder="Project description..."
                      />
                      <input
                        type="text"
                        value={proj.technologies}
                        onChange={(e) => updateProject(proj.id, 'technologies', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        placeholder="Technologies used (e.g., React, Node.js, MongoDB)"
                      />
                    </div>
                  ))}
                  <button
                    onClick={addProject}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-purple-500 hover:text-purple-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Project
                  </button>
                </div>
              )}
            </div>

            {/* Export Buttons */}
            <div className="flex flex-wrap gap-3 pt-4 border-t">
              <Button
                onClick={generatePDF}
                disabled={isGenerating || !isResumeComplete()}
                className="flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                {isGenerating ? 'Generating...' : 'Download as PDF'}
              </Button>
              <Button
                onClick={generateDOCX}
                disabled={isGenerating || !isResumeComplete()}
                variant="secondary"
                className="flex items-center gap-2"
              >
                <FileText className="w-4 h-4" />
                {isGenerating ? 'Generating...' : 'Download as DOCX'}
              </Button>
            </div>

            {!isResumeComplete() && (
              <p className="text-sm text-gray-500 mt-2">
                * Please fill in your name and at least one of: experience, education, or skills
              </p>
            )}
          </div>
        ) : (
          /* Preview Mode with Template Styling */
          <div className="bg-white border rounded-xl p-8 max-w-2xl mx-auto shadow-lg">
            {/* Template-specific header styling */}
            <div className={`text-center border-b pb-6 mb-6 ${
              resumeData.template === 'modern' ? 'bg-gradient-to-r from-purple-600 to-pink-600 -mx-8 -mt-8 p-8 rounded-t-xl' : ''
            }`}>
              <h2 className={`text-2xl font-bold ${
                resumeData.template === 'modern' ? 'text-white' : 
                resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                resumeData.template === 'minimal' ? 'text-gray-700 font-light' :
                'text-orange-600'
              }`}>
                {resumeData.personalInfo.name || 'Your Name'}
              </h2>
              <div className={`text-sm mt-2 space-y-1 ${
                resumeData.template === 'modern' ? 'text-white/80' : 'text-gray-600'
              }`}>
                {resumeData.personalInfo.email && <div>{resumeData.personalInfo.email}</div>}
                {resumeData.personalInfo.phone && <div>{resumeData.personalInfo.phone}</div>}
                {resumeData.personalInfo.location && <div>{resumeData.personalInfo.location}</div>}
                {resumeData.personalInfo.linkedin && <div>LinkedIn: {resumeData.personalInfo.linkedin}</div>}
                {resumeData.personalInfo.portfolio && <div>Portfolio: {resumeData.personalInfo.portfolio}</div>}
              </div>
            </div>

            {resumeData.summary && (
              <div className="mb-6">
                <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
                  resumeData.template === 'modern' ? 'text-purple-600' :
                  resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                  resumeData.template === 'minimal' ? 'text-gray-700' :
                  'text-orange-600'
                }`}>Professional Summary</h3>
                <p className="text-gray-700">{resumeData.summary}</p>
              </div>
            )}

            {resumeData.experience.length > 0 && (
              <div className="mb-6">
                <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
                  resumeData.template === 'modern' ? 'text-purple-600' :
                  resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                  resumeData.template === 'minimal' ? 'text-gray-700' :
                  'text-orange-600'
                }`}>Work Experience</h3>
                {resumeData.experience.map(exp => (
                  <div key={exp.id} className="mb-4">
                    <div className="flex justify-between items-baseline">
                      <span className="font-semibold text-gray-900">{exp.role}</span>
                      <span className="text-sm text-gray-600">
                        {exp.current ? `${exp.startDate} - Present` : `${exp.startDate} - ${exp.endDate}`}
                      </span>
                    </div>
                    <div className="text-gray-600 italic">{exp.company}</div>
                    <p className="text-gray-700 mt-1">{exp.description}</p>
                  </div>
                ))}
              </div>
            )}

            {resumeData.education.length > 0 && (
              <div className="mb-6">
                <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
                  resumeData.template === 'modern' ? 'text-purple-600' :
                  resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                  resumeData.template === 'minimal' ? 'text-gray-700' :
                  'text-orange-600'
                }`}>Education</h3>
                {resumeData.education.map(edu => (
                  <div key={edu.id} className="mb-3">
                    <div className="flex justify-between items-baseline">
                      <span className="font-semibold text-gray-900">{edu.degree} in {edu.field}</span>
                      <span className="text-sm text-gray-600">{edu.graduationDate}</span>
                    </div>
                    <div className="text-gray-600 italic">{edu.institution}</div>
                  </div>
                ))}
              </div>
            )}

            {resumeData.skills && (
              <div className="mb-6">
                <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
                  resumeData.template === 'modern' ? 'text-purple-600' :
                  resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                  resumeData.template === 'minimal' ? 'text-gray-700' :
                  'text-orange-600'
                }`}>Skills</h3>
                <p className="text-gray-700">{resumeData.skills}</p>
              </div>
            )}

            {resumeData.projects.length > 0 && (
              <div>
                <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
                  resumeData.template === 'modern' ? 'text-purple-600' :
                  resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
                  resumeData.template === 'minimal' ? 'text-gray-700' :
                  'text-orange-600'
                }`}>Projects</h3>
                {resumeData.projects.map(proj => (
                  <div key={proj.id} className="mb-3">
                    <div className="font-semibold text-gray-900">{proj.name}</div>
                    {proj.technologies && <div className="text-sm text-gray-600">Technologies: {proj.technologies}</div>}
                    <p className="text-gray-700 mt-1">{proj.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Saved Resumes Sidebar */}
      {savedResumes.length > 0 && (
        <Card title="My Saved Resumes">
          <div className="space-y-3">
            {savedResumes.map(resume => (
              <div key={resume.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900">{resume.name}</h4>
                  <p className="text-sm text-gray-500">
                    {resume.template} template • Updated recently
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => loadResume(resume)}
                    className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg"
                    title="Load"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => deleteResume(resume.id!)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg"
                    title="Delete"
                  >
                    <Trash className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
