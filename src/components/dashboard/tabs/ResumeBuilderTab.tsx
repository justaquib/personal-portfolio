'use client'

import React from 'react'
import { Card, Button } from '../ui'
import { useState, useRef, useEffect } from 'react'
import { jsPDF } from 'jspdf'
import { useAuth } from '@/context/AuthContext'
import { useToast } from '@/components/Toast'
import { 
  User, Mail, Phone, MapPin, Linkedin, Globe, 
  Plus, Trash2, Download, FileText, Briefcase, 
  GraduationCap, Code, Folder, ChevronDown, ChevronUp,
  Save, Edit3, Eye, Sparkles, Layout, Trash, RefreshCw,
  Check, Copy, X, Upload, Award, Settings,
} from 'lucide-react'
import { ResumePreview } from './resume/ResumePreview'
import { PDFPreviewViewer } from './resume/PDFPreviewViewer'
import { downloadResumePDF, generateResumePDF } from './resume/pdfGenerator'
import { ResumeData, Experience, Education, Project, Certification, Website, Language } from './resume/types'
import { TemplateBuilder } from './resume/TemplateBuilder'
import { ResumeToolbar } from './resume/ResumeToolbar'
import QuillEditor from '@/components/QuillEditor'
import { TEMPLATES, SKILL_SUGGESTIONS, SECTION_ORDER } from './resume/constants'
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent
} from '@dnd-kit/core'
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'

// Sortable Section Component
function SortableSection({ id, label, icon: IconComponent, isExpanded, onToggle, children }: {
  id: string
  label: string
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>
  isExpanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1000 : 1,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`border rounded-xl overflow-hidden bg-white ${isDragging ? 'shadow-lg ring-2 ring-blue-500' : ''}`}
    >
      <div
        className="flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors cursor-grab active:cursor-grabbing border-b"
        {...attributes}
        {...listeners}
      >
        <div className="flex items-center gap-3">
          <div className="flex flex-col gap-0.5 text-gray-400">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
            </svg>
          </div>
          <span className="text-gray-700">{IconComponent && <IconComponent className="w-5 h-5" style={{ color: '#212529' }} />}</span>
          <span className="font-medium text-gray-900">{label}</span>
        </div>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onToggle(); }}
          className="p-1 hover:bg-gray-200 rounded"
        >
          {isExpanded ? <ChevronUp className="w-5 h-5 text-gray-600" /> : <ChevronDown className="w-5 h-5 text-gray-600" />}
        </button>
      </div>
      {isExpanded && (
        <div className="border-t border-gray-100">
          {children}
        </div>
      )}
    </div>
  )
}

// Sortable Item Component for list items
function SortableItem({ id, children }: { id: string; children: React.ReactNode }) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging
  } = useSortable({ id })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1000 : 1,
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`relative ${isDragging ? 'shadow-lg ring-2 ring-blue-500' : ''}`}
    >
      <div
        className="absolute left-2 top-1/2 -translate-y-1/2 cursor-grab active:cursor-grabbing p-1 text-gray-400 hover:text-gray-600"
        {...attributes}
        {...listeners}
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
        </svg>
      </div>
      <div className="pl-8">
        {children}
      </div>
    </div>
  )
}

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
  projects: [],
  certifications: [],
  websites: [],
  languages: [],
  sectionOrder: ['personal', 'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'websites', 'languages']
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

// Extract text from PDF
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

export function ResumeBuilderTab() {
  const { user, isSuperAdmin, isAdmin } = useAuth()
  const [resumeData, setResumeData] = useState<ResumeData>(defaultResumeData)
  const [savedResumes, setSavedResumes] = useState<ResumeData[]>([])
  const [activeSection, setActiveSection] = useState<string | null>('personal')
  const [isGenerating, setIsGenerating] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [showPDFPreview, setShowPDFPreview] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [activeTool, setActiveTool] = useState<string | null>(null)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [showResumesDropdown, setShowResumesDropdown] = useState(false)
  const [resumeName, setResumeName] = useState('My Resume')
  const [isSaving, setIsSaving] = useState(false)
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([])
  const [showTemplateBuilder, setShowTemplateBuilder] = useState(false)
  // Track which field is being enhanced
  const [enhancingField, setEnhancingField] = useState<{type: 'summary' | 'experience' | 'project', id?: string} | null>(null)
  // Toggle states for suggestions
  const [showAISuggestions, setShowAISuggestions] = useState(true)
  const [showSkillSuggestions, setShowSkillSuggestions] = useState(false)
  // Section order state for drag and drop
  const [sectionOrder, setSectionOrder] = useState<string[]>(
    defaultResumeData.sectionOrder || ['personal', 'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'websites', 'languages']
  )
  // PDF Import states
  const [isImportingPDF, setIsImportingPDF] = useState(false)
  const [importProgress, setImportProgress] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const resumeRef = useRef<HTMLDivElement>(null)

  // Toast hook
  const { showToast } = useToast()

  // DnD sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

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
          certifications: typeof r.certifications === 'string' ? JSON.parse(r.certifications) : (r.certifications || []),
          websites: typeof r.websites === 'string' ? JSON.parse(r.websites) : (r.websites || []),
          languages: typeof r.languages === 'string' ? JSON.parse(r.languages) : (r.languages || []),
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

  // Handle PDF file selection
  const handlePDFImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Please select a PDF file', 'error')
      return
    }

    setIsImportingPDF(true)
    setImportProgress('Reading PDF file...')

    try {
      // Step 1: Extract text from PDF
      setImportProgress('Extracting text from PDF...')
      const extractedText = await extractTextFromPDF(file)

      if (!extractedText || extractedText.trim().length < 50) {
        showToast('Could not extract enough text from the PDF. Please try a different file.', 'error')
        setIsImportingPDF(false)
        setImportProgress('')
        return
      }

      // Step 2: Parse resume data using AI
      setImportProgress('Analyzing resume with AI...')
      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'parse_resume',
          content: extractedText.substring(0, 50000)
        })
      })

      if (!response.ok) {
        throw new Error('Failed to parse resume data')
      }

      const data = await response.json()
      const parsedData = data.parsedData

      if (!parsedData) {
        showToast('Could not parse resume data. Please try again or enter manually.', 'error')
        setIsImportingPDF(false)
        setImportProgress('')
        return
      }

      // Step 3: Update resume data with parsed information
      setImportProgress('Populating resume form...')
      
      // Generate unique IDs for new entries
      const generateId = () => Date.now().toString() + Math.random().toString(36).substr(2, 9)

      const newResumeData: ResumeData = {
        ...defaultResumeData,
        personalInfo: {
          name: parsedData.personalInfo?.name || '',
          email: parsedData.personalInfo?.email || '',
          phone: parsedData.personalInfo?.phone || '',
          location: parsedData.personalInfo?.location || '',
          linkedin: parsedData.personalInfo?.linkedin || '',
          portfolio: parsedData.personalInfo?.portfolio || ''
        },
        summary: parsedData.summary || '',
        experience: (parsedData.experience || []).map((exp: any) => ({
          id: generateId(),
          company: exp.company || '',
          role: exp.role || '',
          startDate: exp.startDate || '',
          endDate: exp.endDate || '',
          current: exp.current || false,
          description: exp.description || ''
        })),
        education: (parsedData.education || []).map((edu: any) => ({
          id: generateId(),
          institution: edu.institution || '',
          degree: edu.degree || '',
          field: edu.field || '',
          graduationDate: edu.graduationDate || ''
        })),
        skills: parsedData.skills || '',
        projects: (parsedData.projects || []).map((proj: any) => ({
          id: generateId(),
          name: proj.name || '',
          description: proj.description || '',
          technologies: proj.technologies || ''
        })),
        certifications: (parsedData.certifications || []).map((cert: any) => ({
          id: generateId(),
          name: cert.name || '',
          issuer: cert.issuer || '',
          date: cert.date || '',
          url: cert.url || ''
        })),
        websites: (parsedData.websites || []).map((ws: any) => ({
          id: generateId(),
          name: ws.name || '',
          url: ws.url || ''
        })),
        languages: (parsedData.languages || []).map((lang: any) => ({
          id: generateId(),
          name: lang.name || '',
          proficiency: lang.proficiency || ''
        }))
      }

      setResumeData(newResumeData)
      setResumeName('Imported Resume')
      
      // Open all sections to show imported data
      setActiveSection(null)
      
      showToast('Resume imported successfully! Please review and edit the information below.', 'success')
    } catch (error) {
      console.error('Error importing PDF:', error)
      showToast('Failed to import PDF. Please try again or enter your resume information manually.', 'error')
    } finally {
      setIsImportingPDF(false)
      setImportProgress('')
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
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
        { id: Date.now().toString(), company: '', role: '', location: '', startDate: '', endDate: '', current: false, description: '' }
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

  // Certification handlers
  const addCertification = () => {
    setResumeData(prev => ({
      ...prev,
      certifications: [
        ...prev.certifications,
        { id: Date.now().toString(), name: '', issuer: '', date: '', url: '' }
      ]
    }))
  }

  const updateCertification = (id: string, field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      certifications: prev.certifications.map(cert => 
        cert.id === id ? { ...cert, [field]: value } : cert
      )
    }))
  }

  const removeCertification = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      certifications: prev.certifications.filter(cert => cert.id !== id)
    }))
  }

  // Website handlers
  const addWebsite = () => {
    setResumeData(prev => ({
      ...prev,
      websites: [
        ...prev.websites,
        { id: Date.now().toString(), name: '', url: '' }
      ]
    }))
  }

  const updateWebsite = (id: string, field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      websites: prev.websites.map(ws => 
        ws.id === id ? { ...ws, [field]: value } : ws
      )
    }))
  }

  const removeWebsite = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      websites: prev.websites.filter(ws => ws.id !== id)
    }))
  }

  // Language handlers
  const addLanguage = () => {
    setResumeData(prev => ({
      ...prev,
      languages: [
        ...prev.languages,
        { id: Date.now().toString(), name: '', proficiency: '' }
      ]
    }))
  }

  const updateLanguage = (id: string, field: string, value: string) => {
    setResumeData(prev => ({
      ...prev,
      languages: prev.languages.map(lang => 
        lang.id === id ? { ...lang, [field]: value } : lang
      )
    }))
  }

  const removeLanguage = (id: string) => {
    setResumeData(prev => ({
      ...prev,
      languages: prev.languages.filter(lang => lang.id !== id)
    }))
  }

  // Handle drag end for item reordering
  const handleProjectsReorder = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setResumeData(prev => {
        const oldIndex = prev.projects.findIndex(p => p.id === active.id)
        const newIndex = prev.projects.findIndex(p => p.id === over.id)
        return {
          ...prev,
          projects: arrayMove(prev.projects, oldIndex, newIndex)
        }
      })
    }
  }

  const handleCertificationsReorder = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setResumeData(prev => {
        const oldIndex = prev.certifications.findIndex(c => c.id === active.id)
        const newIndex = prev.certifications.findIndex(c => c.id === over.id)
        return {
          ...prev,
          certifications: arrayMove(prev.certifications, oldIndex, newIndex)
        }
      })
    }
  }

  const handleWebsitesReorder = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setResumeData(prev => {
        const oldIndex = prev.websites.findIndex(w => w.id === active.id)
        const newIndex = prev.websites.findIndex(w => w.id === over.id)
        return {
          ...prev,
          websites: arrayMove(prev.websites, oldIndex, newIndex)
        }
      })
    }
  }

  const handleExperienceReorder = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setResumeData(prev => {
        const oldIndex = prev.experience.findIndex(e => e.id === active.id)
        const newIndex = prev.experience.findIndex(e => e.id === over.id)
        return {
          ...prev,
          experience: arrayMove(prev.experience, oldIndex, newIndex)
        }
      })
    }
  }

  const handleEducationReorder = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setResumeData(prev => {
        const oldIndex = prev.education.findIndex(e => e.id === active.id)
        const newIndex = prev.education.findIndex(e => e.id === over.id)
        return {
          ...prev,
          education: arrayMove(prev.education, oldIndex, newIndex)
        }
      })
    }
  }

  // Handle drag end for section reordering
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event
    if (over && active.id !== over.id) {
      setSectionOrder((items) => {
        const oldIndex = items.indexOf(active.id as string)
        const newIndex = items.indexOf(over.id as string)
        return arrayMove(items, oldIndex, newIndex)
      })
    }
  }

  // Reset section order to default
  const resetSectionOrder = () => {
    setSectionOrder(defaultResumeData.sectionOrder || ['personal', 'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'websites', 'languages'])
  }

  const selectTemplate = (templateId: string) => {
    setResumeData(prev => ({ ...prev, template: templateId }))
    // Keep template selector open for user to close manually
  }

  // Enhance summary using Gemini AI - improves the existing summary
  const generateAISummary = async () => {
    // Expand summary section first if collapsed
    setActiveSection('summary')
    setEnhancingField({ type: 'summary' })
    
    setIsLoadingAI(true)
    try {
      const currentSummary = resumeData.summary
      
      let prompt = ''
      
      if (currentSummary) {
        // Enhance existing summary
        prompt = `You are a professional resume writer. Please enhance and improve the following professional summary to make it more impactful, professional, and compelling. Keep it concise (2-4 sentences). Return only the enhanced summary, nothing else.

Current summary:
${currentSummary}`
      } else if (resumeData.experience.length > 0 || resumeData.education.length > 0 || resumeData.skills || resumeData.projects.length > 0) {
        // Create summary from resume data
        prompt = `You are a professional resume writer. Write a compelling professional summary (2-4 sentences) based on the following resume information. Make it impactful and professional. Return only the summary, nothing else.

Experience: ${resumeData.experience.map(exp => `${exp.role} at ${exp.company} (${exp.startDate} - ${exp.current ? 'Present' : exp.endDate}): ${exp.description}`).join('\n')}

Education: ${resumeData.education.map(edu => `${edu.degree} in ${edu.field} from ${edu.institution} (${edu.graduationDate})`).join('\n')}

Skills: ${resumeData.skills}

Projects: ${resumeData.projects.map(proj => `${proj.name}: ${proj.description} (${proj.technologies})`).join('\n')}`
      } else {
        // No content to work with - ask user to add content
        setIsLoadingAI(false)
        showToast('Please add some content to your resume first (experience, education, skills, or projects) before enhancing the summary.', 'info')
        return
      }

      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'enhance',
          prompt: prompt
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.answer) {
          setResumeData(prev => ({ ...prev, summary: data.answer }))
          setAiSuggestions([data.answer])
          setShowAISuggestions(true) // Show suggestions when generated
        } else if (data.error) {
          showToast('Error: ' + data.error, 'error')
        }
      } else {
        const errorData = await response.json()
        showToast('Failed to enhance summary: ' + (errorData.error || 'Unknown error'), 'error')
      }
    } catch (error) {
      console.error('Error enhancing AI summary:', error)
      showToast('Failed to enhance summary. Please try again.', 'error')
    } finally {
      setIsLoadingAI(false)
    }
  }

  // Enhance experience description using Gemini AI
  const generateAIExperienceDescription = async (expId: string) => {
    const exp = resumeData.experience.find(e => e.id === expId)
    if (!exp) return
    
    if (!exp.description) {
      showToast('Please add some content to the description first', 'info')
      return
    }

    setEnhancingField({ type: 'experience', id: expId })
    setIsLoadingAI(true)
    
    try {
      const prompt = `You are a professional resume writer. Please enhance and improve the following work experience description to make it more impactful, professional, and compelling. Use action verbs and quantify achievements where possible. Return only the enhanced description, nothing else.

Current description:
${exp.description}

Job Title: ${exp.role}
Company: ${exp.company}`

      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'enhance',
          prompt: prompt
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.answer) {
          updateExperience(expId, 'description', data.answer)
          showToast('Experience description enhanced!', 'success')
        } else if (data.error) {
          showToast('Error: ' + data.error, 'error')
        }
      }
    } catch (error) {
      console.error('Error enhancing experience:', error)
      showToast('Failed to enhance description. Please try again.', 'error')
    } finally {
      setIsLoadingAI(false)
      setEnhancingField(null)
    }
  }

  // Enhance project description using Gemini AI
  const generateAIProjectDescription = async (projId: string) => {
    const proj = resumeData.projects.find(p => p.id === projId)
    if (!proj) return
    
    if (!proj.description) {
      showToast('Please add some content to the description first', 'info')
      return
    }

    setEnhancingField({ type: 'project', id: projId })
    setIsLoadingAI(true)
    
    try {
      const prompt = `You are a professional resume writer. Please enhance and improve the following project description to make it more impactful, professional, and compelling. Highlight your technical skills and achievements. Return only the enhanced description, nothing else.

Current description:
${proj.description}

Project Name: ${proj.name}
Technologies: ${proj.technologies}`

      const response = await fetch('/api/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'enhance',
          prompt: prompt
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.answer) {
          updateProject(projId, 'description', data.answer)
          showToast('Project description enhanced!', 'success')
        } else if (data.error) {
          showToast('Error: ' + data.error, 'error')
        }
      }
    } catch (error) {
      console.error('Error enhancing project:', error)
      showToast('Failed to enhance description. Please try again.', 'error')
    } finally {
      setIsLoadingAI(false)
      setEnhancingField(null)
    }
  }

  // Save resume to database
  const saveResume = async () => {
    if (!user?.id) {
      showToast('Please sign in to save your resume', 'info')
      return
    }

    // Check if we're updating an existing resume
    const isUpdating = resumeData.id !== undefined;
    
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
        certifications: resumeData.certifications,
        websites: resumeData.websites,
        languages: resumeData.languages,
        isDefault: savedResumes.length === 0 && !isUpdating
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
        showToast(isUpdating ? 'Resume updated successfully!' : 'Resume saved successfully!', 'success')
      }
    } catch (error) {
      console.error('Error saving resume:', error)
      showToast('Failed to save resume', 'error')
    } finally {
      setIsSaving(false)
    }
  }

  // Update saved resume
  const updateSavedResume = async () => {
    if (!user?.id || !resumeData.id) {
      showToast('No resume selected to update', 'info')
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
        certifications: resumeData.certifications,
        websites: resumeData.websites,
        languages: resumeData.languages,
        isDefault: false
      }

      const response = await fetch('/api/resumes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        await loadResumes()
        showToast('Resume updated successfully!', 'success')
      }
    } catch (error) {
      console.error('Error updating resume:', error)
      showToast('Failed to update resume', 'error')
    } finally {
      setIsSaving(false)
    }
  }

  // Load selected resume
  const loadResume = (resume: ResumeData) => {
    setResumeData(resume)
    setResumeName(resume.name)
    setShowTemplates(false)
    setActiveTool(null)
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
  
  ${resumeData.certifications.length > 0 ? `
  <div class="section">
    <h2>CERTIFICATIONS</h2>
    <div class="badges">
    ${resumeData.certifications.map(cert => `
      <span class="badge" style="display: inline-block; background: #9333ea; color: white; padding: 4px 12px; border-radius: 20px; margin: 2px; font-size: 12px;">${cert.issuer ? cert.name + ' (' + cert.issuer + ')' : cert.name}</span>
    `).join('')}
    </div>
  </div>
  ` : ''}
  
  ${resumeData.languages.length > 0 ? `
  <div class="section">
    <h2>LANGUAGES</h2>
    <div class="badges">
    ${resumeData.languages.map(lang => `
      <span class="badge" style="display: inline-block; background: #4f46e5; color: white; padding: 4px 12px; border-radius: 20px; margin: 2px; font-size: 12px;">${lang.name} - ${lang.proficiency}</span>
    `).join('')}
    </div>
  </div>
  ` : ''}
  
  ${resumeData.websites.length > 0 ? `
  <div class="section">
    <h2>WEBSITES & PORTFOLIO</h2>
    <div class="badges">
    ${resumeData.websites.map(ws => `
      <span class="badge" style="display: inline-block; background: #0891b2; color: white; padding: 4px 12px; border-radius: 20px; margin: 2px; font-size: 12px;">${ws.name}: ${ws.url}</span>
    `).join('')}
    </div>
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

      showToast('DOCX file downloaded! Note: The file is saved as .doc which can be opened in Microsoft Word.', 'success')
    } catch (error) {
      console.error('Error generating DOCX:', error)
      showToast('Failed to generate DOCX. Please try again.', 'error')
    } finally {
      setIsGenerating(false)
    }
  }

  const toggleSection = (section: string) => {
    setActiveSection(activeSection === section ? null : section)
  }

  // Check if resume has any content to save
  const hasResumeContent = (): boolean => {
    return !!(resumeData.personalInfo.name || 
           resumeData.personalInfo.email || 
           resumeData.personalInfo.phone || 
           resumeData.summary || 
           resumeData.experience.length > 0 || 
           resumeData.education.length > 0 || 
           resumeData.skills || 
           resumeData.projects.length > 0)
  }

  // Check if resume is complete enough for export
  const isResumeComplete = () => {
    return resumeData.personalInfo.name && 
           (resumeData.experience.length > 0 || resumeData.education.length > 0 || resumeData.skills)
  }

  const selectedTemplate = TEMPLATES.find(t => t.id === resumeData.template) || TEMPLATES[0]

  return (
    <div className="space-y-6">
      <Card>
        {/* Header with title and toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-2 mb-6">
          {/* Title */}
          <h2 className="text-xl font-semibold text-gray-900">Resume Builder</h2>
          
          {/* Toolbar - consolidated in ResumeToolbar component */}
          <ResumeToolbar
            resumeData={resumeData}
            savedResumes={savedResumes}
            showPreview={showPreview}
            showPDFPreview={showPDFPreview}
            showTemplates={showTemplates}
            showTemplateBuilder={showTemplateBuilder}
            activeTool={activeTool}
            isSuperAdmin={isSuperAdmin || false}
            isAdmin={isAdmin || false}
            isImportingPDF={isImportingPDF}
            importProgress={importProgress}
            selectedTemplate={selectedTemplate}
            resumeName={resumeName}
            isSaving={isSaving}
            hasResumeContent={hasResumeContent}
            fileInputRef={fileInputRef}
            showResumesDropdown={showResumesDropdown}
            onImportPDF={handlePDFImport}
            onToggleTemplates={() => {
              setActiveTool(activeTool === 'templates' ? null : 'templates')
              setShowTemplates(!showTemplates)
            }}
            onTogglePreview={(preview) => {
              setActiveTool(preview ? 'preview' : 'edit')
              setShowPreview(preview)
              setShowPDFPreview(false)
              setShowTemplates(false)
              setShowTemplateBuilder(false)
            }}
            onTogglePDFPreview={(preview) => {
              setActiveTool(preview ? 'pdfPreview' : 'edit')
              setShowPDFPreview(preview)
              setShowPreview(false)
              setShowTemplates(false)
              setShowTemplateBuilder(false)
            }}
            onToggleTemplateBuilder={() => {
              setActiveTool(activeTool === 'templateBuilder' ? null : 'templateBuilder')
              setShowTemplateBuilder(!showTemplateBuilder)
            }}
            onResetSectionOrder={resetSectionOrder}
            onSave={() => setShowSaveModal(true)}
            onUpdateResume={updateSavedResume}
            onLoadResume={loadResume}
            onSelectTemplate={selectTemplate}
            onCloseTemplates={() => {
              setShowTemplates(false)
              setActiveTool(null)
            }}
            onCloseTemplateBuilder={() => {
              setShowTemplateBuilder(false)
              setActiveTool(null)
            }}
            onToggleResumesDropdown={() => setShowResumesDropdown(!showResumesDropdown)}
          />
        </div>

        {/* Template Selector Dropdown */}
        {showTemplates && (
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-medium">Choose a Template</h3>
              <button 
                onClick={() => {
                  setShowTemplates(false)
                  setActiveTool(null)
                }}
                className="p-1 hover:bg-gray-200 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {TEMPLATES.map(template => (
                <button
                  key={template.id}
                  onClick={() => selectTemplate(template.id)}
                  className={`p-4 rounded-xl border-2 transition-all ${
                    resumeData.template === template.id 
                      ? 'border bg-white shadow-md' 
                      : 'border bg-white'
                  }`}
                  style={{
                    borderColor: resumeData.template === template.id ? '#212529' : '#dee2e6',
                  }}
                >
                  <div className={`h-8 rounded-lg mb-3 ${template.preview}`}></div>
                  <h4 className="font-medium text-gray-900">{template.name}</h4>
                  <p className="text-xs text-gray-500 mt-1">{template.description}</p>
                  {resumeData.template === template.id && (
                    <div className="mt-2 flex items-center gap-1 text-sm" style={{ color: '#212529' }}>
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
                <h3 className="text-lg font-semibold">{resumeData.id ? 'Update Resume' : 'Save Resume'}</h3>
                <button onClick={() => setShowSaveModal(false)} className="p-1 hover:bg-gray-100 rounded">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <input
                type="text"
                value={resumeName}
                onChange={(e) => setResumeName(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg mb-4"
                style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
                placeholder="Resume name"
              />
              <div className="flex gap-3">
                {resumeData.id ? (
                  <>
                    <Button 
                      onClick={updateSavedResume} 
                      disabled={isSaving || !hasResumeContent()} 
                      className="flex-1"
                      style={{
                        backgroundColor: '#212529',
                        color: '#ffffff'
                      }}
                    >
                      {isSaving ? 'Updating...' : 'Update Resume'}
                    </Button>
                    <Button variant="secondary" onClick={() => setShowSaveModal(false)}>
                      Cancel
                    </Button>
                  </>
                ) : (
                  <>
                    <Button 
                      onClick={saveResume} 
                      disabled={isSaving || !hasResumeContent()} 
                      className="flex-1"
                      style={{
                        opacity: isSaving || !hasResumeContent() ? 0.5 : 1,
                        cursor: isSaving || !hasResumeContent() ? 'not-allowed' : 'pointer'
                      }}
                    >
                      {isSaving ? 'Saving...' : 'Save Resume'}
                    </Button>
                    <Button variant="secondary" onClick={() => setShowSaveModal(false)}>
                      Cancel
                    </Button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {!showPreview && !showPDFPreview ? (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={sectionOrder}
              strategy={verticalListSortingStrategy}
            >
              <div className="space-y-4">
                {sectionOrder.map((sectionId) => {
                  const section = SECTION_ORDER.find(s => s.id === sectionId)
                  if (!section) return null
                  
                  // Map icon string names to actual icon components
                  const iconMap: Record<string, React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
                    User,
                    FileText,
                    Briefcase,
                    GraduationCap,
                    Code,
                    Folder,
                    Award,
                    Globe,
                  }
                  const IconComponent = iconMap[section.icon as string] || User
                  
                  return (
                    <SortableSection
                      key={sectionId}
                      id={sectionId}
                      label={section.label}
                      icon={IconComponent}
                      isExpanded={activeSection === sectionId}
                      onToggle={() => toggleSection(sectionId)}
                    >
                      {sectionId === 'personal' && (
                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Full Name *</label>
                            <input
                              type="text"
                              value={resumeData.personalInfo.name}
                              onChange={(e) => updatePersonalInfo('name', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2"
                              style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
                              placeholder="John Doe"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Email *</label>
                            <input
                              type="email"
                              value={resumeData.personalInfo.email}
                              onChange={(e) => updatePersonalInfo('email', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2"
                              style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
                              placeholder="john@example.com"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                            <input
                              type="tel"
                              value={resumeData.personalInfo.phone}
                              onChange={(e) => updatePersonalInfo('phone', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2"
                              style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
                              placeholder="+1 (555) 123-4567"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                            <input
                              type="text"
                              value={resumeData.personalInfo.location}
                              onChange={(e) => updatePersonalInfo('location', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                              placeholder="New York, NY"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">LinkedIn</label>
                            <input
                              type="text"
                              value={resumeData.personalInfo.linkedin}
                              onChange={(e) => updatePersonalInfo('linkedin', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                              placeholder="linkedin.com/in/johndoe"
                            />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">Portfolio</label>
                            <input
                              type="text"
                              value={resumeData.personalInfo.portfolio}
                              onChange={(e) => updatePersonalInfo('portfolio', e.target.value)}
                              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                              placeholder="johndoe.com"
                            />
                          </div>
                        </div>
                      )}

                      {sectionId === 'summary' && activeSection === 'summary' && (
                        <div className="p-4">
                          <QuillEditor
                            value={resumeData.summary}
                            onChange={(value: string) => setResumeData(prev => ({ ...prev, summary: value }))}
                            placeholder="Write a brief summary of your professional background and career goals..."
                            height="120px"
                            onEnhance={generateAISummary}
                            isEnhancing={isLoadingAI}
                          />
                          {aiSuggestions.length > 0 && showAISuggestions && (
                            <div className="mt-3 p-3 bg-gray-100 rounded-lg">
                              <div className="flex justify-between items-center mb-2">
                                <p className="text-sm text-gray-700 mb-0">AI Suggestion:</p>
                                <button
                                  type="button"
                                  onClick={() => setShowAISuggestions(false)}
                                  className="text-gray-400 hover:text-gray-600"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                              <p className="text-sm text-gray-700">{aiSuggestions[0]}</p>
                              <button
                                onClick={() => {
                                  setResumeData(prev => ({ ...prev, summary: aiSuggestions[0] }))
                                  setAiSuggestions([])
                                }}
                                className="mt-2 text-sm text-gray-600 hover:text-gray-700 flex items-center gap-1"
                              >
                                <Copy className="w-4 h-4" /> Use this
                              </button>
                            </div>
                          )}

                          {(!showAISuggestions || aiSuggestions.length === 0) && (
                            <button
                              type="button"
                              onClick={() => setShowAISuggestions(true)}
                              className="mt-3 text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
                            >
                              <Sparkles className="w-4 h-4" /> Show AI Suggestions
                            </button>
                          )}
                        </div>
                      )}

                      {sectionId === 'experience' && (
                        <div className="p-4">
                          <DndContext
                            sensors={sensors}
                            collisionDetection={closestCenter}
                            onDragEnd={handleExperienceReorder}
                          >
                            <SortableContext
                              items={resumeData.experience.map(e => e.id)}
                              strategy={verticalListSortingStrategy}
                            >
                              <div className="space-y-4">
                                {resumeData.experience.map((exp, index) => (
                                  <SortableItem key={exp.id} id={exp.id}>
                                    <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                      <div className="flex justify-between items-start">
                                        <span className="text-sm font-medium text-gray-500">Experience {index + 1}</span>
                                        <button onClick={() => removeExperience(exp.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={exp.company} onChange={(e) => updateExperience(exp.id, 'company', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Company Name" />
                                        <input type="text" value={exp.role} onChange={(e) => updateExperience(exp.id, 'role', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Job Title" />
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                        <input type="text" value={exp.startDate} onChange={(e) => updateExperience(exp.id, 'startDate', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Start Date" />
                                        <input type="text" value={exp.endDate} onChange={(e) => updateExperience(exp.id, 'endDate', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="End Date" disabled={exp.current} />
                                        <label className="flex items-center gap-2">
                                          <input type="checkbox" checked={exp.current} onChange={(e) => updateExperience(exp.id, 'current', e.target.checked)} className="w-4 h-4 rounded" />
                                          <span className="text-sm text-gray-700">Currently working</span>
                                        </label>
                                      </div>
                                      <input 
                                        type="text" 
                                        value={exp.location || ''} 
                                        onChange={(e) => updateExperience(exp.id, 'location', e.target.value)} 
                                        className="px-3 py-2 border rounded-lg w-full" 
                                        placeholder="Location (e.g., New York, NY)" 
                                      />
                                      <div className="md:col-span-2">
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                                        <QuillEditor
                                          value={exp.description}
                                          onChange={(value: string) => updateExperience(exp.id, 'description', value)}
                                          placeholder="Describe your responsibilities and achievements..."
                                          height="100px"
                                          onEnhance={() => generateAIExperienceDescription(exp.id)}
                                          isEnhancing={isLoadingAI && enhancingField?.type === 'experience' && enhancingField?.id === exp.id}
                                        />
                                      </div>
                                    </div>
                                  </SortableItem>
                                ))}
                              </div>
                            </SortableContext>
                          </DndContext>
                          <button onClick={addExperience} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 mt-4">
                            <Plus className="w-5 h-5" /> Add Experience
                          </button>
                        </div>
                      )}

                      {sectionId === 'education' && (
                        <div className="p-4">
                          <DndContext
                            sensors={sensors}
                            collisionDetection={closestCenter}
                            onDragEnd={handleEducationReorder}
                          >
                            <SortableContext
                              items={resumeData.education.map(e => e.id)}
                              strategy={verticalListSortingStrategy}
                            >
                              <div className="space-y-4">
                                {resumeData.education.map((edu, index) => (
                                  <SortableItem key={edu.id} id={edu.id}>
                                    <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                      <div className="flex justify-between items-start">
                                        <span className="text-sm font-medium text-gray-500">Education {index + 1}</span>
                                        <button onClick={() => removeEducation(edu.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={edu.institution} onChange={(e) => updateEducation(edu.id, 'institution', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="University/College" />
                                        <input type="text" value={edu.graduationDate} onChange={(e) => updateEducation(edu.id, 'graduationDate', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Graduation Date" />
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={edu.degree} onChange={(e) => updateEducation(edu.id, 'degree', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Degree" />
                                        <input type="text" value={edu.field} onChange={(e) => updateEducation(edu.id, 'field', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Field of Study" />
                                      </div>
                                    </div>
                                  </SortableItem>
                                ))}
                              </div>
                            </SortableContext>
                          </DndContext>
                          <button onClick={addEducation} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 mt-4">
                            <Plus className="w-5 h-5" /> Add Education
                          </button>
                        </div>
                      )}

                      {sectionId === 'skills' && (
                        <div className="p-4">
                          <div className="relative">
                            <textarea
                              value={resumeData.skills}
                              onChange={(e) => setResumeData(prev => ({ ...prev, skills: e.target.value }))}
                              onFocus={() => setShowSkillSuggestions(true)}
                              className="w-full px-3 py-2 border rounded-lg"
                              style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529', minHeight: '100px' }}
                              placeholder="List your skills (e.g., JavaScript, Python, React...)"
                            />
                            {/* Skill Suggestions Dropdown */}
                            {showSkillSuggestions && (
                              <div className="mt-2 p-3 bg-gray-50 rounded-lg">
                                <div className="flex justify-between items-center mb-2">
                                  <p className="text-sm text-gray-600 mb-0">Suggested skills (click to add):</p>
                                  <button
                                    type="button"
                                    onClick={() => setShowSkillSuggestions(false)}
                                    className="text-gray-400 hover:text-gray-600"
                                  >
                                    <X className="w-4 h-4" />
                                  </button>
                                </div>
                                <div className="flex flex-wrap gap-2">
                                  {SKILL_SUGGESTIONS.filter(skill => {
                                    const currentSkills = (resumeData.skills || '').toLowerCase()
                                    return !currentSkills.includes(skill.toLowerCase())
                                  }).slice(0, 20).map((skill, index) => (
                                    <button
                                      key={index}
                                      type="button"
                                      onClick={() => {
                                        const currentSkills = resumeData.skills || ''
                                        const newSkills = currentSkills ? currentSkills + ', ' + skill : skill
                                        setResumeData(prev => ({ ...prev, skills: newSkills }))
                                      }}
                                      className="px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                                      style={{ color: '#212529' }}
                                    >
                                      + {skill}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                          <p className="text-sm text-gray-500 mt-2">Separate skills with commas</p>
                        </div>
                      )}

                      {sectionId === 'projects' && (
                        <div className="p-4">
                          <DndContext
                            sensors={sensors}
                            collisionDetection={closestCenter}
                            onDragEnd={handleProjectsReorder}
                          >
                            <SortableContext
                              items={resumeData.projects.map(p => p.id)}
                              strategy={verticalListSortingStrategy}
                            >
                              <div className="space-y-4">
                                {resumeData.projects.map((proj, index) => (
                                  <SortableItem key={proj.id} id={proj.id}>
                                    <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                      <div className="flex justify-between items-start">
                                        <span className="text-sm font-medium text-gray-500">Project {index + 1}</span>
                                        <button onClick={() => removeProject(proj.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
                                      <input type="text" value={proj.name} onChange={(e) => updateProject(proj.id, 'name', e.target.value)} className="w-full px-3 py-2 border rounded-lg" placeholder="Project Name" />
                                      <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                                        <QuillEditor
                                          value={proj.description}
                                          onChange={(value: string) => updateProject(proj.id, 'description', value)}
                                          placeholder="Describe the project, your role, and achievements..."
                                          height="100px"
                                          onEnhance={() => generateAIProjectDescription(proj.id)}
                                          isEnhancing={isLoadingAI && enhancingField?.type === 'project' && enhancingField?.id === proj.id}
                                        />
                                      </div>
                                      <input type="text" value={proj.technologies} onChange={(e) => updateProject(proj.id, 'technologies', e.target.value)} className="w-full px-3 py-2 border rounded-lg" placeholder="Technologies used" />
                                    </div>
                                  </SortableItem>
                                ))}
                              </div>
                            </SortableContext>
                          </DndContext>
                          <button onClick={addProject} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 mt-4">
                            <Plus className="w-5 h-5" /> Add Project
                          </button>
                        </div>
                      )}

                      {sectionId === 'certifications' && (
                        <div className="p-4">
                          <DndContext
                            sensors={sensors}
                            collisionDetection={closestCenter}
                            onDragEnd={handleCertificationsReorder}
                          >
                            <SortableContext
                              items={resumeData.certifications.map(c => c.id)}
                              strategy={verticalListSortingStrategy}
                            >
                              <div className="space-y-4">
                                {resumeData.certifications.map((cert, index) => (
                                  <SortableItem key={cert.id} id={cert.id}>
                                    <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                      <div className="flex justify-between items-start">
                                        <span className="text-sm font-medium text-gray-500">Certification {index + 1}</span>
                                        <button onClick={() => removeCertification(cert.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={cert.name} onChange={(e) => updateCertification(cert.id, 'name', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Certification Name" />
                                        <input type="text" value={cert.issuer} onChange={(e) => updateCertification(cert.id, 'issuer', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Issuing Organization" />
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={cert.date} onChange={(e) => updateCertification(cert.id, 'date', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Date" />
                                        <input type="text" value={cert.url} onChange={(e) => updateCertification(cert.id, 'url', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Certificate URL" />
                                      </div>
                                    </div>
                                  </SortableItem>
                                ))}
                              </div>
                            </SortableContext>
                          </DndContext>
                          <button onClick={addCertification} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 mt-4">
                            <Plus className="w-5 h-5" /> Add Certification
                          </button>
                        </div>
                      )}

                      {sectionId === 'websites' && (
                        <div className="p-4">
                          <DndContext
                            sensors={sensors}
                            collisionDetection={closestCenter}
                            onDragEnd={handleWebsitesReorder}
                          >
                            <SortableContext
                              items={resumeData.websites.map(w => w.id)}
                              strategy={verticalListSortingStrategy}
                            >
                              <div className="space-y-4">
                                {resumeData.websites.map((ws, index) => (
                                  <SortableItem key={ws.id} id={ws.id}>
                                    <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                      <div className="flex justify-between items-start">
                                        <span className="text-sm font-medium text-gray-500">Website {index + 1}</span>
                                        <button onClick={() => removeWebsite(ws.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <input type="text" value={ws.name} onChange={(e) => updateWebsite(ws.id, 'name', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Website Name (e.g., GitHub)" />
                                        <input type="text" value={ws.url} onChange={(e) => updateWebsite(ws.id, 'url', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="URL" />
                                      </div>
                                    </div>
                                  </SortableItem>
                                ))}
                              </div>
                            </SortableContext>
                          </DndContext>
                          <button onClick={addWebsite} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 mt-4">
                            <Plus className="w-5 h-5" /> Add Website
                          </button>
                        </div>
                      )}

                      {sectionId === 'languages' && (
                        <div className="p-4 space-y-4">
                          {resumeData.languages.map((lang, index) => (
                            <div key={lang.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                              <div className="flex justify-between items-start">
                                <span className="text-sm font-medium text-gray-500">Language {index + 1}</span>
                                <button onClick={() => removeLanguage(lang.id)} className="p-1 text-red-500 hover:bg-red-50 rounded">
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                <input type="text" value={lang.name} onChange={(e) => updateLanguage(lang.id, 'name', e.target.value)} className="px-3 py-2 border rounded-lg" placeholder="Language (e.g., English)" />
                                <select value={lang.proficiency} onChange={(e) => updateLanguage(lang.id, 'proficiency', e.target.value)} className="px-3 py-2 border rounded-lg">
                                  <option value="">Select Proficiency</option>
                                  <option value="Native">Native</option>
                                  <option value="Fluent">Fluent</option>
                                  <option value="Advanced">Advanced</option>
                                  <option value="Intermediate">Intermediate</option>
                                  <option value="Basic">Basic</option>
                                </select>
                              </div>
                            </div>
                          ))}
                          <button onClick={addLanguage} className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500">
                            <Plus className="w-5 h-5" /> Add Language
                          </button>
                        </div>
                      )}
                    </SortableSection>
                  )
                })}

                {/* Export Buttons */}
                <div className="flex flex-wrap gap-3 pt-4 border-t">
                  <Button
                    onClick={() => downloadResumePDF(resumeData)}
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
            </SortableContext>
          </DndContext>
        ) : showPDFPreview ? (
          /* PDF Preview Mode */
          <PDFPreviewViewer resumeData={resumeData} />
        ) : (
          /* HTML Preview Mode */
          <ResumePreview resumeData={resumeData} />
        )}
      </Card>

      {/* Template Builder Modal */}
      {showTemplateBuilder && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b">
              <h2 className="text-xl font-semibold">Template Builder</h2>
              <button 
                onClick={() => {
                  setShowTemplateBuilder(false)
                  setActiveTool(null)
                }} 
                className="p-1 hover:bg-gray-100 rounded"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="overflow-y-auto max-h-[calc(90vh-80px)]">
              <TemplateBuilder onClose={() => {
                setShowTemplateBuilder(false)
                setActiveTool(null)
              }} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
