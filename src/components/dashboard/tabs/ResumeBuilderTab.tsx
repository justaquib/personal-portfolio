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
  Check, Copy, X, Upload, Award
} from 'lucide-react'
import { ResumePreview } from './resume/ResumePreview'
import { downloadResumePDF } from './resume/pdfGenerator'
import { ResumeData, Experience, Education, Project, Certification, Website, Language } from './resume/types'

// Template definitions
const TEMPLATES = [
  {
    id: 'modern',
    name: 'Modern',
    description: 'Clean and professional with a contemporary look',
    preview: 'bg-gradient-to-r from-gray-600 to-gray-800'
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

// Skill suggestions for resume
const SKILL_SUGGESTIONS = [
  // Programming Languages
  'JavaScript', 'TypeScript', 'Python', 'Java', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'PHP', 'Swift', 'Kotlin',
  // Frontend
  'React', 'Vue.js', 'Angular', 'Next.js', 'Nuxt.js', 'Svelte', 'HTML', 'CSS', 'Tailwind CSS', 'Sass',
  // Backend
  'Node.js', 'Express.js', 'Django', 'Flask', 'Spring Boot', 'Ruby on Rails', 'Laravel', 'ASP.NET',
  // Databases
  'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'SQLite',
  // Cloud & DevOps
  'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform', 'Jenkins', 'CI/CD',
  // Tools & Others
  'Git', 'GitHub', 'GitLab', 'Linux', 'REST API', 'GraphQL', 'Agile', 'Scrum', 'TDD'
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
  projects: [],
  certifications: [],
  websites: [],
  languages: []
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
  const { user } = useAuth()
  const [resumeData, setResumeData] = useState<ResumeData>(defaultResumeData)
  const [savedResumes, setSavedResumes] = useState<ResumeData[]>([])
  const [activeSection, setActiveSection] = useState<string | null>('personal')
  const [isGenerating, setIsGenerating] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)
  const [showSaveModal, setShowSaveModal] = useState(false)
  const [showResumesDropdown, setShowResumesDropdown] = useState(false)
  const [resumeName, setResumeName] = useState('My Resume')
  const [isSaving, setIsSaving] = useState(false)
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [aiSuggestions, setAiSuggestions] = useState<string[]>([])
  // PDF Import states
  const [isImportingPDF, setIsImportingPDF] = useState(false)
  const [importProgress, setImportProgress] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
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
      alert('Please select a PDF file')
      return
    }

    setIsImportingPDF(true)
    setImportProgress('Reading PDF file...')

    try {
      // Step 1: Extract text from PDF
      setImportProgress('Extracting text from PDF...')
      const extractedText = await extractTextFromPDF(file)

      if (!extractedText || extractedText.trim().length < 50) {
        alert('Could not extract enough text from the PDF. Please try a different file.')
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
        alert('Could not parse resume data. Please try again or enter manually.')
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
      
      alert('Resume imported successfully! Please review and edit the information below.')
    } catch (error) {
      console.error('Error importing PDF:', error)
      alert('Failed to import PDF. Please try again or enter your resume information manually.')
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

  const selectTemplate = (templateId: string) => {
    setResumeData(prev => ({ ...prev, template: templateId }))
    // Keep template selector open for user to close manually
  }

  // Enhance summary using Gemini AI - improves the existing summary
  const generateAISummary = async () => {
    // Expand summary section first if collapsed
    setActiveSection('summary')
    
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
        alert('Please add some content to your resume first (experience, education, skills, or projects) before enhancing the summary.')
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
        } else if (data.error) {
          alert('Error: ' + data.error)
        }
      } else {
        const errorData = await response.json()
        alert('Failed to enhance summary: ' + (errorData.error || 'Unknown error'))
      }
    } catch (error) {
      console.error('Error enhancing AI summary:', error)
      alert('Failed to enhance summary. Please try again.')
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
        alert(isUpdating ? 'Resume updated successfully!' : 'Resume saved successfully!')
      }
    } catch (error) {
      console.error('Error saving resume:', error)
      alert('Failed to save resume')
    } finally {
      setIsSaving(false)
    }
  }

  // Update saved resume
  const updateSavedResume = async () => {
    if (!user?.id || !resumeData.id) {
      alert('No resume selected to update')
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
        alert('Resume updated successfully!')
      }
    } catch (error) {
      console.error('Error updating resume:', error)
      alert('Failed to update resume')
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

  // Check if resume has any content to save
  const hasResumeContent = () => {
    return resumeData.personalInfo.name || 
           resumeData.personalInfo.email || 
           resumeData.personalInfo.phone || 
           resumeData.summary || 
           resumeData.experience.length > 0 || 
           resumeData.education.length > 0 || 
           resumeData.skills || 
           resumeData.projects.length > 0
  }

  // Check if resume is complete enough for export
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
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handlePDFImport}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isImportingPDF}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors"
            style={{ backgroundColor: '#e9ecef', color: '#212529' }}
          >
            {isImportingPDF ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Upload className="w-4 h-4" />
            )}
            {isImportingPDF ? 'Importing...' : 'Import from PDF'}
          </button>
          <button
            onClick={() => setShowTemplates(!showTemplates)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors"
            style={{ backgroundColor: '#e9ecef', color: '#212529' }}
          >
            <Layout className="w-4 h-4" />
            {selectedTemplate.name} Template
          </button>
          
          <button
            onClick={() => setShowSaveModal(true)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors"
            style={{ backgroundColor: '#e9ecef', color: '#212529' }}
          >
            <Save className="w-4 h-4" />
            {resumeData.id ? 'Update' : 'Save'}
          </button>

          {savedResumes.length > 0 && (
            <div className="relative">
              <button 
                onClick={() => setShowResumesDropdown(!showResumesDropdown)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors"
                style={{ backgroundColor: '#e9ecef', color: '#212529' }}
              >
                <FileText className="w-4 h-4" />
                My Resumes ({savedResumes.length})
                <ChevronDown className={`w-4 h-4 transition-transform ${showResumesDropdown ? 'rotate-180' : ''}`} />
              </button>
              
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
                          loadResume(resume)
                          setShowResumesDropdown(false)
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

          {/* Import Progress Indicator */}
          {isImportingPDF && (
            <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg text-blue-700">
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span className="text-sm">{importProgress}</span>
            </div>
          )}

          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => setShowPreview(false)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                !showPreview ? '' : ''
              }`}
              style={{
                backgroundColor: !showPreview ? '#212529' : '#e9ecef',
                color: !showPreview ? '#ffffff' : '#212529',
              }}
            >
              <Edit3 className="w-4 h-4" />
              Edit
            </button>
            <button
              onClick={() => setShowPreview(true)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                showPreview ? '' : ''
              }`}
              style={{
                backgroundColor: showPreview ? '#212529' : '#e9ecef',
                color: showPreview ? '#ffffff' : '#212529',
              }}
            >
              <Eye className="w-4 h-4" />
              Preview
            </button>
          </div>
        </div>

        {/* Template Selector Dropdown */}
        {showTemplates && (
          <div className="mb-6 p-4 bg-gray-50 rounded-xl">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-medium">Choose a Template</h3>
              <button 
                onClick={() => setShowTemplates(false)}
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

        {!showPreview ? (
          <div className="space-y-4">
            {/* Personal Information Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('personal')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <User className="w-5 h-5" style={{ color: '#212529' }} />
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
            </div>

            {/* Summary Section with AI */}
            <div className="border rounded-xl overflow-hidden">
              <div
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5" style={{ color: '#212529' }} />
                  <span className="font-medium">Professional Summary</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); generateAISummary(); }}
                    disabled={isLoadingAI}
                    className="flex items-center gap-1 px-2 py-1 text-xs rounded-lg transition-colors"
                    style={{ 
                      backgroundColor: isLoadingAI ? '#6c757d' : '#212529',
                      color: '#ffffff'
                    }}
                    title="Enhance with AI"
                  >
                    {isLoadingAI ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
                    Enhance
                  </button>
                </div>
                <button onClick={() => toggleSection('summary')} className="p-1">
                  {activeSection === 'summary' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </button>
              </div>
              
              {activeSection === 'summary' && (
                <div className="p-4">
                  <textarea
                    value={resumeData.summary}
                    onChange={(e) => setResumeData(prev => ({ ...prev, summary: e.target.value }))}
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                    style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529', minHeight: '120px' }}
                    placeholder="Write a brief summary of your professional background and career goals..."
                  />
                  {aiSuggestions.length > 0 && (
                    <div className="mt-3 p-3 bg-gray-100 rounded-lg">
                      <p className="text-sm text-gray-700 mb-2">AI Suggestion:</p>
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
                  <Briefcase className="w-5 h-5" style={{ color: '#212529' }} />
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
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Company Name"
                        />
                        <input
                          type="text"
                          value={exp.role}
                          onChange={(e) => updateExperience(exp.id, 'role', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Job Title"
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <input
                          type="text"
                          value={exp.startDate}
                          onChange={(e) => updateExperience(exp.id, 'startDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Start Date (e.g., Jan 2020)"
                        />
                        <input
                          type="text"
                          value={exp.endDate}
                          onChange={(e) => updateExperience(exp.id, 'endDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="End Date (e.g., Dec 2023)"
                          disabled={exp.current}
                        />
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={exp.current}
                            onChange={(e) => updateExperience(exp.id, 'current', e.target.checked)}
                            className="w-4 h-4 text-gray-600 rounded"
                          />
                          <span className="text-sm text-gray-700">Currently working</span>
                        </label>
                      </div>
                      <textarea
                        value={exp.description}
                        onChange={(e) => updateExperience(exp.id, 'description', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                        rows={3}
                        placeholder="Describe your responsibilities and achievements..."
                      />
                    </div>
                  ))}
                  <button
                    onClick={addExperience}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
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
                  <GraduationCap className="w-5 h-5" style={{ color: '#212529' }} />
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
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="University/College"
                        />
                        <input
                          type="text"
                          value={edu.graduationDate}
                          onChange={(e) => updateEducation(edu.id, 'graduationDate', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Graduation Date (e.g., May 2020)"
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={edu.degree}
                          onChange={(e) => updateEducation(edu.id, 'degree', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Degree (e.g., Bachelor of Science)"
                        />
                        <input
                          type="text"
                          value={edu.field}
                          onChange={(e) => updateEducation(edu.id, 'field', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Field of Study (e.g., Computer Science)"
                        />
                      </div>
                    </div>
                  ))}
                  <button
                    onClick={addEducation}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
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
                  <Code className="w-5 h-5" style={{ color: '#212529' }} />
                  <span className="font-medium">Skills</span>
                </div>
                {activeSection === 'skills' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'skills' && (
                <div className="p-4">
                  <textarea
                    value={resumeData.skills}
                    onChange={(e) => setResumeData(prev => ({ ...prev, skills: e.target.value }))}
                    className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                    rows={4}
                    placeholder="List your skills (e.g., JavaScript, Python, React, Node.js, AWS, Docker...)"
                  />
                  <p className="text-sm text-gray-500 mt-2">Separate skills with commas</p>
                  
                  {/* Skill Suggestions */}
                  <div className="mt-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Suggested Skills (click to add):</p>
                    <div className="flex flex-wrap gap-2">
                      {SKILL_SUGGESTIONS.map(skill => {
                        const isAdded = resumeData.skills.toLowerCase().split(',').map(s => s.trim()).includes(skill.toLowerCase())
                        return (
                          <button
                            key={skill}
                            onClick={() => {
                              const currentSkills = resumeData.skills ? resumeData.skills.split(',').map(s => s.trim()) : []
                              if (!currentSkills.map(s => s.toLowerCase()).includes(skill.toLowerCase())) {
                                const newSkills = [...currentSkills, skill].join(', ')
                                setResumeData(prev => ({ ...prev, skills: newSkills }))
                              }
                            }}
                            disabled={isAdded}
                            className={`px-3 py-1 rounded-full text-sm transition-colors ${
                              isAdded 
                                ? 'bg-green-100 text-green-700 cursor-default' 
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:text-gray-900'
                            }`}
                          >
                            {isAdded ? `✓ ${skill}` : skill}
                          </button>
                        )
                      })}
                    </div>
                  </div>
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
                  <Folder className="w-5 h-5" style={{ color: '#212529' }} />
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
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                        placeholder="Project Name"
                      />
                      <textarea
                        value={proj.description}
                        onChange={(e) => updateProject(proj.id, 'description', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                        rows={2}
                        placeholder="Project description..."
                      />
                      <input
                        type="text"
                        value={proj.technologies}
                        onChange={(e) => updateProject(proj.id, 'technologies', e.target.value)}
                        className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                        placeholder="Technologies used (e.g., React, Node.js, MongoDB)"
                      />
                    </div>
                  ))}
                  <button
                    onClick={addProject}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Project
                  </button>
                </div>
              )}
            </div>

            {/* Certifications Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('certifications')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Award className="w-5 h-5" style={{ color: '#212529' }} />
                  <span className="font-medium">Certifications</span>
                  <span className="text-sm text-gray-500">({resumeData.certifications.length})</span>
                </div>
                {activeSection === 'certifications' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'certifications' && (
                <div className="p-4 space-y-4">
                  {resumeData.certifications.map((cert, index) => (
                    <div key={cert.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Certification {index + 1}</span>
                        <button
                          onClick={() => removeCertification(cert.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={cert.name}
                          onChange={(e) => updateCertification(cert.id, 'name', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Certification Name"
                        />
                        <input
                          type="text"
                          value={cert.issuer}
                          onChange={(e) => updateCertification(cert.id, 'issuer', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Issuing Organization"
                        />
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={cert.date}
                          onChange={(e) => updateCertification(cert.id, 'date', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Date (e.g., Jan 2024)"
                        />
                        <input
                          type="text"
                          value={cert.url}
                          onChange={(e) => updateCertification(cert.id, 'url', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Certificate URL (optional)"
                        />
                      </div>
                    </div>
                  ))}
                  <button
                    onClick={addCertification}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Certification
                  </button>
                </div>
              )}
            </div>

            {/* Websites Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('websites')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5" style={{ color: '#212529' }} />
                  <span className="font-medium">Websites & Portfolio</span>
                  <span className="text-sm text-gray-500">({resumeData.websites.length})</span>
                </div>
                {activeSection === 'websites' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'websites' && (
                <div className="p-4 space-y-4">
                  {resumeData.websites.map((ws, index) => (
                    <div key={ws.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Website {index + 1}</span>
                        <button
                          onClick={() => removeWebsite(ws.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={ws.name}
                          onChange={(e) => updateWebsite(ws.id, 'name', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Website Name (e.g., GitHub, Blog)"
                        />
                        <input
                          type="text"
                          value={ws.url}
                          onChange={(e) => updateWebsite(ws.id, 'url', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="URL (e.g., github.com/johndoe)"
                        />
                      </div>
                    </div>
                  ))}
                  <button
                    onClick={addWebsite}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Website
                  </button>
                </div>
              )}
            </div>

            {/* Languages Section */}
            <div className="border rounded-xl overflow-hidden">
              <button
                onClick={() => toggleSection('languages')}
                className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Globe className="w-5 h-5" style={{ color: '#212529' }} />
                  <span className="font-medium">Languages</span>
                  <span className="text-sm text-gray-500">({resumeData.languages.length})</span>
                </div>
                {activeSection === 'languages' ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
              </button>
              
              {activeSection === 'languages' && (
                <div className="p-4 space-y-4">
                  {resumeData.languages.map((lang, index) => (
                    <div key={lang.id} className="p-4 bg-gray-50 rounded-lg space-y-3">
                      <div className="flex justify-between items-start">
                        <span className="text-sm font-medium text-gray-500">Language {index + 1}</span>
                        <button
                          onClick={() => removeLanguage(lang.id)}
                          className="p-1 text-red-500 hover:bg-red-50 rounded"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <input
                          type="text"
                          value={lang.name}
                          onChange={(e) => updateLanguage(lang.id, 'name', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                          placeholder="Language (e.g., English)"
                        />
                        <select
                          value={lang.proficiency}
                          onChange={(e) => updateLanguage(lang.id, 'proficiency', e.target.value)}
                          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-gray-500 focus:border-transparent focus:outline-none"
                        >
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
                  <button
                    onClick={addLanguage}
                    className="w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-500 hover:text-gray-600 transition-colors"
                  >
                    <Plus className="w-5 h-5" />
                    Add Language
                  </button>
                </div>
              )}
            </div>

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
        ) : (
          /* Preview Mode with Template Styling */
          <ResumePreview resumeData={resumeData} />
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
                    className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg"
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
