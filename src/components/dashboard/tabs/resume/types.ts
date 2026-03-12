export interface Experience {
  id: string
  company: string
  role: string
  startDate: string
  endDate: string
  current: boolean
  description: string
}

export interface Education {
  id: string
  institution: string
  degree: string
  field: string
  graduationDate: string
}

export interface Project {
  id: string
  name: string
  description: string
  technologies: string
}

export interface Certification {
  id: string
  name: string
  issuer: string
  date: string
  url: string
}

export interface Website {
  id: string
  name: string
  url: string
}

export interface Language {
  id: string
  name: string
  proficiency: string
}

export interface TemplateData {
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

export interface ResumeData {
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
  certifications: Certification[]
  websites: Website[]
  languages: Language[]
  isDefault?: boolean
  sectionOrder?: string[]
}
