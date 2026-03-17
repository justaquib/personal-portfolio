import React from 'react'
import { Document, Page, PDFViewer, pdf } from '@react-pdf/renderer'
import { ResumeData } from '../types'
import { TemplateType, templateColors, createTemplateStyles } from './styles'
import Header from './Header'
import Footer from './Footer'

// Import all sections
import {
  SummarySection,
  ExperienceSection,
  EducationSection,
  SkillsSection,
  ProjectsSection,
  CertificationsSection,
  LanguagesSection,
  WebsitesSection
} from './sections'

// Base Template Props
interface BaseTemplateProps {
  data: ResumeData
  template: TemplateType
  showFooter?: boolean
}

// Modern Template
const ModernTemplate: React.FC<BaseTemplateProps> = ({ data, showFooter = false }) => {
  const styles = createTemplateStyles('modern')
  const colors = templateColors.modern

  return (
    <Page size="A4" style={styles.page}>
      <Header data={data} template="modern" styles={styles} />
      
      <SummarySection data={data} styles={styles} />
      <ExperienceSection data={data} styles={styles} />
      <EducationSection data={data} styles={styles} />
      <SkillsSection data={data} styles={styles} accentColor={colors.accent} />
      <ProjectsSection data={data} styles={styles} />
      <CertificationsSection data={data} styles={styles} badgeColor={colors.primary} />
      <LanguagesSection data={data} styles={styles} badgeColor={colors.primary} />
      <WebsitesSection data={data} styles={styles} />

      {showFooter && <Footer pageNumber={1} totalPages={1} />}
    </Page>
  )
}

// Classic Template
const ClassicTemplate: React.FC<BaseTemplateProps> = ({ data, showFooter = false }) => {
  const styles = createTemplateStyles('classic')
  const colors = templateColors.classic

  return (
    <Page size="A4" style={styles.page}>
      <Header data={data} template="classic" styles={styles} />
      
      <SummarySection data={data} styles={styles} title="PROFESSIONAL SUMMARY" />
      <ExperienceSection data={data} styles={styles} title="WORK EXPERIENCE" />
      <EducationSection data={data} styles={styles} title="EDUCATION" />
      <SkillsSection data={data} styles={styles} title="SKILLS" />
      <ProjectsSection data={data} styles={styles} title="PROJECTS" />
      <CertificationsSection data={data} styles={styles} title="CERTIFICATIONS" badgeColor={colors.accent} />
      <LanguagesSection data={data} styles={styles} title="LANGUAGES" badgeColor={colors.accent} />
      <WebsitesSection data={data} styles={styles} title="WEBSITES" />

      {showFooter && <Footer pageNumber={1} totalPages={1} />}
    </Page>
  )
}

// Minimal Template
const MinimalTemplate: React.FC<BaseTemplateProps> = ({ data, showFooter = false }) => {
  const styles = createTemplateStyles('minimal')
  const colors = templateColors.minimal

  return (
    <Page size="A4" style={styles.page}>
      <Header data={data} template="minimal" styles={styles} />
      
      <SummarySection data={data} styles={styles} title="About" />
      <ExperienceSection data={data} styles={styles} title="Experience" />
      <EducationSection data={data} styles={styles} title="Education" />
      <SkillsSection data={data} styles={styles} title="Skills" accentColor={colors.accent} />
      <ProjectsSection data={data} styles={styles} title="Projects" />
      <CertificationsSection data={data} styles={styles} title="Certifications" badgeColor={colors.accent} />
      <LanguagesSection data={data} styles={styles} title="Languages" badgeColor={colors.accent} />
      <WebsitesSection data={data} styles={styles} />

      {showFooter && <Footer pageNumber={1} totalPages={1} />}
    </Page>
  )
}

// Creative Template
const CreativeTemplate: React.FC<BaseTemplateProps> = ({ data, showFooter = false }) => {
  const styles = createTemplateStyles('creative')
  const colors = templateColors.creative

  return (
    <Page size="A4" style={styles.page}>
      <Header data={data} template="creative" styles={styles} />
      
      <SummarySection data={data} styles={styles} title="Summary" />
      <ExperienceSection data={data} styles={styles} title="Experience" />
      <EducationSection data={data} styles={styles} title="Education" />
      <SkillsSection data={data} styles={styles} title="Skills" accentColor={colors.accent} />
      <ProjectsSection data={data} styles={styles} title="Projects" />
      <CertificationsSection data={data} styles={styles} title="Certifications" badgeColor={colors.accent} />
      <LanguagesSection data={data} styles={styles} title="Languages" badgeColor={colors.primary} />
      <WebsitesSection data={data} styles={styles} />

      {showFooter && <Footer pageNumber={1} totalPages={1} />}
    </Page>
  )
}

// Main Resume Document
interface ResumeDocumentProps {
  data: ResumeData
  template?: TemplateType
}

export const ResumeDocument: React.FC<ResumeDocumentProps> = ({ data, template = 'modern' }) => {
  const renderTemplate = () => {
    switch (template) {
      case 'classic':
        return <ClassicTemplate data={data} template="classic" />
      case 'minimal':
        return <MinimalTemplate data={data} template="minimal" />
      case 'creative':
        return <CreativeTemplate data={data} template="creative" />
      case 'modern':
      default:
        return <ModernTemplate data={data} template="modern" />
    }
  }

  return <Document>{renderTemplate()}</Document>
}

// PDF Preview Component
interface PDFPreviewProps {
  data: ResumeData
  template?: TemplateType
}

export const PDFPreview: React.FC<PDFPreviewProps> = ({ data, template = 'modern' }) => {
  return (
    <PDFViewer width="100%" height="100%" style={{ border: 'none' }}>
      <ResumeDocument data={data} template={template} />
    </PDFViewer>
  )
}

// Generate PDF as Blob
export const generatePDFBlob = async (data: ResumeData, template: TemplateType = 'modern') => {
  const blob = await pdf(<ResumeDocument data={data} template={template} />).toBlob()
  return blob
}

// Generate PDF as Base64
export const generatePDFBase64 = async (data: ResumeData, template: TemplateType = 'modern') => {
  const blob = await generatePDFBlob(data, template)
  const arrayBuffer = await blob.arrayBuffer()
  const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)))
  return base64
}

// Download PDF
export const downloadResumePDF = async (data: ResumeData, template: TemplateType = 'modern') => {
  const blob = await generatePDFBlob(data, template)
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = data.personalInfo.name
    ? `${data.personalInfo.name.replace(/\s+/g, '_')}_Resume.pdf`
    : 'Resume.pdf'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// Export template types
export type { TemplateType }
export { templateColors, createTemplateStyles }

// Export individual components for customization
export { default as Header } from './Header'
export { default as Footer } from './Footer'

// Export sections
export {
  SummarySection,
  ExperienceSection,
  EducationSection,
  SkillsSection,
  ProjectsSection,
  CertificationsSection,
  LanguagesSection,
  WebsitesSection
} from './sections'

export default ResumeDocument
