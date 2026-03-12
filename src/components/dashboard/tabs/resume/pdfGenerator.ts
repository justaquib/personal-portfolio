import { jsPDF } from 'jspdf'
import { ResumeData } from './types'

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

export function generateResumePDF(resumeData: ResumeData): jsPDF {
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

  const primaryColor = colors[resumeData.template] || colors.modern
  const accentColor = accentColors[resumeData.template] || accentColors.modern

  // Template-specific header
  if (resumeData.template === 'modern') {
    doc.setFillColor(...accentColor)
    doc.rect(0, 0, pageWidth, 35, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(24)
    doc.setFont('helvetica', 'bold')
    doc.text(resumeData.personalInfo.name || 'Your Name', margin, 22)
    y = 45
  } else if (resumeData.template === 'classic') {
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
    doc.setFontSize(28)
    doc.setFont('helvetica', 'light')
    doc.setTextColor(...primaryColor)
    doc.text(resumeData.personalInfo.name || 'Your Name', margin, y + 5)
    y += 18
  } else if (resumeData.template === 'creative') {
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

  // Certifications - Display as badges/pills
  if (resumeData.certifications.length > 0) {
    if (y > pageHeight - 40) {
      doc.addPage()
      y = margin
    }

    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...primaryColor)
    doc.text('CERTIFICATIONS', margin, y)
    y += 6

    let badgeX = margin
    const badgeHeight = 6
    const badgePadding = 3
    const badgeSpacing = 4

    resumeData.certifications.forEach(cert => {
      const certText = cert.issuer ? `${cert.name} (${cert.issuer})` : cert.name
      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      const textWidth = doc.getTextWidth(certText)
      const badgeWidth = textWidth + badgePadding * 2

      if (badgeX + badgeWidth > pageWidth - margin) {
        badgeX = margin
        y += badgeHeight + badgeSpacing
      }

      doc.setFillColor(...accentColor)
      doc.roundedRect(badgeX, y, badgeWidth, badgeHeight, 2, 2, 'F')

      doc.setTextColor(255, 255, 255)
      doc.text(certText, badgeX + badgePadding, y + 4.5)

      badgeX += badgeWidth + badgeSpacing
    })
    y += badgeHeight + 10
  }

  // Languages - Display as badges/pills
  if (resumeData.languages.length > 0) {
    if (y > pageHeight - 40) {
      doc.addPage()
      y = margin
    }

    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...primaryColor)
    doc.text('LANGUAGES', margin, y)
    y += 6

    let badgeX = margin
    const badgeHeight = 6
    const badgePadding = 3
    const badgeSpacing = 4

    resumeData.languages.forEach(lang => {
      const langText = `${lang.name} - ${lang.proficiency}`
      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      const textWidth = doc.getTextWidth(langText)
      const badgeWidth = textWidth + badgePadding * 2

      if (badgeX + badgeWidth > pageWidth - margin) {
        badgeX = margin
        y += badgeHeight + badgeSpacing
      }

      doc.setFillColor(...primaryColor)
      doc.roundedRect(badgeX, y, badgeWidth, badgeHeight, 2, 2, 'F')

      doc.setTextColor(255, 255, 255)
      doc.text(langText, badgeX + badgePadding, y + 4.5)

      badgeX += badgeWidth + badgeSpacing
    })
    y += badgeHeight + 10
  }

  // Websites - Display as badges/pills
  if (resumeData.websites.length > 0) {
    if (y > pageHeight - 40) {
      doc.addPage()
      y = margin
    }

    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...primaryColor)
    doc.text('WEBSITES & PORTFOLIO', margin, y)
    y += 6

    let badgeX = margin
    const badgeHeight = 6
    const badgePadding = 3
    const badgeSpacing = 4

    resumeData.websites.forEach(ws => {
      const wsText = `${ws.name}: ${ws.url}`
      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      const textWidth = doc.getTextWidth(wsText)
      const badgeWidth = textWidth + badgePadding * 2

      if (badgeX + badgeWidth > pageWidth - margin) {
        badgeX = margin
        y += badgeHeight + badgeSpacing
      }

      doc.setFillColor(100, 100, 100)
      doc.roundedRect(badgeX, y, badgeWidth, badgeHeight, 2, 2, 'F')

      doc.setTextColor(255, 255, 255)
      doc.text(wsText, badgeX + badgePadding, y + 4.5)

      badgeX += badgeWidth + badgeSpacing
    })
    y += badgeHeight + 10
  }

  return doc
}

export function downloadResumePDF(resumeData: ResumeData): void {
  const doc = generateResumePDF(resumeData)
  
  const fileName = resumeData.personalInfo.name 
    ? `${resumeData.personalInfo.name.replace(/\s+/g, '_')}_Resume.pdf`
    : 'Resume.pdf'
  
  doc.save(fileName)
}
