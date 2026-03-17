import React from 'react'
import {
  Document,
  Page,
  Text,
  View,
  StyleSheet,
  Font,
  PDFViewer,
  pdf,
  BlobProvider
} from '@react-pdf/renderer'
import { ResumeData, TemplateData } from './types'

// Register fonts
Font.register({
  family: 'Inter',
  fonts: [
    { src: 'https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjp-Ek-_EeA.woff2', fontWeight: 'normal' },
    { src: 'https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuI6fAZ9hjp-Ek-_EeA.woff2', fontWeight: 'bold' }
  ]
})

// PDF Styles based on template
const createStyles = (template: string, colors: { primary: string; accent: string }) => {
  const baseStyles = StyleSheet.create({
    page: {
      padding: 30,
      fontFamily: 'Inter',
      fontSize: 10,
      lineHeight: 1.5,
      backgroundColor: '#ffffff'
    },
    header: {
      marginBottom: 15
    },
    name: {
      fontSize: 24,
      fontWeight: 'bold',
      marginBottom: 5
    },
    contactRow: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 5,
      fontSize: 9,
      color: '#666666'
    },
    contactItem: {
      flexDirection: 'row',
      alignItems: 'center'
    },
    section: {
      marginBottom: 15
    },
    sectionTitle: {
      fontSize: 12,
      fontWeight: 'bold',
      marginBottom: 8,
      textTransform: 'uppercase',
      letterSpacing: 1
    },
    sectionDivider: {
      height: 1,
      marginBottom: 10
    },
    jobTitle: {
      fontSize: 11,
      fontWeight: 'bold'
    },
    company: {
      fontSize: 10,
      fontStyle: 'italic',
      color: '#666666'
    },
    date: {
      fontSize: 9,
      color: '#888888'
    },
    description: {
      fontSize: 10,
      color: '#444444',
      marginTop: 4,
      textAlign: 'justify'
    },
    skillBadge: {
      paddingHorizontal: 8,
      paddingVertical: 3,
      borderRadius: 3,
      marginRight: 5,
      marginBottom: 5
    },
    skillText: {
      fontSize: 9,
      color: '#ffffff'
    },
    projectName: {
      fontSize: 11,
      fontWeight: 'bold'
    },
    projectTech: {
      fontSize: 9,
      fontStyle: 'italic',
      color: '#666666',
      marginTop: 2
    },
    certBadge: {
      paddingHorizontal: 8,
      paddingVertical: 4,
      borderRadius: 12,
      marginRight: 6,
      marginBottom: 4
    },
    certText: {
      fontSize: 8,
      color: '#ffffff'
    },
    educationDegree: {
      fontSize: 11,
      fontWeight: 'bold'
    },
    educationSchool: {
      fontSize: 10,
      fontStyle: 'italic',
      color: '#666666'
    }
  })

  // Template-specific overrides
  const templateStyles: Record<string, any> = {
    modern: StyleSheet.create({
      ...baseStyles,
      header: {
        backgroundColor: colors.accent,
        padding: 15,
        marginHorizontal: -30,
        marginTop: -30,
        marginBottom: 20,
        paddingTop: 30
      },
      name: {
        ...baseStyles.name,
        color: '#ffffff'
      },
      contactRow: {
        ...baseStyles.contactRow,
        color: '#ffffff'
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        color: colors.accent
      },
      sectionDivider: {
        ...baseStyles.sectionDivider,
        backgroundColor: colors.accent
      }
    }),
    classic: StyleSheet.create({
      ...baseStyles,
      name: {
        ...baseStyles.name,
        textAlign: 'center',
        marginBottom: 10
      },
      contactRow: {
        ...baseStyles.contactRow,
        justifyContent: 'center',
        marginBottom: 15
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        borderBottomWidth: 1,
        borderBottomColor: '#333333',
        paddingBottom: 4
      }
    }),
    minimal: StyleSheet.create({
      ...baseStyles,
      name: {
        ...baseStyles.name,
        fontWeight: 'normal',
        fontSize: 28,
        color: colors.primary
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        color: colors.accent,
        fontSize: 11
      }
    }),
    creative: StyleSheet.create({
      ...baseStyles,
      header: {
        backgroundColor: colors.accent,
        padding: 20,
        marginHorizontal: -30,
        marginTop: -30,
        marginBottom: 20,
        paddingTop: 35,
        borderLeftWidth: 5,
        borderLeftColor: colors.primary
      },
      name: {
        ...baseStyles.name,
        color: '#ffffff',
        fontSize: 28
      },
      contactRow: {
        ...baseStyles.contactRow,
        color: '#ffffff'
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        color: colors.accent,
        backgroundColor: '#f5f5f5',
        padding: 8,
        marginHorizontal: -10,
        paddingHorizontal: 10
      }
    })
  }

  return templateStyles[template] || templateStyles.modern
}

// Color schemes for templates
const templateColors: Record<string, { primary: string; accent: string }> = {
  modern: { primary: '#1a1a1a', accent: '#9333ea' },
  classic: { primary: '#2d2d2d', accent: '#404040' },
  minimal: { primary: '#3b82f6', accent: '#60a5fa' },
  creative: { primary: '#ea580c', accent: '#f97316' }
}

// Modern Template
const ModernTemplate: React.FC<{ data: ResumeData }> = ({ data }) => {
  const colors = templateColors.modern
  const styles = createStyles('modern', colors)

  return (
    <Page size="A4" style={styles.page}>
      {/* Header with accent background */}
      <View style={styles.header}>
        <Text style={styles.name}>{data.personalInfo.name || 'Your Name'}</Text>
        <View style={styles.contactRow}>
          {data.personalInfo.email && <Text>{data.personalInfo.email}</Text>}
          {data.personalInfo.phone && <Text> | {data.personalInfo.phone}</Text>}
          {data.personalInfo.location && <Text> | {data.personalInfo.location}</Text>}
          {data.personalInfo.linkedin && <Text> | LinkedIn: {data.personalInfo.linkedin}</Text>}
          {data.personalInfo.portfolio && <Text> | Portfolio: {data.personalInfo.portfolio}</Text>}
        </View>
      </View>

      {/* Summary */}
      {data.summary && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Professional Summary</Text>
          <View style={styles.sectionDivider} />
          <Text style={styles.description}>{data.summary}</Text>
        </View>
      )}

      {/* Experience */}
      {data.experience.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Work Experience</Text>
          <View style={styles.sectionDivider} />
          {data.experience.map((exp, index) => (
            <View key={exp.id || index} style={{ marginBottom: 12 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.jobTitle}>{exp.role}</Text>
                <Text style={styles.date}>
                  {exp.startDate} - {exp.current ? 'Present' : exp.endDate}
                </Text>
              </View>
              <Text style={styles.company}>{exp.company} | {exp.location}</Text>
              <Text style={styles.description}>{exp.description}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Education */}
      {data.education.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Education</Text>
          <View style={styles.sectionDivider} />
          {data.education.map((edu, index) => (
            <View key={edu.id || index} style={{ marginBottom: 8 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.educationDegree}>{edu.degree} in {edu.field}</Text>
                <Text style={styles.date}>{edu.graduationDate}</Text>
              </View>
              <Text style={styles.educationSchool}>{edu.institution}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Skills */}
      {data.skills && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Skills</Text>
          <View style={styles.sectionDivider} />
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.skills.split(',').map((skill, index) => (
              <View key={index} style={[styles.skillBadge, { backgroundColor: colors.accent }]}>
                <Text style={styles.skillText}>{skill.trim()}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Projects */}
      {data.projects.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Projects</Text>
          <View style={styles.sectionDivider} />
          {data.projects.map((proj, index) => (
            <View key={proj.id || index} style={{ marginBottom: 10 }}>
              <Text style={styles.projectName}>{proj.name}</Text>
              <Text style={styles.description}>{proj.description}</Text>
              {proj.technologies && <Text style={styles.projectTech}>Technologies: {proj.technologies}</Text>}
            </View>
          ))}
        </View>
      )}

      {/* Certifications */}
      {data.certifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Certifications</Text>
          <View style={styles.sectionDivider} />
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.certifications.map((cert, index) => (
              <View key={cert.id || index} style={[styles.certBadge, { backgroundColor: colors.primary }]}>
                <Text style={styles.certText}>{cert.name}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Languages */}
      {data.languages.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Languages</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.languages.map((lang, index) => (
              <Text key={lang.id || index} style={{ marginRight: 15, fontSize: 10 }}>
                {lang.name} - {lang.proficiency}
              </Text>
            ))}
          </View>
        </View>
      )}

      {/* Websites */}
      {data.websites.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Websites & Portfolio</Text>
          {data.websites.map((ws, index) => (
            <Text key={ws.id || index} style={{ fontSize: 9, marginBottom: 2 }}>
              {ws.name}: {ws.url}
            </Text>
          ))}
        </View>
      )}
    </Page>
  )
}

// Classic Template
const ClassicTemplate: React.FC<{ data: ResumeData }> = ({ data }) => {
  const colors = templateColors.classic
  const styles = createStyles('classic', colors)

  return (
    <Page size="A4" style={styles.page}>
      {/* Centered Header */}
      <View style={styles.header}>
        <Text style={styles.name}>{data.personalInfo.name || 'Your Name'}</Text>
        <View style={styles.contactRow}>
          {data.personalInfo.email && <Text>{data.personalInfo.email}</Text>}
          {data.personalInfo.phone && <Text> | {data.personalInfo.phone}</Text>}
          {data.personalInfo.location && <Text> | {data.personalInfo.location}</Text>}
        </View>
        <View style={styles.contactRow}>
          {data.personalInfo.linkedin && <Text>LinkedIn: {data.personalInfo.linkedin}</Text>}
          {data.personalInfo.portfolio && <Text> | Portfolio: {data.personalInfo.portfolio}</Text>}
        </View>
      </View>

      {/* Summary */}
      {data.summary && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PROFESSIONAL SUMMARY</Text>
          <Text style={styles.description}>{data.summary}</Text>
        </View>
      )}

      {/* Experience */}
      {data.experience.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>WORK EXPERIENCE</Text>
          {data.experience.map((exp, index) => (
            <View key={exp.id || index} style={{ marginBottom: 12 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.jobTitle}>{exp.role}</Text>
                <Text style={styles.date}>{exp.startDate} - {exp.current ? 'Present' : exp.endDate}</Text>
              </View>
              <Text style={styles.company}>{exp.company}, {exp.location}</Text>
              <Text style={styles.description}>{exp.description}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Education */}
      {data.education.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>EDUCATION</Text>
          {data.education.map((edu, index) => (
            <View key={edu.id || index} style={{ marginBottom: 8 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.educationDegree}>{edu.degree} in {edu.field}</Text>
                <Text style={styles.date}>{edu.graduationDate}</Text>
              </View>
              <Text style={styles.educationSchool}>{edu.institution}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Skills */}
      {data.skills && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>SKILLS</Text>
          <Text style={styles.description}>{data.skills}</Text>
        </View>
      )}

      {/* Projects */}
      {data.projects.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>PROJECTS</Text>
          {data.projects.map((proj, index) => (
            <View key={proj.id || index} style={{ marginBottom: 8 }}>
              <Text style={styles.projectName}>{proj.name}</Text>
              <Text style={styles.description}>{proj.description}</Text>
              {proj.technologies && <Text style={styles.projectTech}>Technologies: {proj.technologies}</Text>}
            </View>
          ))}
        </View>
      )}

      {/* Certifications */}
      {data.certifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>CERTIFICATIONS</Text>
          <Text style={styles.description}>
            {data.certifications.map(c => `${c.name} (${c.issuer}) - ${c.date}`).join(' | ')}
          </Text>
        </View>
      )}

      {/* Languages */}
      {data.languages.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>LANGUAGES</Text>
          <Text style={styles.description}>
            {data.languages.map(l => `${l.name}: ${l.proficiency}`).join(' | ')}
          </Text>
        </View>
      )}
    </Page>
  )
}

// Minimal Template
const MinimalTemplate: React.FC<{ data: ResumeData }> = ({ data }) => {
  const colors = templateColors.minimal
  const styles = createStyles('minimal', colors)

  return (
    <Page size="A4" style={styles.page}>
      {/* Clean Header */}
      <View style={{ marginBottom: 20 }}>
        <Text style={styles.name}>{data.personalInfo.name || 'Your Name'}</Text>
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 5 }}>
          {data.personalInfo.email && <Text style={{ fontSize: 9 }}>{data.personalInfo.email}</Text>}
          {data.personalInfo.phone && <Text style={{ fontSize: 9 }}>{data.personalInfo.phone}</Text>}
          {data.personalInfo.location && <Text style={{ fontSize: 9 }}>{data.personalInfo.location}</Text>}
          {data.personalInfo.linkedin && <Text style={{ fontSize: 9 }}>{data.personalInfo.linkedin}</Text>}
          {data.personalInfo.portfolio && <Text style={{ fontSize: 9 }}>{data.personalInfo.portfolio}</Text>}
        </View>
      </View>

      {/* Summary */}
      {data.summary && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About</Text>
          <Text style={{ fontSize: 10, color: '#555555', lineHeight: 1.6 }}>{data.summary}</Text>
        </View>
      )}

      {/* Experience */}
      {data.experience.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Experience</Text>
          {data.experience.map((exp, index) => (
            <View key={exp.id || index} style={{ marginBottom: 15 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 2 }}>
                <Text style={{ fontSize: 11, fontWeight: 'bold' }}>{exp.role}</Text>
                <Text style={{ fontSize: 9, color: '#888' }}>
                  {exp.startDate} - {exp.current ? 'Present' : exp.endDate}
                </Text>
              </View>
              <Text style={{ fontSize: 10, color: '#666' }}>{exp.company}</Text>
              <Text style={{ fontSize: 10, color: '#555', marginTop: 4 }}>{exp.description}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Education */}
      {data.education.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Education</Text>
          {data.education.map((edu, index) => (
            <View key={edu.id || index} style={{ marginBottom: 10 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={{ fontSize: 10, fontWeight: 'bold' }}>{edu.degree} in {edu.field}</Text>
                <Text style={{ fontSize: 9, color: '#888' }}>{edu.graduationDate}</Text>
              </View>
              <Text style={{ fontSize: 10, color: '#666' }}>{edu.institution}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Skills */}
      {data.skills && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Skills</Text>
          <Text style={{ fontSize: 10, color: '#555' }}>{data.skills}</Text>
        </View>
      )}

      {/* Projects */}
      {data.projects.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Projects</Text>
          {data.projects.map((proj, index) => (
            <View key={proj.id || index} style={{ marginBottom: 10 }}>
              <Text style={{ fontSize: 10, fontWeight: 'bold' }}>{proj.name}</Text>
              <Text style={{ fontSize: 10, color: '#555' }}>{proj.description}</Text>
              {proj.technologies && <Text style={{ fontSize: 9, color: '#888' }}>{proj.technologies}</Text>}
            </View>
          ))}
        </View>
      )}

      {/* Certifications */}
      {data.certifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Certifications</Text>
          {data.certifications.map((cert, index) => (
            <Text key={cert.id || index} style={{ fontSize: 10, marginBottom: 3 }}>
              {cert.name} - {cert.issuer} ({cert.date})
            </Text>
          ))}
        </View>
      )}
    </Page>
  )
}

// Creative Template
const CreativeTemplate: React.FC<{ data: ResumeData }> = ({ data }) => {
  const colors = templateColors.creative
  const styles = createStyles('creative', colors)

  return (
    <Page size="A4" style={styles.page}>
      {/* Bold Header */}
      <View style={styles.header}>
        <Text style={styles.name}>{data.personalInfo.name || 'Your Name'}</Text>
        <View style={styles.contactRow}>
          {data.personalInfo.email && <Text>{data.personalInfo.email}</Text>}
          {data.personalInfo.phone && <Text> | {data.personalInfo.phone}</Text>}
          {data.personalInfo.location && <Text> | {data.personalInfo.location}</Text>}
        </View>
        <View style={styles.contactRow}>
          {data.personalInfo.linkedin && <Text>{data.personalInfo.linkedin}</Text>}
          {data.personalInfo.portfolio && <Text> | {data.personalInfo.portfolio}</Text>}
        </View>
      </View>

      {/* Summary */}
      {data.summary && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <Text style={styles.description}>{data.summary}</Text>
        </View>
      )}

      {/* Experience */}
      {data.experience.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Experience</Text>
          {data.experience.map((exp, index) => (
            <View key={exp.id || index} style={{ marginBottom: 12 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.jobTitle}>{exp.role}</Text>
                <Text style={styles.date}>
                  {exp.startDate} - {exp.current ? 'Present' : exp.endDate}
                </Text>
              </View>
              <Text style={styles.company}>{exp.company}</Text>
              <Text style={styles.description}>{exp.description}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Education */}
      {data.education.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Education</Text>
          {data.education.map((edu, index) => (
            <View key={edu.id || index} style={{ marginBottom: 8 }}>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Text style={styles.educationDegree}>{edu.degree} in {edu.field}</Text>
                <Text style={styles.date}>{edu.graduationDate}</Text>
              </View>
              <Text style={styles.educationSchool}>{edu.institution}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Skills */}
      {data.skills && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Skills</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.skills.split(',').map((skill, index) => (
              <View key={index} style={[styles.skillBadge, { backgroundColor: colors.accent }]}>
                <Text style={styles.skillText}>{skill.trim()}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Projects */}
      {data.projects.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Projects</Text>
          {data.projects.map((proj, index) => (
            <View key={proj.id || index} style={{ marginBottom: 10 }}>
              <Text style={styles.projectName}>{proj.name}</Text>
              <Text style={styles.description}>{proj.description}</Text>
              {proj.technologies && <Text style={styles.projectTech}>{proj.technologies}</Text>}
            </View>
          ))}
        </View>
      )}

      {/* Certifications */}
      {data.certifications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Certifications</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.certifications.map((cert, index) => (
              <View key={cert.id || index} style={[styles.certBadge, { backgroundColor: colors.accent }]}>
                <Text style={styles.certText}>{cert.name}</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Languages */}
      {data.languages.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Languages</Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {data.languages.map((lang, index) => (
              <View key={lang.id || index} style={[styles.certBadge, { backgroundColor: colors.primary }]}>
                <Text style={styles.certText}>{lang.name}: {lang.proficiency}</Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </Page>
  )
}

// Main Resume Document
interface ResumeDocumentProps {
  data: ResumeData
  template?: string
}

export const ResumeDocument: React.FC<ResumeDocumentProps> = ({ data, template = 'modern' }) => {
  const renderTemplate = () => {
    switch (template) {
      case 'classic':
        return <ClassicTemplate data={data} />
      case 'minimal':
        return <MinimalTemplate data={data} />
      case 'creative':
        return <CreativeTemplate data={data} />
      case 'modern':
      default:
        return <ModernTemplate data={data} />
    }
  }

  return (
    <Document>
      {renderTemplate()}
    </Document>
  )
}

// PDF Viewer Component for preview
interface PDFPreviewProps {
  data: ResumeData
  template?: string
}

export const PDFPreview: React.FC<PDFPreviewProps> = ({ data, template = 'modern' }) => {
  return (
    <PDFViewer width="100%" height="100%" style={{ border: 'none' }}>
      <ResumeDocument data={data} template={template} />
    </PDFViewer>
  )
}

// Generate PDF as blob
export const generatePDFBlob = async (data: ResumeData, template: string = 'modern') => {
  const blob = await pdf(<ResumeDocument data={data} template={template} />).toBlob()
  return blob
}

// Generate PDF as array buffer and convert to base64
export const generatePDFBase64 = async (data: ResumeData, template: string = 'modern') => {
  const blob = await generatePDFBlob(data, template)
  const arrayBuffer = await blob.arrayBuffer()
  const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)))
  return base64
}

// Download PDF
export const downloadResumePDF = async (data: ResumeData, template: string = 'modern') => {
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

export default ResumeDocument
