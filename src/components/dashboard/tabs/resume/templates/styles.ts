import { StyleSheet } from '@react-pdf/renderer'

// Template color schemes
export const templateColors: Record<string, { primary: string; accent: string }> = {
  modern: { primary: '#1a1a1a', accent: '#9333ea' },
  classic: { primary: '#2d2d2d', accent: '#404040' },
  minimal: { primary: '#3b82f6', accent: '#60a5fa' },
  creative: { primary: '#ea580c', accent: '#f97316' }
}

// Template type
export type TemplateType = 'modern' | 'classic' | 'minimal' | 'creative'

// Base styles for all templates
export const baseStyles = StyleSheet.create({
  page: {
    padding: 30,
    fontFamily: 'Helvetica',
    fontSize: 10,
    lineHeight: 1.5,
    backgroundColor: '#ffffff'
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

// Create template-specific styles
export const createTemplateStyles = (template: TemplateType) => {
  const colors = templateColors[template]

  const styles: Record<string, any> = {
    // Modern template
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
        fontSize: 24,
        fontWeight: 'bold',
        color: '#ffffff',
        marginBottom: 5
      },
      contactRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 5,
        fontSize: 9,
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

    // Classic template
    classic: StyleSheet.create({
      ...baseStyles,
      header: {
        marginBottom: 15
      },
      name: {
        fontSize: 22,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 10,
        color: colors.primary
      },
      contactRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: 5,
        fontSize: 9,
        color: '#666666',
        marginBottom: 15
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        borderBottomWidth: 1,
        borderBottomColor: '#333333',
        paddingBottom: 4,
        color: colors.primary
      }
    }),

    // Minimal template
    minimal: StyleSheet.create({
      ...baseStyles,
      header: {
        marginBottom: 20
      },
      name: {
        fontSize: 28,
        fontWeight: 'normal',
        color: colors.primary,
        marginBottom: 5
      },
      contactRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
        fontSize: 9,
        color: '#666666'
      },
      sectionTitle: {
        ...baseStyles.sectionTitle,
        color: colors.accent,
        fontSize: 11
      }
    }),

    // Creative template
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
        fontSize: 28,
        fontWeight: 'bold',
        color: '#ffffff',
        marginBottom: 5
      },
      contactRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 5,
        fontSize: 9,
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

  return styles[template] || styles.modern
}

export default { templateColors, baseStyles, createTemplateStyles }
