import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'
import RichText from '../RichText'

interface ExperienceSectionProps {
  data: ResumeData
  styles: any
  title?: string
}

export const ExperienceSection: React.FC<ExperienceSectionProps> = ({ 
  data, 
  styles, 
  title = 'Work Experience' 
}) => {
  if (!data.experience || data.experience.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      
      {data.experience.map((exp, index) => (
        <View key={exp.id || index} style={{ marginBottom: 12 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Text style={styles.jobTitle}>{exp.role}</Text>
            <Text style={styles.date}>
              {exp.startDate} - {exp.current ? 'Present' : exp.endDate}
            </Text>
          </View>
          <Text style={styles.company}>
            {exp.company}
            {exp.location && ` | ${exp.location}`}
          </Text>
          {exp.description && <RichText html={exp.description} />}
        </View>
      ))}
    </View>
  )
}

export default ExperienceSection
