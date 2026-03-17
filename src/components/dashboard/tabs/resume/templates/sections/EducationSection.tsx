import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface EducationSectionProps {
  data: ResumeData
  styles: any
  title?: string
}

export const EducationSection: React.FC<EducationSectionProps> = ({ 
  data, 
  styles, 
  title = 'Education' 
}) => {
  if (!data.education || data.education.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      
      {data.education.map((edu, index) => (
        <View key={edu.id || index} style={{ marginBottom: 8 }}>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
            <Text style={styles.educationDegree}>
              {edu.degree}
              {edu.field && ` in ${edu.field}`}
            </Text>
            <Text style={styles.date}>{edu.graduationDate}</Text>
          </View>
          <Text style={styles.educationSchool}>{edu.institution}</Text>
        </View>
      ))}
    </View>
  )
}

export default EducationSection
