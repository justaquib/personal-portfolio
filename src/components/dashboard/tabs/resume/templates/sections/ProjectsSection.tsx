import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'
import RichText from '../RichText'

interface ProjectsSectionProps {
  data: ResumeData
  styles: any
  title?: string
}

export const ProjectsSection: React.FC<ProjectsSectionProps> = ({ 
  data, 
  styles, 
  title = 'Projects' 
}) => {
  if (!data.projects || data.projects.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      
      {data.projects.map((proj, index) => (
        <View key={proj.id || index} style={{ marginBottom: 10 }}>
          <Text style={styles.projectName}>{proj.name}</Text>
          {proj.description && <RichText html={proj.description} />}
          {proj.technologies && <Text style={styles.projectTech}>Technologies: {proj.technologies}</Text>}
        </View>
      ))}
    </View>
  )
}

export default ProjectsSection
