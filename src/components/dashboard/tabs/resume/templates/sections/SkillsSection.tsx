import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface SkillsSectionProps {
  data: ResumeData
  styles: any
  title?: string
  accentColor?: string
}

export const SkillsSection: React.FC<SkillsSectionProps> = ({ 
  data, 
  styles, 
  title = 'Skills',
  accentColor = '#9333ea'
}) => {
  if (!data.skills) return null

  const skills = data.skills.split(',').map(s => s.trim())

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {skills.map((skill, index) => (
          <View key={index} style={[styles.skillBadge, { backgroundColor: accentColor }]}>
            <Text style={styles.skillText}>{skill}</Text>
          </View>
        ))}
      </View>
    </View>
  )
}

export default SkillsSection
