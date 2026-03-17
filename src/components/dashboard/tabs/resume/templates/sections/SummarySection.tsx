import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface SummarySectionProps {
  data: ResumeData
  styles: any
  title?: string
}

export const SummarySection: React.FC<SummarySectionProps> = ({ data, styles, title = 'Professional Summary' }) => {
  if (!data.summary) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      <Text style={styles.description}>{data.summary}</Text>
    </View>
  )
}

export default SummarySection
