import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface WebsitesSectionProps {
  data: ResumeData
  styles: any
  title?: string
}

export const WebsitesSection: React.FC<WebsitesSectionProps> = ({ 
  data, 
  styles, 
  title = 'Websites & Portfolio' 
}) => {
  if (!data.websites || data.websites.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      {data.websites.map((ws, index) => (
        <Text key={ws.id || index} style={{ fontSize: 9, marginBottom: 3, color: '#555555' }}>
          {ws.name}: {ws.url}
        </Text>
      ))}
    </View>
  )
}

export default WebsitesSection
