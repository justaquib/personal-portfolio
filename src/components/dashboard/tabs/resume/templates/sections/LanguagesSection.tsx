import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface LanguagesSectionProps {
  data: ResumeData
  styles: any
  title?: string
  badgeColor?: string
}

export const LanguagesSection: React.FC<LanguagesSectionProps> = ({ 
  data, 
  styles, 
  title = 'Languages',
  badgeColor = '#1a1a1a'
}) => {
  if (!data.languages || data.languages.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {data.languages.map((lang, index) => (
          <View key={lang.id || index} style={[styles.certBadge, { backgroundColor: badgeColor }]}>
            <Text style={styles.certText}>{lang.name} - {lang.proficiency}</Text>
          </View>
        ))}
      </View>
    </View>
  )
}

export default LanguagesSection
