import React from 'react'
import { View, Text } from '@react-pdf/renderer'
import { ResumeData } from '../../types'

interface CertificationsSectionProps {
  data: ResumeData
  styles: any
  title?: string
  badgeColor?: string
}

export const CertificationsSection: React.FC<CertificationsSectionProps> = ({ 
  data, 
  styles, 
  title = 'Certifications',
  badgeColor = '#1a1a1a'
}) => {
  if (!data.certifications || data.certifications.length === 0) return null

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {styles.sectionDivider && <View style={styles.sectionDivider} />}
      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {data.certifications.map((cert, index) => (
          <View key={cert.id || index} style={[styles.certBadge, { backgroundColor: badgeColor }]}>
            <Text style={styles.certText}>
              {cert.name}
              {cert.issuer && ` (${cert.issuer})`}
            </Text>
          </View>
        ))}
      </View>
    </View>
  )
}

export default CertificationsSection
