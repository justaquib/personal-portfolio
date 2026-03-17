import React from 'react'
import { View, Text, StyleSheet } from '@react-pdf/renderer'
import { ResumeData } from '../types'
import { TemplateType } from './styles'

interface HeaderProps {
  data: ResumeData
  template: TemplateType
  styles: any
}

export const Header: React.FC<HeaderProps> = ({ data, template, styles }) => {
  const renderModernHeader = () => (
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
  )

  const renderClassicHeader = () => (
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
  )

  const renderMinimalHeader = () => (
    <View style={styles.header}>
      <Text style={styles.name}>{data.personalInfo.name || 'Your Name'}</Text>
      <View style={styles.contactRow}>
        {data.personalInfo.email && <Text>{data.personalInfo.email}</Text>}
        {data.personalInfo.phone && <Text>{data.personalInfo.phone}</Text>}
        {data.personalInfo.location && <Text>{data.personalInfo.location}</Text>}
        {data.personalInfo.linkedin && <Text>{data.personalInfo.linkedin}</Text>}
        {data.personalInfo.portfolio && <Text>{data.personalInfo.portfolio}</Text>}
      </View>
    </View>
  )

  const renderCreativeHeader = () => (
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
  )

  const renderHeader = () => {
    switch (template) {
      case 'classic':
        return renderClassicHeader()
      case 'minimal':
        return renderMinimalHeader()
      case 'creative':
        return renderCreativeHeader()
      case 'modern':
      default:
        return renderModernHeader()
    }
  }

  return <>{renderHeader()}</>
}

export default Header
