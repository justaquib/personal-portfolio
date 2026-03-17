import React from 'react'
import { Text, View, StyleSheet } from '@react-pdf/renderer'

// Styles for rich text elements
const styles = StyleSheet.create({
  p: {
    marginBottom: 8,
    lineHeight: 1.5,
    fontSize: 10,
    color: '#333333'
  },
  h1: { fontSize: 18, fontWeight: 'bold', marginBottom: 10, marginTop: 12 },
  h2: { fontSize: 14, fontWeight: 'bold', marginBottom: 8, marginTop: 10 },
  h3: { fontSize: 12, fontWeight: 'bold', marginBottom: 6, marginTop: 8 },
  ul: { marginLeft: 15, marginBottom: 8 },
  ol: { marginLeft: 15, marginBottom: 8 },
  li: { marginBottom: 4 },
  code: { fontFamily: 'Courier', fontSize: 9, backgroundColor: '#f5f5f5', padding: 2 },
  strong: { fontWeight: 'bold' },
  em: { fontStyle: 'italic' },
  a: { color: '#0066cc', textDecoration: 'underline' },
  blockquote: { borderLeftWidth: 3, borderLeftColor: '#ccc', paddingLeft: 10, fontStyle: 'italic' },
  div: { marginBottom: 6 },
  small: { fontSize: 8 },
  del: { textDecoration: 'line-through' }
})

interface RichTextProps {
  html: string
}

// Unescape HTML entities
const unescape = (str: string): string => {
  if (!str) return ''
  return str
    .replace(/&nbsp;/g, ' ')
    .replace(/&/g, '&')
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, "'")
    .replace(/&mdash;/g, '—')
    .replace(/&ndash;/g, '–')
}

// Process inline tags
const processInline = (text: string): React.ReactNode => {
  if (!text) return null
  
  // Handle code
  const codeRe = /<code[^>]*>([\s\S]*?)<\/code>/gi
  if (codeRe.test(text)) {
    const parts: React.ReactNode[] = []
    let last = 0
    let m
    while ((m = codeRe.exec(text)) !== null) {
      if (m.index > last) parts.push(unescape(text.slice(last, m.index)))
      parts.push(<Text key={m.index} style={styles.code}>{unescape(m[1])}</Text>)
      last = m.index + m[0].length
    }
    if (last < text.length) parts.push(unescape(text.slice(last)))
    return parts.length ? parts : text
  }
  
  // Handle other inline
  const inlineRe = /<(strong|b|i|em|u|del|a|small)[^>]*>([\s\S]*?)<\/\1>/gi
  if (inlineRe.test(text)) {
    const parts: React.ReactNode[] = []
    let last = 0
    let m
    while ((m = inlineRe.exec(text)) !== null) {
      if (m.index > last) parts.push(unescape(text.slice(last, m.index)))
      const tag = m[1].toLowerCase()
      const content = m[2]
      if (tag === 'strong' || tag === 'b') parts.push(<Text key={m.index} style={styles.strong}>{processInline(content)}</Text>)
      else if (tag === 'em' || tag === 'i') parts.push(<Text key={m.index} style={styles.em}>{processInline(content)}</Text>)
      else if (tag === 'a') parts.push(<Text key={m.index} style={styles.a}>{processInline(content)}</Text>)
      else if (tag === 'small') parts.push(<Text key={m.index} style={styles.small}>{processInline(content)}</Text>)
      else if (tag === 'u') parts.push(<Text key={m.index} style={{ textDecoration: 'underline' }}>{processInline(content)}</Text>)
      else if (tag === 'del') parts.push(<Text key={m.index} style={styles.del}>{processInline(content)}</Text>)
      last = m.index + m[0].length
    }
    if (last < text.length) parts.push(unescape(text.slice(last)))
    return parts.length ? parts : text
  }
  
  return unescape(text)
}

// Get content between tags
const getTagContent = (html: string): string => {
  const openEnd = html.indexOf('>')
  const closeStart = html.lastIndexOf('</')
  if (openEnd === -1 || closeStart === -1) return html
  return html.slice(openEnd + 1, closeStart)
}

// Parse a block element
const parseBlockElement = (html: string, idx: number): React.ReactNode => {
  if (!html || !html.trim()) return null
  
  const tagMatch = html.match(/^<([a-z]+)/i)
  if (!tagMatch) return <Text key={idx} style={styles.p}>{processInline(html)}</Text>
  
  const tag = tagMatch[1].toLowerCase()
  const content = getTagContent(html)
  
  switch (tag) {
    case 'p':
      return <Text key={idx} style={styles.p}>{processInline(content)}</Text>
    case 'h1':
      return <Text key={idx} style={styles.h1}>{processInline(content)}</Text>
    case 'h2':
      return <Text key={idx} style={styles.h2}>{processInline(content)}</Text>
    case 'h3':
      return <Text key={idx} style={styles.h3}>{processInline(content)}</Text>
    case 'blockquote':
      return <Text key={idx} style={styles.blockquote}>{processInline(content)}</Text>
    case 'ul':
      return <View key={idx} style={styles.ul}>{parseList(content, 'ul', idx)}</View>
    case 'ol':
      return <View key={idx} style={styles.ol}>{parseList(content, 'ol', idx)}</View>
    case 'li':
      return <Text key={idx} style={styles.li}>{processInline(content)}</Text>
    case 'div':
      return <View key={idx} style={styles.div}>{processInline(content)}</View>
    default:
      return <Text key={idx} style={styles.p}>{processInline(content)}</Text>
  }
}

// Parse list items
const parseList = (content: string, listType: string, baseIdx: number): React.ReactNode[] => {
  const items: React.ReactNode[] = []
  const liRe = /<li[^>]*>([\s\S]*?)<\/li>/gi
  let m
  let counter = 1
  
  while ((m = liRe.exec(content)) !== null) {
    const itemContent = m[1]
    const prefix = listType === 'ol' ? `${counter}. ` : '• '
    
    // Check for p tags inside li
    const pRe = /<p[^>]*>([\s\S]*?)<\/p>/gi
    const paragraphs: string[] = []
    let pMatch
    while ((pMatch = pRe.exec(itemContent)) !== null) {
      paragraphs.push(pMatch[1])
    }
    
    if (paragraphs.length > 0) {
      const textEls = paragraphs.map((para, pIdx) => (
        <Text key={`${m!.index}-${pIdx}`} style={styles.li}>
          {pIdx === 0 ? prefix : '  '}{processInline(para)}
        </Text>
      ))
      items.push(<View key={m.index}>{textEls}</View>)
    } else {
      items.push(<Text key={m.index} style={styles.li}>{prefix}{processInline(itemContent)}</Text>)
    }
    counter++
  }
  
  // Fallback
  if (items.length === 0) {
    const lines = content.split(/\n/).filter(l => l.trim())
    lines.forEach((line, i) => {
      const prefix = listType === 'ol' ? `${i + 1}. ` : '• '
      items.push(<Text key={i} style={styles.li}>{prefix}{processInline(line)}</Text>)
    })
  }
  
  return items
}

export const RichText: React.FC<RichTextProps> = ({ html }) => {
  if (!html || typeof html !== 'string') {
    return <Text style={styles.p}>{html || ''}</Text>
  }
  
  // Check if there are HTML tags
  if (!/<[a-z][\s\S]*>/i.test(html)) {
    return <Text style={styles.p}>{unescape(html)}</Text>
  }
  
  const cleanHtml = html.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim()
  const elements: React.ReactNode[] = []
  
  // Match block elements
  const blockRe = /<(p|h1|h2|h3|ul|ol|blockquote|div|li)[^>]*>[\s\S]*?<\/\1>/gi
  let m
  let lastIdx = 0
  
  while ((m = blockRe.exec(cleanHtml)) !== null) {
    if (m.index > lastIdx) {
      const before = cleanHtml.slice(lastIdx, m.index).trim()
      if (before) {
        if (/<[a-z]/i.test(before)) {
          const p = parseBlockElement(before, lastIdx)
          if (p) elements.push(p)
        } else {
          elements.push(<Text key={lastIdx}>{unescape(before)}</Text>)
        }
      }
    }
    
    const parsed = parseBlockElement(m[0], m.index)
    if (parsed) elements.push(parsed)
    
    lastIdx = m.index + m[0].length
  }
  
  if (lastIdx < cleanHtml.length) {
    const remaining = cleanHtml.slice(lastIdx).trim()
    if (remaining) {
      if (/<[a-z]/i.test(remaining)) {
        const p = parseBlockElement(remaining, lastIdx)
        if (p) elements.push(p)
      } else {
        elements.push(<Text key={lastIdx}>{unescape(remaining)}</Text>)
      }
    }
  }
  
  if (elements.length === 0) {
    return <Text style={styles.p}>{unescape(html)}</Text>
  }
  
  return <>{elements}</>
}

export default RichText
