import React from 'react'
import { View, Text } from '@react-pdf/renderer'

interface FooterProps {
  pageNumber: number
  totalPages: number
}

export const Footer: React.FC<FooterProps> = ({ pageNumber, totalPages }) => {
  return (
    <View
      style={{
        position: 'absolute',
        bottom: 20,
        left: 30,
        right: 30,
        flexDirection: 'row',
        justifyContent: 'space-between',
        borderTopWidth: 1,
        borderTopColor: '#e5e5e5',
        paddingTop: 8
      }}
    >
      <Text
        style={{
          fontSize: 8,
          color: '#999999'
        }}
        render={({ pageNumber, totalPages }) => `Resume - ${pageNumber} / ${totalPages}`}
        fixed
      />
      <Text
        style={{
          fontSize: 8,
          color: '#999999'
        }}
        render={({ pageNumber }) => `Page ${pageNumber}`}
        fixed
      />
    </View>
  )
}

export default Footer
