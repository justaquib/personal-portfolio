'use client'

import { ReactNode } from 'react'

interface BadgeProps {
  children: ReactNode
  variant?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'danger' | 'info'
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function Badge({ 
  children, 
  variant = 'default',
  size = 'md',
  className = '' 
}: BadgeProps) {
  const variantClasses = {
    default: 'bg-gray-900 text-white',
    primary: 'bg-blue-600 text-white',
    secondary: 'bg-gray-500 text-white',
    success: 'bg-green-600 text-white',
    warning: 'bg-yellow-500 text-white',
    danger: 'bg-red-600 text-white',
    info: 'bg-cyan-600 text-white',
  }

  const sizeClasses = {
    sm: 'text-xs px-1.5 py-0.5 min-w-[18px]',
    md: 'text-xs px-2 py-0.5 min-w-[20px]',
    lg: 'text-sm px-2.5 py-1 min-w-[24px]',
  }

  return (
    <span 
      className={`
        inline-flex items-center justify-center font-medium rounded-full
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${className}
      `}
    >
      {children}
    </span>
  )
}
