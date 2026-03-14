'use client'

import { useState, ReactNode } from 'react'

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right'

interface TooltipProps {
  content: string
  children: ReactNode
  position?: TooltipPosition
  animation?: boolean
}

export function Tooltip({ 
  content, 
  children, 
  position = 'top',
  animation = true 
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false)

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  }

  const arrowClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-1 border-b-gray-900 border-t-0 border-l-0 border-r-0',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-1 border-t-gray-900 border-b-0 border-l-0 border-r-0',
    left: 'right-full top-1/2 -translate-y-1/2 mr-1 border-r-gray-900 border-l-0 border-t-0 border-b-0',
    right: 'left-full top-1/2 -translate-y-1/2 ml-1 border-l-gray-900 border-r-0 border-t-0 border-b-0',
  }

  const animationClasses = animation ? {
    top: 'animate-tooltip-fade-in-up',
    bottom: 'animate-tooltip-fade-in-down',
    left: 'animate-tooltip-fade-in-left',
    right: 'animate-tooltip-fade-in-right',
  } : {
    top: '',
    bottom: '',
    left: '',
    right: '',
  }

  return (
    <div 
      className="relative inline-flex overflow-visible"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
      onFocus={() => setIsVisible(true)}
      onBlur={() => setIsVisible(false)}
    >
      {children}
      
      {isVisible && content && (
        <>
          {/* Tooltip arrow */}
          <div 
            className={`absolute border-[6px] border-transparent ${arrowClasses[position]}`}
          />
          
          {/* Tooltip content */}
          <div 
            className={`
              absolute z-[9999] px-3 py-2 text-sm font-medium text-white 
              bg-gray-900 rounded-lg shadow-lg whitespace-nowrap
              pointer-events-none
              ${positionClasses[position]}
              ${animationClasses[position]}
            `}
            role="tooltip"
          >
            {content}
          </div>
        </>
      )}
    </div>
  )
}

// Keyframe animations are added to globals.css
// For usage, import and use like:
// <Tooltip content="Your tooltip text" position="top">
//   <button>Hover me</button>
// </Tooltip>
