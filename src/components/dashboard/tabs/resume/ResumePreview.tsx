'use client'

import React from 'react'
import { ResumeData } from './types'

interface ResumePreviewProps {
  resumeData: ResumeData
}

export function ResumePreview({ resumeData }: ResumePreviewProps) {
  const selectedTemplate = {
    id: resumeData.template,
    name: resumeData.template.charAt(0).toUpperCase() + resumeData.template.slice(1)
  }

  return (
    <div className="bg-white border rounded-xl p-8 max-w-2xl mx-auto shadow-lg">
      {/* Template-specific header styling */}
      <div className={`text-center border-b pb-6 mb-6 ${
        resumeData.template === 'modern' ? 'bg-gradient-to-r from-gray-600 to-gray-800 -mx-8 -mt-8 p-8 rounded-t-xl' : ''
      }`}>
        <h2 className={`text-2xl font-bold ${
          resumeData.template === 'modern' ? 'text-white' : 
          resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
          resumeData.template === 'minimal' ? 'text-gray-700 font-light' :
          'text-orange-600'
        }`}>
          {resumeData.personalInfo.name || 'Your Name'}
        </h2>
        <div className={`text-sm mt-2 space-y-1 ${
          resumeData.template === 'modern' ? 'text-white/80' : 'text-gray-600'
        }`}>
          <div className='flex flex-row items-center justify-center gap-2'>
            {resumeData.personalInfo.email && <div>{resumeData.personalInfo.email}</div>}
            <span className="text-gray-400">|</span>
            {resumeData.personalInfo.phone && <div>{resumeData.personalInfo.phone}</div>}
          </div>
          {resumeData.personalInfo.location && <div>{resumeData.personalInfo.location}</div>}
          <div className='flex flex-row items-center justify-center gap-2'>
            {resumeData.personalInfo.linkedin && <div>LinkedIn: {resumeData.personalInfo.linkedin}</div>}
            <span className="text-gray-400">|</span>
            {resumeData.personalInfo.portfolio && <div>Portfolio: {resumeData.personalInfo.portfolio}</div>}
          </div>
        </div>
      </div>

      {resumeData.summary && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Professional Summary</h3>
          <div className="text-gray-700" dangerouslySetInnerHTML={{ __html: resumeData.summary }} />
        </div>
      )}

      {resumeData.experience.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Work Experience</h3>
          {resumeData.experience.map(exp => (
            <div key={exp.id} className="mb-4">
              <div className="flex justify-between items-baseline">
                <span className="font-semibold text-gray-900">{exp.role}</span>
                <span className="text-sm text-gray-600">
                  {exp.current ? `${exp.startDate} - Present` : `${exp.startDate} - ${exp.endDate}`}
                </span>
              </div>
              <div className="text-gray-600 italic">{exp.company}</div>
              {exp.location && <div className="text-gray-600">{exp.location}</div>}
              <div className="text-gray-700 mt-1" dangerouslySetInnerHTML={{ __html: exp.description }} />
            </div>
          ))}
        </div>
      )}

      {resumeData.education.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Education</h3>
          {resumeData.education.map(edu => (
            <div key={edu.id} className="mb-3">
              <div className="flex justify-between items-baseline">
                <span className="font-semibold text-gray-900">{edu.degree} in {edu.field}</span>
                <span className="text-sm text-gray-600">{edu.graduationDate}</span>
              </div>
              <div className="text-gray-600 italic">{edu.institution}</div>
            </div>
          ))}
        </div>
      )}

      {resumeData.skills && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Skills</h3>
          <p className="text-gray-700">{resumeData.skills}</p>
        </div>
      )}

      {resumeData.projects.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Projects</h3>
          {resumeData.projects.map(proj => (
            <div key={proj.id} className="mb-3">
              <div className="font-semibold text-gray-900">{proj.name}</div>
              {proj.technologies && <div className="text-sm text-gray-600">Technologies: {proj.technologies}</div>}
              <div className="text-gray-700 mt-1" dangerouslySetInnerHTML={{ __html: proj.description }} />
            </div>
          ))}
        </div>
      )}

      {/* Certifications - Display as badges/pills */}
      {resumeData.certifications.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Certifications</h3>
          <div className="flex flex-wrap gap-2">
            {resumeData.certifications.map(cert => (
              <span
                key={cert.id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: resumeData.template === 'modern' ? '#9333ea' :
                    resumeData.template === 'classic' ? '#333' :
                    resumeData.template === 'minimal' ? '#3b82f6' : '#ea580c',
                  color: 'white'
                }}
              >
                {cert.issuer ? `${cert.name} (${cert.issuer})` : cert.name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Languages - Display as badges/pills */}
      {resumeData.languages.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Languages</h3>
          <div className="flex flex-wrap gap-2">
            {resumeData.languages.map(lang => (
              <span
                key={lang.id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: resumeData.template === 'modern' ? '#4f46e5' :
                    resumeData.template === 'classic' ? '#555' :
                    resumeData.template === 'minimal' ? '#6366f1' : '#f97316',
                  color: 'white'
                }}
              >
                {lang.name} - {lang.proficiency}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Websites - Display as badges/pills */}
      {resumeData.websites.length > 0 && (
        <div className="mb-6">
          <h3 className={`text-lg font-bold border-b pb-2 mb-3 ${
            resumeData.template === 'modern' ? 'text-gray-600' :
            resumeData.template === 'classic' ? 'text-gray-900 font-serif' :
            resumeData.template === 'minimal' ? 'text-gray-700' :
            'text-orange-600'
          }`}>Websites & Portfolio</h3>
          <div className="flex flex-wrap gap-2">
            {resumeData.websites.map(ws => (
              <span
                key={ws.id}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: resumeData.template === 'modern' ? '#0891b2' :
                    resumeData.template === 'classic' ? '#666' :
                    resumeData.template === 'minimal' ? '#14b8a6' : '#e11d48',
                  color: 'white'
                }}
              >
                {ws.name}: {ws.url}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
