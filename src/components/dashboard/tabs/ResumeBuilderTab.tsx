'use client'

import { Card, EmptyState, Button } from '../ui'

export function ResumeBuilderTab() {
  const handleBuildResume = () => {
    // TODO: Implement resume builder functionality
    alert('Resume Builder is coming soon!')
  }

  return (
    <Card title="ATS Compatible Resume Builder">
      <div className="text-center py-12">
        <div className="w-20 h-20 mx-auto mb-6 bg-purple-100 rounded-full flex items-center justify-center">
          <svg className="w-10 h-10 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Build Your Professional Resume
        </h3>
        <p className="text-gray-600 max-w-md mx-auto mb-8">
          Create ATS-optimized resumes that pass through applicant tracking systems. 
          Stand out to recruiters with professionally formatted resumes.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-2xl mx-auto mb-8">
          <div className="p-4 bg-gray-50 rounded-xl">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h4 className="font-medium text-gray-900 mb-1">ATS Compatible</h4>
            <p className="text-sm text-gray-500">Optimized format that passes through all major ATS systems</p>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-xl">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
              </svg>
            </div>
            <h4 className="font-medium text-gray-900 mb-1">Multiple Templates</h4>
            <p className="text-sm text-gray-500">Choose from professional templates designed by experts</p>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-xl">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </div>
            <h4 className="font-medium text-gray-900 mb-1">Easy Export</h4>
            <p className="text-sm text-gray-500">Download as PDF or Word document instantly</p>
          </div>
        </div>

        <Button size="lg" onClick={handleBuildResume}>
          Start Building Resume
        </Button>
      </div>
    </Card>
  )
}
