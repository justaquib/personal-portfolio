'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import { ResumeBuilderTab } from '@/components/dashboard/tabs/ResumeBuilderTab'
import type { TabType } from '@/types/database'

export default function ToolsPage() {
  const router = useRouter()
  const { user, loading: authLoading, signOut } = useAuth()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Listen for sidebar collapse changes
  useEffect(() => {
    const checkCollapsed = () => {
      const collapsed = localStorage.getItem('sidebar-collapsed')
      setSidebarCollapsed(collapsed ? JSON.parse(collapsed) : false)
    }

    checkCollapsed()
    window.addEventListener('storage', checkCollapsed)

    return () => window.removeEventListener('storage', checkCollapsed)
  }, [])

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login')
    }
  }, [user, authLoading, router])

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#f8f9fa' }}>
        <div className="text-center">
          <div
            className="w-12 h-12 border-4 animate-spin mx-auto mb-4"
            style={{ borderColor: '#6c757d', borderTopColor: 'transparent', borderRadius: '50%' }}
          />
          <p style={{ color: '#6c757d' }}>Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  return (
    <div className="min-h-screen" style={{ backgroundColor: '#f8f9fa' }}>
      <Sidebar activeTab='resume-builder' onTabChange={(tab) => {
        // Handle navigation to other main sections
        if (['send', 'contacts', 'services', 'payments', 'templates', 'history', 'earnings'].includes(tab)) {
          router.push('/dashboard/payment-tracking')
        } else if (tab === 'analytics') {
          router.push('/dashboard/analytics')
        } else if (tab === 'team') {
          router.push('/dashboard/team')
        } else if (tab === 'profile') {
          router.push('/dashboard/account')
        }
      }} />
      <main className={`transition-all duration-300 overflow-y-auto min-h-screen ${
        sidebarCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold" style={{ color: '#212529' }}>Tools</h1>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>Productivity tools and utilities</p>
          </div>

          {/* Tools Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {/* Resume Builder Tool */}
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mr-3">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">ATS Resume Builder</h3>
                  <p className="text-sm text-gray-500">Create ATS-compatible resumes</p>
                </div>
              </div>
              <button
                onClick={() => {/* Could add modal or direct render */}}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Open Resume Builder
              </button>
            </div>

            {/* Placeholder for future tools */}
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 border-dashed">
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-500">More tools coming soon</p>
                </div>
              </div>
            </div>
          </div>

          {/* Resume Builder Content */}
          <div className="bg-white rounded-lg shadow-lg">
            <ResumeBuilderTab />
          </div>
        </div>
      </main>
    </div>
  )
}