'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import { DashboardTopbar } from '@/components/dashboard/DashboardTopbar'
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
    window.addEventListener('sidebar-toggle', checkCollapsed)

    return () => {
      window.removeEventListener('storage', checkCollapsed)
      window.removeEventListener('sidebar-toggle', checkCollapsed)
    }
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
    <div className="dashboard-theme">
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
      <main className="dashboard-shell-main overflow-y-auto" data-sidebar-collapsed={sidebarCollapsed}>
        <DashboardTopbar />
        <div className="mx-auto max-w-7xl px-4 py-7 sm:px-6 sm:py-9 lg:px-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="dashboard-page-title">Tools</h1>
            <p className="dashboard-page-description mt-1">Productivity tools and utilities</p>
          </div>

          {/* Tools Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {/* Resume Builder Tool */}
            <div className="dashboard-panel p-5 sm:p-6">
              <div className="flex items-center mb-4">
                <div className="mr-3 flex h-10 w-10 items-center justify-center rounded-lg bg-[#edf4ec]">
                  <svg className="h-6 w-6 text-[#4b7851]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-base font-semibold text-[#334a38]">ATS Resume Builder</h3>
                  <p className="text-xs text-[#839087]">Create ATS-compatible resumes</p>
                </div>
              </div>
              <button
                onClick={() => document.getElementById('resume-builder-content')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                className="w-full rounded-xl bg-[#254f36] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#1d412c]"
              >
                Open Resume Builder
              </button>
            </div>

            {/* Placeholder for future tools */}
            <div className="rounded-2xl border border-dashed border-[#d8e1d7] bg-white p-5 sm:p-6">
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <svg className="h-6 w-6 text-[#98a49a]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                  </div>
                  <p className="text-sm text-[#89958c]">More tools coming soon</p>
                </div>
              </div>
            </div>
          </div>

          {/* Resume Builder Content */}
          <div id="resume-builder-content" className="dashboard-panel scroll-mt-6 overflow-hidden">
            <ResumeBuilderTab />
          </div>
        </div>
      </main>
    </div>
  )
}
