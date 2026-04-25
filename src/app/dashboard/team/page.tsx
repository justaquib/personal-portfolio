'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import { TeamMembersTab } from '@/components/dashboard/tabs/TeamMembersTab'
import type { TabType } from '@/types/database'

export default function TeamPage() {
  const router = useRouter()
  const { user, loading: authLoading, signOut, isAdmin, isSuperAdmin } = useAuth()

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login')
    }
  }, [user, authLoading, router])

  // Redirect if not admin
  useEffect(() => {
    if (!authLoading && user && !isAdmin && !isSuperAdmin) {
      router.push('/dashboard/payment-tracking')
    }
  }, [user, authLoading, isAdmin, isSuperAdmin, router])

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

  if (!isAdmin && !isSuperAdmin) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#f8f9fa' }}>
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Access Denied</h2>
          <p className="text-gray-600">You need admin privileges to access team management.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#f8f9fa' }}>
      <Sidebar activeTab='team' onTabChange={(tab) => {
        // Handle navigation to other main sections
        if (['send', 'contacts', 'services', 'payments', 'templates', 'history', 'earnings', 'analytics'].includes(tab)) {
          router.push('/dashboard/payment-tracking')
        } else if (tab === 'resume-builder') {
          router.push('/dashboard/tools')
        } else if (tab === 'profile') {
          router.push('/dashboard/account')
        }
      }} />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold" style={{ color: '#212529' }}>Team Management</h1>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>Manage team members and their access roles</p>
          </div>

          {/* Team Management Content */}
          <div className="bg-white rounded-lg shadow-lg">
            <TeamMembersTab />
          </div>
        </div>
      </main>
    </div>
  )
}