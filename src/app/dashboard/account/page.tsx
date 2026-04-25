'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import ProfileTab from '@/components/dashboard/tabs/ProfileTab'
import type { TabType } from '@/types/database'

export default function AccountPage() {
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
    <div className="min-h-screen" style={{ backgroundColor: '#f8f9fa' }}>
      <Sidebar activeTab='profile' onTabChange={(tab) => {
        // Handle navigation to other main sections
        if (['send', 'contacts', 'services', 'payments', 'templates', 'history', 'earnings'].includes(tab)) {
          router.push('/dashboard/payment-tracking')
        } else if (tab === 'analytics') {
          router.push('/dashboard/analytics')
        } else if (tab === 'resume-builder') {
          router.push('/dashboard/tools')
        } else if (tab === 'team') {
          router.push('/dashboard/team')
        }
      }} />
      <main className={`transition-all duration-300 overflow-y-auto min-h-screen ${
        sidebarCollapsed ? 'ml-16' : 'ml-64'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold" style={{ color: '#212529' }}>Account</h1>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>Manage your account settings and preferences</p>
          </div>

          {/* Account/Profile Content */}
          <div className="max-w-4xl">
            <ProfileTab user={user} />
          </div>
        </div>
      </main>
    </div>
  )
}