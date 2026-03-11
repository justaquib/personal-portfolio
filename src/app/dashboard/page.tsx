'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import { SendTab } from '@/components/dashboard/tabs/SendTab'
import { ContactsTab } from '@/components/dashboard/tabs/ContactsTab'
import { ServicesTab } from '@/components/dashboard/tabs/ServicesTab'
import { PaymentsTab } from '@/components/dashboard/tabs/PaymentsTab'
import { TemplatesTab } from '@/components/dashboard/tabs/TemplatesTab'
import { HistoryTab } from '@/components/dashboard/tabs/HistoryTab'
import { EarningsTab } from '@/components/dashboard/tabs/EarningsTab'
import { ResumeBuilderTab } from '@/components/dashboard/tabs/ResumeBuilderTab'
import type { TabType } from '@/types/database'

const tabTitles: Record<TabType, { title: string; description: string }> = {
  send: { 
    title: 'Send Notification', 
    description: 'Send payment notifications to your contacts via WhatsApp' 
  },
  contacts: { 
    title: 'Contacts', 
    description: 'Manage your contacts and their information' 
  },
  services: { 
    title: 'Services', 
    description: 'Manage your services and subscriptions' 
  },
  payments: { 
    title: 'Payments', 
    description: 'Track and manage payment records' 
  },
  templates: { 
    title: 'Templates', 
    description: 'Create and manage message templates' 
  },
  history: { 
    title: 'History', 
    description: 'View your notification history' 
  },
  earnings: { 
    title: 'Earnings', 
    description: 'Track your earnings and revenue' 
  },
  'resume-builder': { 
    title: 'ATS Resume Builder', 
    description: 'Build ATS-compatible resumes' 
  },
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, loading: authLoading, signOut } = useAuth()
  const [activeTab, setActiveTab] = useState<TabType>('send')

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

  const { title, description } = tabTitles[activeTab]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'send':
        return <SendTab userId={user.id} />
      case 'contacts':
        return <ContactsTab userId={user.id} />
      case 'services':
        return <ServicesTab userId={user.id} />
      case 'payments':
        return <PaymentsTab userId={user.id} />
      case 'templates':
        return <TemplatesTab userId={user.id} />
      case 'history':
        return <HistoryTab />
      case 'earnings':
        return <EarningsTab userId={user.id} />
      case 'resume-builder':
        return <ResumeBuilderTab />
      default:
        return <SendTab userId={user.id} />
    }
  }

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#f8f9fa' }}>
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold" style={{ color: '#212529' }}>{title}</h1>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>{description}</p>
          </div>

          {/* Tab Content */}
          {renderTabContent()}
        </div>
      </main>
    </div>
  )
}
