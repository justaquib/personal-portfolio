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
import { AnalyticsTab } from '@/components/dashboard/tabs/AnalyticsTab'
import type { TabType } from '@/types/database'

type PaymentTabType = 'send' | 'contacts' | 'services' | 'payments' | 'templates' | 'history' | 'earnings'

const tabTitles: Record<PaymentTabType, { title: string; description: string }> = {
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
}

export default function PaymentTrackingPage() {
  const router = useRouter()
  const { user, loading: authLoading, signOut } = useAuth()
  const [activeTab, setActiveTab] = useState<PaymentTabType>('send')

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
      default:
        return <SendTab userId={user.id} />
    }
  }

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#f8f9fa' }}>
      <Sidebar activeTab={activeTab as TabType} onTabChange={(tab) => {
        // Handle navigation to other main sections
        if (tab === 'analytics') {
          router.push('/dashboard/analytics')
        } else if (tab === 'resume-builder') {
          router.push('/dashboard/tools')
        } else if (tab === 'team') {
          router.push('/dashboard/team')
        } else if (tab === 'profile') {
          router.push('/dashboard/account')
        } else {
          setActiveTab(tab as PaymentTabType)
        }
      }} />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold" style={{ color: '#212529' }}>Payment Tracking</h1>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>Manage payments, contacts, and track earnings</p>
          </div>

          {/* Sub-tabs for Payment Tracking */}
          <div className="mb-6">
            <div className="flex gap-2 border-b border-gray-200">
              {(Object.keys(tabTitles) as PaymentTabType[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tabTitles[tab].title}
                </button>
              ))}
            </div>
          </div>

          {/* Current sub-tab title and description */}
          <div className="mb-6">
            <h2 className="text-xl font-semibold" style={{ color: '#212529' }}>{title}</h2>
            <p className="text-sm mt-1" style={{ color: '#6c757d' }}>{description}</p>
          </div>

          {/* Tab Content */}
          {renderTabContent()}
        </div>
      </main>
    </div>
  )
}