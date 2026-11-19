'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { Sidebar } from '@/components/dashboard/Sidebar'
import { DashboardTopbar } from '@/components/dashboard/DashboardTopbar'
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

  const { title, description } = tabTitles[activeTab]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'send':
        return <SendTab key="send" userId={user.id} />
      case 'contacts':
        return <ContactsTab key="contacts" userId={user.id} />
      case 'services':
        return <ServicesTab key="services" userId={user.id} />
      case 'payments':
        return <PaymentsTab key="payments" userId={user.id} />
      case 'templates':
        return <TemplatesTab key="templates" userId={user.id} />
      case 'history':
        return <HistoryTab key="history" />
      case 'earnings':
        return <EarningsTab key="earnings" userId={user.id} />
      default:
        return <SendTab key="send" userId={user.id} />
    }
  }

  return (
      <div className="dashboard-theme">
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
      <main className="dashboard-shell-main overflow-y-auto" data-sidebar-collapsed={sidebarCollapsed}>
        <DashboardTopbar />
        <div className="mx-auto max-w-7xl px-4 py-7 sm:px-6 sm:py-9 lg:px-8">
          {/* Page Header */}
          <div className="mb-8">
            <h1 className="dashboard-page-title">Payment tracking</h1>
            <p className="dashboard-page-description mt-1">Manage payments, contacts, and track earnings</p>
          </div>

          {/* Sub-tabs for Payment Tracking */}
          <div className="mb-6">
            <div className="flex snap-x gap-1 overflow-x-auto rounded-2xl border border-[#e2e8e0] bg-white p-1.5 shadow-[0_8px_24px_-22px_rgba(34,64,42,.4)]">
              {(Object.keys(tabTitles) as PaymentTabType[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`shrink-0 rounded-xl px-3.5 py-2 text-xs font-medium transition-colors sm:text-sm ${
                    activeTab === tab
                      ? 'bg-[#edf4ec] text-[#315b3b]'
                      : 'text-[#78867b] hover:bg-[#f7f9f6] hover:text-[#3f5d46]'
                  }`}
                >
                  {tabTitles[tab].title}
                </button>
              ))}
            </div>
          </div>

          {/* Current sub-tab title and description */}
          <div className="mb-5 mt-6 flex items-end justify-between gap-3">
            <div><h2 className="text-lg font-semibold tracking-tight text-[#334a38]">{title}</h2><p className="mt-1 text-xs text-[#87938a]">{description}</p></div>
            <span className="hidden rounded-full border border-[#e1e8df] bg-white px-3 py-1.5 text-[10px] font-medium text-[#829087] sm:inline-flex">Workspace</span>
          </div>

          {/* Tab Content */}
          {renderTabContent()}
        </div>
      </main>
    </div>
  )
}
