'use client'

import { useAuth } from '@/context/AuthContext'
import { useRouter } from 'next/navigation'
import { TabType } from '@/types/database'
import { 
  Send, 
  Users, 
  Package, 
  CreditCard, 
  FileText, 
  History, 
  DollarSign,
  FileUp,
  LogOut,
  LayoutDashboard
} from 'lucide-react'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const { user, signOut } = useAuth()
  const router = useRouter()

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  const tabs: { id: TabType; label: string; icon: React.ReactNode; category: 'payment' | 'resume' }[] = [
    { id: 'send', label: 'Send', icon: <Send className="w-5 h-5" />, category: 'payment' },
    { id: 'contacts', label: 'Contacts', icon: <Users className="w-5 h-5" />, category: 'payment' },
    { id: 'services', label: 'Services', icon: <Package className="w-5 h-5" />, category: 'payment' },
    { id: 'payments', label: 'Payments', icon: <CreditCard className="w-5 h-5" />, category: 'payment' },
    { id: 'templates', label: 'Templates', icon: <FileText className="w-5 h-5" />, category: 'payment' },
    { id: 'history', label: 'History', icon: <History className="w-5 h-5" />, category: 'payment' },
    { id: 'earnings', label: 'Earnings', icon: <DollarSign className="w-5 h-5" />, category: 'payment' },
    { id: 'resume-builder', label: 'Resume Builder', icon: <FileUp className="w-5 h-5" />, category: 'resume' },
  ]

  const paymentTabs = tabs.filter(t => t.category === 'payment')
  const resumeTabs = tabs.filter(t => t.category === 'resume')

  return (
    <aside className="w-64 bg-white border-r border-gray-200 min-h-screen flex flex-col">
      {/* Logo/Brand */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-600 rounded-xl flex items-center justify-center">
            <LayoutDashboard className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">Dashboard</h1>
            <p className="text-xs text-gray-500">Payment Tracker</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        {/* Payment Tracking Section */}
        <div className="px-4 mb-2">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
            Payment Tracking
          </p>
          <ul className="space-y-1">
            {paymentTabs.map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => onTabChange(tab.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-purple-50 text-purple-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Resume Builder Section */}
        <div className="px-4 mt-6">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 px-2">
            Tools
          </p>
          <ul className="space-y-1">
            {resumeTabs.map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => onTabChange(tab.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'bg-purple-50 text-purple-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                  {tab.id === 'resume-builder' && (
                    <span className="ml-auto text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full">
                      New
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      {/* User Section */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-center gap-3 mb-3 px-2">
          <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
            <span className="text-purple-700 font-medium text-sm">
              {user?.user_metadata?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
            </span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">
              {user?.user_metadata?.full_name || 'User'}
            </p>
            <p className="text-xs text-gray-500 truncate">
              {user?.email}
            </p>
          </div>
        </div>
        <button
          onClick={handleSignOut}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-red-600 hover:bg-red-50 transition-colors"
        >
          <LogOut className="w-5 h-5" />
          Sign Out
        </button>
      </div>
    </aside>
  )
}
