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
  LayoutDashboard,
  UserPlus,
  User
} from 'lucide-react'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const { user, signOut, isAdmin, isSuperAdmin, role } = useAuth()
  const router = useRouter()

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  const roleLabels: Record<string, string> = {
    super_admin: 'Super Admin',
    admin: 'Admin',
    editor: 'Editor',
    viewer: 'Viewer'
  }

  const roleColors: Record<string, string> = {
    super_admin: '#6f42c1',
    admin: '#dc3545',
    editor: '#28a745',
    viewer: '#6c757d'
  }

  const tabs: { id: TabType; label: string; icon: React.ReactNode; category: 'payment' | 'resume' | 'team' | 'account' }[] = [
    { id: 'send', label: 'Send', icon: <Send className="w-5 h-5" />, category: 'payment' },
    { id: 'contacts', label: 'Contacts', icon: <Users className="w-5 h-5" />, category: 'payment' },
    { id: 'services', label: 'Services', icon: <Package className="w-5 h-5" />, category: 'payment' },
    { id: 'payments', label: 'Payments', icon: <CreditCard className="w-5 h-5" />, category: 'payment' },
    { id: 'templates', label: 'Templates', icon: <FileText className="w-5 h-5" />, category: 'payment' },
    { id: 'history', label: 'History', icon: <History className="w-5 h-5" />, category: 'payment' },
    { id: 'earnings', label: 'Earnings', icon: <DollarSign className="w-5 h-5" />, category: 'payment' },
    { id: 'resume-builder', label: 'Resume Builder', icon: <FileUp className="w-5 h-5" />, category: 'resume' },
    { id: 'team', label: 'Team', icon: <UserPlus className="w-5 h-5" />, category: 'team' },
    { id: 'profile', label: 'Profile', icon: <User className="w-5 h-5" />, category: 'account' },
  ]

  const paymentTabs = tabs.filter(t => t.category === 'payment')
  const resumeTabs = tabs.filter(t => t.category === 'resume')
  const teamTabs = tabs.filter(t => t.category === 'team' && (isAdmin || isSuperAdmin))
  const accountTabs = tabs.filter(t => t.category === 'account')

  return (
    <aside 
      className="w-64 min-h-screen flex flex-col"
      style={{ backgroundColor: '#f8f9fa', borderRight: '1px solid #dee2e6' }}
    >
      {/* Logo/Brand */}
      <div className="p-6" style={{ borderBottom: '1px solid #dee2e6' }}>
        <div className="flex items-center gap-3">
          <div 
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: '#212529' }}
          >
            <LayoutDashboard className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold" style={{ color: '#212529' }}>Dashboard</h1>
            <p className="text-xs" style={{ color: '#6c757d' }}>Just Aquib</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4">
        {/* Payment Tracking Section */}
        <div className="px-4 mb-2">
          <p 
            className="text-xs font-semibold uppercase tracking-wider mb-2 px-2"
            style={{ color: '#adb5bd' }}
          >
            Payment Tracking
          </p>
          <ul className="space-y-1">
            {paymentTabs.map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => onTabChange(tab.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? ''
                      : ''
                  }`}
                  style={{
                    backgroundColor: activeTab === tab.id ? '#dee2e6' : 'transparent',
                    color: activeTab === tab.id ? '#212529' : '#6c757d'
                  }}
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
          <p 
            className="text-xs font-semibold uppercase tracking-wider mb-2 px-2"
            style={{ color: '#adb5bd' }}
          >
            Tools
          </p>
          <ul className="space-y-1">
            {resumeTabs.map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => onTabChange(tab.id)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
                  style={{
                    backgroundColor: activeTab === tab.id ? '#dee2e6' : 'transparent',
                    color: activeTab === tab.id ? '#212529' : '#6c757d'
                  }}
                >
                  {tab.icon}
                  {tab.label}
                  {tab.id === 'resume-builder' && (
                    <span 
                      className="ml-auto text-xs px-2 py-0.5 rounded-full"
                      style={{ backgroundColor: '#e9ecef', color: '#495057' }}
                    >
                      New
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Team Management Section - Only for Super Admins */}
        {(isAdmin || isSuperAdmin) && (
          <div className="px-4 mt-6">
            <p 
              className="text-xs font-semibold uppercase tracking-wider mb-2 px-2"
              style={{ color: '#adb5bd' }}
            >
              Team Management
            </p>
            <ul className="space-y-1">
              {teamTabs.map((tab) => (
                <li key={tab.id}>
                  <button
                    onClick={() => onTabChange(tab.id)}
                    className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                      backgroundColor: activeTab === tab.id ? '#dee2e6' : 'transparent',
                      color: activeTab === tab.id ? '#212529' : '#6c757d'
                    }}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Account Section - Visible to all users */}
        <div className="px-4 mt-6">
          <p 
            className="text-xs font-semibold uppercase tracking-wider mb-2 px-2"
            style={{ color: '#adb5bd' }}
          >
            Account
          </p>
          <ul className="space-y-1">
            {accountTabs.map((tab) => (
              <li key={tab.id}>
                <button
                  onClick={() => onTabChange(tab.id)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
                  style={{
                    backgroundColor: activeTab === tab.id ? '#dee2e6' : 'transparent',
                    color: activeTab === tab.id ? '#212529' : '#6c757d'
                  }}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      {/* User Section */}
      <div className="p-4" style={{ borderTop: '1px solid #dee2e6' }}>
        <button
          onClick={() => onTabChange('profile')}
          className="w-full flex items-center gap-3 mb-3 px-2 py-2 rounded-lg hover:bg-gray-200 transition-colors"
        >
          <div 
            className="w-8 h-8 rounded-full flex items-center justify-center overflow-hidden"
            style={{ backgroundColor: '#e9ecef' }}
          >
            {user?.user_metadata?.avatar_url ? (
              <img 
                src={user.user_metadata.avatar_url} 
                alt="Avatar" 
                className="w-full h-full object-cover"
              />
            ) : (
              <span 
                className="font-medium text-sm capitalize"
                style={{ color: '#495057' }}
              >
                {user?.user_metadata?.full_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
              </span>
            )}
          </div>
          <div className="flex-1 min-w-0 text-left">
            <p className="text-sm font-medium truncate" style={{ color: '#212529' }}>
              {user?.user_metadata?.full_name || 'User'}
            </p>
            <p className="text-xs truncate" style={{ color: '#6c757d' }}>
              {user?.email}
            </p>
            {role && (
              <span
                className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium mt-1"
                style={{ 
                  backgroundColor: `${roleColors[role]}20`,
                  color: roleColors[role]
                }}
              >
                {roleLabels[role]}
              </span>
            )}
          </div>
        </button>
        <button
          onClick={handleSignOut}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
          style={{ color: '#495057' }}
        >
          <LogOut className="w-5 h-5" />
          Sign Out
        </button>
      </div>
    </aside>
  )
}
