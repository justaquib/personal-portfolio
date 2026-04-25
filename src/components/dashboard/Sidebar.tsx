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
  User,
  BarChart3
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

  const mainSections = [
    {
      id: 'payment-tracking',
      label: 'Payment Tracking',
      icon: <DollarSign className="w-5 h-5" />,
      description: 'Manage payments, contacts, and track earnings',
      category: 'main'
    },
    {
      id: 'tools',
      label: 'Tools',
      icon: <FileUp className="w-5 h-5" />,
      description: 'Productivity tools and utilities',
      category: 'main'
    },
    {
      id: 'team',
      label: 'Team Management',
      icon: <UserPlus className="w-5 h-5" />,
      description: 'Manage team members and roles',
      category: 'admin'
    },
    {
      id: 'account',
      label: 'Account',
      icon: <User className="w-5 h-5" />,
      description: 'Account settings and preferences',
      category: 'main'
    }
  ]

  // Legacy tabs for backward compatibility (not used in new structure)
  const tabs: { id: TabType; label: string; icon: React.ReactNode; category: 'payment' | 'resume' | 'team' | 'account' }[] = [
    { id: 'send', label: 'Send', icon: <Send className="w-5 h-5" />, category: 'payment' },
    { id: 'contacts', label: 'Contacts', icon: <Users className="w-5 h-5" />, category: 'payment' },
    { id: 'services', label: 'Services', icon: <Package className="w-5 h-5" />, category: 'payment' },
    { id: 'payments', label: 'Payments', icon: <CreditCard className="w-5 h-5" />, category: 'payment' },
    { id: 'templates', label: 'Templates', icon: <FileText className="w-5 h-5" />, category: 'payment' },
    { id: 'history', label: 'History', icon: <History className="w-5 h-5" />, category: 'payment' },
    { id: 'earnings', label: 'Earnings', icon: <DollarSign className="w-5 h-5" />, category: 'payment' },
    { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-5 h-5" />, category: 'payment' },
    { id: 'resume-builder', label: 'Resume Builder', icon: <FileUp className="w-5 h-5" />, category: 'resume' },
    { id: 'team', label: 'Team', icon: <UserPlus className="w-5 h-5" />, category: 'team' },
    { id: 'profile', label: 'Profile', icon: <User className="w-5 h-5" />, category: 'account' },
  ]

  const mainSectionsFiltered = mainSections.filter(section => {
    if (section.category === 'admin' && !isAdmin && !isSuperAdmin) {
      return false
    }
    return true
  })

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
        {/* Main Sections */}
        <div className="px-4">
          <ul className="space-y-2">
            {mainSectionsFiltered.map((section) => {
              const isActive = (() => {
                if (section.id === 'payment-tracking') {
                  return ['send', 'contacts', 'services', 'payments', 'templates', 'history', 'earnings', 'analytics'].includes(activeTab)
                }
                if (section.id === 'tools') {
                  return activeTab === 'resume-builder'
                }
                if (section.id === 'team') {
                  return activeTab === 'team'
                }
                if (section.id === 'account') {
                  return activeTab === 'profile'
                }
                return false
              })()

              return (
                <li key={section.id}>
                  <button
                    onClick={() => {
                      // Navigate to the appropriate page
                      if (section.id === 'payment-tracking') {
                        window.location.href = '/dashboard/payment-tracking'
                      } else if (section.id === 'tools') {
                        window.location.href = '/dashboard/tools'
                      } else if (section.id === 'team') {
                        window.location.href = '/dashboard/team'
                      } else if (section.id === 'account') {
                        window.location.href = '/dashboard/account'
                      }
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-colors ${
                      isActive ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50'
                    }`}
                    style={{
                      color: isActive ? '#1e40af' : '#6c757d'
                    }}
                  >
                    <div className={`p-1 rounded ${isActive ? 'bg-blue-100' : 'bg-gray-100'}`}>
                      {section.icon}
                    </div>
                    <div className="flex-1 text-left">
                      <div className="font-medium">{section.label}</div>
                      <div className="text-xs opacity-75">{section.description}</div>
                    </div>
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      </nav>

      {/* User Section */}
      <div className="p-4" style={{ borderTop: '1px solid #dee2e6' }}>
        <button
          onClick={() => window.location.href = '/dashboard/account'}
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
