'use client'

import { useAuth } from '@/context/AuthContext'
import { useRouter } from 'next/navigation'
import { TabType } from '@/types/database'
import { useState, useEffect } from 'react'
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  CreditCard,
  FileUp,
  LayoutDashboard,
  LogOut,
  User,
  UserPlus,
} from 'lucide-react'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const { user, signOut, isAdmin, isSuperAdmin, role } = useAuth()
  const router = useRouter()
  const [isCollapsed, setIsCollapsed] = useState(false)

  useEffect(() => {
    const saved = localStorage.getItem('sidebar-collapsed')
    if (saved) setIsCollapsed(JSON.parse(saved))
  }, [])

  const toggleCollapse = () => {
    const nextState = !isCollapsed
    setIsCollapsed(nextState)
    localStorage.setItem('sidebar-collapsed', JSON.stringify(nextState))
    window.dispatchEvent(new CustomEvent('sidebar-toggle', { detail: { collapsed: nextState } }))
  }

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  const sections: { id: string; label: string; description: string; icon: React.ReactNode; active: boolean; adminOnly?: boolean; tab: TabType; route: string }[] = [
    { id: 'payment-tracking', label: 'Payment tracking', description: 'Payments, contacts, earnings', icon: <CreditCard className="h-[18px] w-[18px]" />, active: ['send', 'contacts', 'services', 'payments', 'templates', 'history', 'earnings'].includes(activeTab), tab: 'send', route: '/dashboard/payment-tracking' },
    { id: 'analytics', label: 'Analytics', description: 'Traffic and engagement', icon: <BarChart3 className="h-[18px] w-[18px]" />, active: activeTab === 'analytics', tab: 'analytics', route: '/dashboard/analytics' },
    { id: 'tools', label: 'Tools', description: 'Resume builder and utilities', icon: <FileUp className="h-[18px] w-[18px]" />, active: activeTab === 'resume-builder', tab: 'resume-builder', route: '/dashboard/tools' },
    { id: 'team', label: 'Team management', description: 'Members and access', icon: <UserPlus className="h-[18px] w-[18px]" />, active: activeTab === 'team', adminOnly: true, tab: 'team', route: '/dashboard/team' },
    { id: 'account', label: 'Account', description: 'Profile and preferences', icon: <User className="h-[18px] w-[18px]" />, active: activeTab === 'profile', tab: 'profile', route: '/dashboard/account' },
  ]
  const visibleSections = sections.filter((section) => !section.adminOnly || isAdmin || isSuperAdmin)

  const navigate = (section: (typeof sections)[number]) => {
    if (section.id === 'payment-tracking') onTabChange(section.tab)
    router.push(section.route)
  }

  const initials = (user?.user_metadata?.full_name || user?.email || 'U').charAt(0).toUpperCase()
  const displayName = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Account'

  return (
    <>
      <aside className={`fixed inset-y-0 left-0 z-40 hidden flex-col border-r border-[#e4e9e2] bg-white transition-[width] duration-200 md:flex ${isCollapsed ? 'w-[76px]' : 'w-[252px]'}`}>
        <div className={`flex h-[76px] shrink-0 items-center border-b border-[#edf0eb] ${isCollapsed ? 'justify-center px-3' : 'justify-between px-5'}`}>
          <button onClick={() => router.push('/dashboard/payment-tracking')} className="flex min-w-0 items-center gap-3 text-left" title="Open dashboard">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-[#254f36] text-white"><LayoutDashboard className="h-[17px] w-[17px]" /></span>
            {!isCollapsed && <span className="min-w-0"><span className="block truncate text-sm font-semibold tracking-tight text-[#2d4433]">Just Aquib</span><span className="mt-0.5 block text-[10px] text-[#8c988e]">Workspace</span></span>}
          </button>
          {!isCollapsed && <button onClick={toggleCollapse} aria-label="Collapse sidebar" title="Collapse sidebar" className="rounded-lg p-1.5 text-[#8a978d] transition hover:bg-[#f1f5ef] hover:text-[#426149]"><ChevronLeft className="h-4 w-4" /></button>}
          {isCollapsed && <button onClick={toggleCollapse} aria-label="Expand sidebar" title="Expand sidebar" className="absolute -right-3 top-[60px] grid h-6 w-6 place-items-center rounded-full border border-[#e1e7df] bg-white text-[#708073] shadow-sm hover:text-[#315d3d]"><ChevronRight className="h-3.5 w-3.5" /></button>}
        </div>

        <nav aria-label="Dashboard navigation" className="flex-1 overflow-y-auto px-3 py-5">
          <p className={`mb-3 px-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-[#a0aaa1] ${isCollapsed ? 'sr-only' : ''}`}>Workspace</p>
          <ul className="space-y-1.5">
            {visibleSections.map((section) => <li key={section.id}><button onClick={() => navigate(section)} title={isCollapsed ? section.label : undefined} aria-current={section.active ? 'page' : undefined} className={`group flex w-full items-center gap-3 rounded-xl text-left transition ${isCollapsed ? 'justify-center px-2 py-3' : 'px-3 py-2.5'} ${section.active ? 'bg-[#edf4ec] text-[#315b3b]' : 'text-[#69796d] hover:bg-[#f7f9f6] hover:text-[#365541]'}`}><span className={`grid h-8 w-8 shrink-0 place-items-center rounded-lg transition ${section.active ? 'bg-white text-[#4d7b53] shadow-[0_1px_4px_rgba(34,66,40,.08)]' : 'text-[#819087] group-hover:bg-white group-hover:text-[#4f7456]'}`}>{section.icon}</span>{!isCollapsed && <span className="min-w-0 flex-1"><span className="block truncate text-xs font-semibold">{section.label}</span><span className="mt-0.5 block truncate text-[10px] text-[#9aa49b]">{section.description}</span></span>}{section.active && !isCollapsed && <span className="h-1.5 w-1.5 rounded-full bg-[#6f9a70]" />}</button></li>)}
          </ul>
        </nav>

        <div className={`shrink-0 border-t border-[#edf0eb] p-3 ${isCollapsed ? 'px-2' : ''}`}>
          <button onClick={() => router.push('/dashboard/account')} title={isCollapsed ? displayName : undefined} className={`mb-1 flex w-full items-center gap-2.5 rounded-xl p-2 text-left transition hover:bg-[#f6f8f5] ${isCollapsed ? 'justify-center' : ''}`}>
            {user?.user_metadata?.avatar_url ? <img src={user.user_metadata.avatar_url} alt="" className="h-8 w-8 shrink-0 rounded-full object-cover" /> : <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[#edf3ec] text-xs font-semibold text-[#4a704e]">{initials}</span>}
            {!isCollapsed && <span className="min-w-0 flex-1"><span className="block truncate text-xs font-semibold text-[#45584a]">{displayName}</span><span className="block truncate text-[10px] text-[#98a198]">{role ? role.replace('_', ' ') : user?.email}</span></span>}
          </button>
          <button onClick={handleSignOut} title={isCollapsed ? 'Sign out' : undefined} className={`flex w-full items-center gap-2.5 rounded-xl px-2 py-2 text-xs font-medium text-[#7b8780] transition hover:bg-[#fff2ef] hover:text-[#9d4f3e] ${isCollapsed ? 'justify-center' : ''}`}><LogOut className="h-4 w-4 shrink-0" />{!isCollapsed && 'Sign out'}</button>
        </div>
      </aside>

      <nav aria-label="Dashboard navigation" className="fixed inset-x-0 bottom-0 z-40 flex h-[66px] items-stretch justify-around border-t border-[#e1e7df] bg-white/95 px-1 pb-[env(safe-area-inset-bottom)] backdrop-blur-lg md:hidden">
        {visibleSections.map((section) => <button key={section.id} onClick={() => navigate(section)} aria-current={section.active ? 'page' : undefined} className={`flex min-w-0 flex-1 flex-col items-center justify-center gap-1 px-1 text-[9px] font-medium transition ${section.active ? 'text-[#35613f]' : 'text-[#89958c]'}`}><span className={`grid h-7 w-9 place-items-center rounded-lg ${section.active ? 'bg-[#edf4ec]' : ''}`}>{section.icon}</span><span className="max-w-full truncate">{section.id === 'payment-tracking' ? 'Payments' : section.label === 'Team management' ? 'Team' : section.label}</span></button>)}
      </nav>
    </>
  )
}
