'use client'

import { LogOut } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'

export function DashboardTopbar() {
  const { user, signOut } = useAuth()
  const router = useRouter()
  const name = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Account'

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  return (
    <header className="dashboard-topbar">
      <span className="mr-auto hidden text-[10px] font-semibold uppercase tracking-[0.16em] text-[#8a978c] sm:block">Personal workspace</span>
      <span className="max-w-[42vw] truncate text-xs font-medium text-[#5d6e61]">{name}</span>
      <button onClick={handleSignOut} className="inline-flex items-center gap-2 rounded-lg border border-[#e0e7df] bg-white px-3 py-2 text-xs font-medium text-[#6c7b70] transition hover:bg-[#f6f8f5] hover:text-[#3e5945]"><LogOut className="h-3.5 w-3.5" /><span className="hidden sm:inline">Sign out</span></button>
    </header>
  )
}
