'use client'

import { useEffect, useState, type FormEvent } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { ArrowRight, Check, Eye, EyeOff, LayoutDashboard, LockKeyhole, ShieldCheck } from 'lucide-react'

type LoginMode = 'account' | 'demo'

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [isSignUp, setIsSignUp] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [demoUsername, setDemoUsername] = useState('justaquib')
  const [demoPassword, setDemoPassword] = useState('')
  const [demoError, setDemoError] = useState<string | null>(null)
  const [demoLoading, setDemoLoading] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [mode, setMode] = useState<LoginMode>('account')
  const { signInWithEmail, signUpWithEmail, user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get('demo') === '1') setMode('demo')
  }, [])

  useEffect(() => {
    if (!loading && user) router.replace('/dashboard')
  }, [user, loading, router])

  const changeMode = (nextMode: LoginMode) => {
    setMode(nextMode)
    setError(null)
    setDemoError(null)
    setSuccess(null)
  }

  const handleEmailSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setIsLoading(true)
    setError(null)
    setSuccess(null)

    if (!email.trim() || !password.trim()) {
      setError('Enter your email and password to continue.')
      setIsLoading(false)
      return
    }

    try {
      if (isSignUp) {
        await signUpWithEmail(email.trim(), password)
        setSuccess('Your account is ready. Sign in with your new credentials.')
        setIsSignUp(false)
        setPassword('')
      } else {
        await signInWithEmail(email.trim(), password)
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Authentication failed. Please try again.'
      if (message.includes('Invalid login credentials')) setError('That email and password combination was not recognized.')
      else if (message.includes('Email not confirmed')) setError('Confirm your email address before signing in.')
      else if (message.includes('User already registered')) setError('An account with this email already exists. Sign in instead.')
      else if (message.includes('Password should be at least')) setError('Use a password with at least six characters.')
      else setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDemoSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setDemoLoading(true)
    setDemoError(null)
    try {
      const response = await fetch('/api/demo-auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: demoUsername, password: demoPassword }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Demo sign-in failed.')
      router.replace('/dashboard/demo')
    } catch (err) {
      setDemoError(err instanceof Error ? err.message : 'Demo sign-in failed. Please try again.')
    } finally {
      setDemoLoading(false)
    }
  }

  if (loading) {
    return <main className="grid min-h-screen place-items-center bg-[#f5f6f2]"><div className="h-9 w-9 animate-spin rounded-full border-[3px] border-[#dce8dc] border-t-[#4d7954]" /></main>
  }

  if (user) return null

  const inputClass = 'h-12 w-full rounded-xl border border-[#dfe7df] bg-white px-3.5 text-sm text-[#33483a] outline-none transition placeholder:text-[#a0aaa2] focus:border-[#a5c0a7] focus:ring-4 focus:ring-[#eaf2e9]'

  return (
    <main className="min-h-screen overflow-hidden bg-[#f5f6f2] text-[#263b2e]">
      <div className="mx-auto flex min-h-screen max-w-[1320px] flex-col px-4 sm:px-7 lg:px-10">
        <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-[#e1e6de]">
          <a href="/" className="flex items-center gap-2.5" aria-label="Just Aquib home">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-[#254f36] text-white"><LayoutDashboard className="h-4 w-4" /></span>
            <span className="text-sm font-semibold tracking-tight text-[#2d4433]">Just Aquib</span>
          </a>
          <span className="hidden items-center gap-1.5 text-[11px] font-medium text-[#7f8c82] sm:flex"><LockKeyhole className="h-3.5 w-3.5" />Secure workspace access</span>
        </header>

        <div className="grid flex-1 items-center gap-10 py-8 lg:grid-cols-[minmax(0,1fr)_440px] lg:gap-20 lg:py-12">
          <section className="mx-auto w-full max-w-xl lg:mx-0">
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-[#d9e5d8] bg-white px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-[#67806a]"><span className="h-1.5 w-1.5 rounded-full bg-[#75a077]" />Personal workspace</p>
            <h1 className="max-w-lg text-4xl font-semibold leading-[1.08] tracking-[-0.055em] text-[#21382b] sm:text-5xl">A clear view of the work behind the work.</h1>
            <p className="mt-5 max-w-md text-sm leading-6 text-[#738178] sm:text-base sm:leading-7">Sign in to manage payments, clients, analytics, and the tools that keep your work moving.</p>
            <div className="mt-8 grid max-w-lg gap-3 sm:grid-cols-3">
              {[
                { title: 'Manage', text: 'Payments and clients' },
                { title: 'Review', text: 'Activity and analytics' },
                { title: 'Build', text: 'Resumes and tools' },
              ].map((item) => <div key={item.title} className="rounded-xl border border-[#e2e8e0] bg-white/80 p-3.5"><span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#5b7a5f]">{item.title}</span><p className="mt-1.5 text-xs leading-5 text-[#7d8a80]">{item.text}</p></div>)}
            </div>
          </section>

          <section className="mx-auto w-full max-w-[440px] rounded-2xl border border-[#e0e7df] bg-white p-5 shadow-[0_22px_70px_-44px_rgba(30,66,42,.35)] sm:p-7">
            <div className="grid grid-cols-2 rounded-xl bg-[#f3f6f2] p-1" role="tablist" aria-label="Sign-in type">
              <button type="button" role="tab" aria-selected={mode === 'account'} onClick={() => changeMode('account')} className={`h-10 rounded-lg text-xs font-semibold transition ${mode === 'account' ? 'bg-white text-[#2f5138] shadow-[0_1px_5px_rgba(36,67,42,.1)]' : 'text-[#7f8b81] hover:text-[#45634a]'}`}>Account sign in</button>
              <button type="button" role="tab" aria-selected={mode === 'demo'} onClick={() => changeMode('demo')} className={`h-10 rounded-lg text-xs font-semibold transition ${mode === 'demo' ? 'bg-white text-[#2f5138] shadow-[0_1px_5px_rgba(36,67,42,.1)]' : 'text-[#7f8b81] hover:text-[#45634a]'}`}>Demo access</button>
            </div>

            {mode === 'account' ? (
              <div className="pt-6" role="tabpanel">
                <div className="mb-5"><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#78907a]">{isSignUp ? 'New account' : 'Welcome back'}</p><h2 className="mt-1.5 text-xl font-semibold tracking-tight text-[#2a4031]">{isSignUp ? 'Create your account' : 'Sign in to your workspace'}</h2><p className="mt-1.5 text-xs leading-5 text-[#849087]">{isSignUp ? 'Set up access to your dashboard.' : 'Use your Supabase account credentials.'}</p></div>
                {error && <div role="alert" className="mb-4 rounded-xl border border-[#efd2c9] bg-[#fff6f3] px-3.5 py-3 text-xs leading-5 text-[#984d3d]">{error}</div>}
                {success && <div role="status" className="mb-4 rounded-xl border border-[#d5e5d4] bg-[#f0f7ef] px-3.5 py-3 text-xs leading-5 text-[#416c48]">{success}</div>}
                <form onSubmit={handleEmailSignIn} className="space-y-4">
                  <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Email address</span><input autoComplete="email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" required className={inputClass} disabled={isLoading} /></label>
                  <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Password</span><span className="relative block"><input autoComplete={isSignUp ? 'new-password' : 'current-password'} type={showPassword ? 'text' : 'password'} value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" required className={`${inputClass} pr-12`} disabled={isLoading} /><button type="button" onClick={() => setShowPassword((visible) => !visible)} aria-label={showPassword ? 'Hide password' : 'Show password'} className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-[#8b978d] hover:bg-[#f2f6f1] hover:text-[#41664a]">{showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button></span></label>
                  <button type="submit" disabled={isLoading} className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#254f36] px-4 text-sm font-semibold text-white transition hover:bg-[#1d412c] disabled:cursor-wait disabled:opacity-60">{isLoading ? 'Please wait…' : isSignUp ? 'Create account' : 'Sign in'}{!isLoading && <ArrowRight className="h-4 w-4" />}</button>
                </form>
                <p className="mt-5 text-center text-xs text-[#829087]">{isSignUp ? 'Already have an account?' : 'Need an account?'} <button type="button" onClick={() => { setIsSignUp((value) => !value); setError(null); setSuccess(null); }} className="font-semibold text-[#47734f] hover:text-[#2f5938]">{isSignUp ? 'Sign in' : 'Create one'}</button></p>
              </div>
            ) : (
              <div className="pt-6" role="tabpanel">
                <div className="mb-5"><p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#78907a]">Isolated preview</p><h2 className="mt-1.5 text-xl font-semibold tracking-tight text-[#2a4031]">Enter the demo workspace</h2><p className="mt-1.5 text-xs leading-5 text-[#849087]">Read-only dashboard preview with sample data. Your account records stay separate.</p></div>
                {demoError && <div role="alert" className="mb-4 rounded-xl border border-[#efd2c9] bg-[#fff6f3] px-3.5 py-3 text-xs leading-5 text-[#984d3d]">{demoError}</div>}
                <form onSubmit={handleDemoSignIn} className="space-y-4">
                  <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Username</span><input autoComplete="username" value={demoUsername} onChange={(event) => setDemoUsername(event.target.value)} required className={inputClass} disabled={demoLoading} /></label>
                  <label className="block"><span className="mb-1.5 block text-xs font-semibold text-[#53665a]">Demo password</span><span className="relative block"><input autoComplete="current-password" type={showPassword ? 'text' : 'password'} value={demoPassword} onChange={(event) => setDemoPassword(event.target.value)} placeholder="Enter demo password" required className={`${inputClass} pr-12`} disabled={demoLoading} /><button type="button" onClick={() => setShowPassword((visible) => !visible)} aria-label={showPassword ? 'Hide password' : 'Show password'} className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-[#8b978d] hover:bg-[#f2f6f1] hover:text-[#41664a]">{showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}</button></span></label>
                  <button type="submit" disabled={demoLoading} className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#254f36] px-4 text-sm font-semibold text-white transition hover:bg-[#1d412c] disabled:cursor-wait disabled:opacity-60">{demoLoading ? 'Opening demo…' : 'Open demo dashboard'}{!demoLoading && <ArrowRight className="h-4 w-4" />}</button>
                </form>
                <p className="mt-5 flex items-center justify-center gap-1.5 text-[11px] text-[#87938a]"><ShieldCheck className="h-3.5 w-3.5 text-[#6c936f]" />Demo session expires after four hours</p>
              </div>
            )}
          </section>
        </div>

        <footer className="flex min-h-12 items-center justify-center border-t border-[#e1e6de] py-3 text-[10px] text-[#9aa49b]">© {new Date().getFullYear()} Just Aquib · Private workspace</footer>
      </div>
    </main>
  )
}
