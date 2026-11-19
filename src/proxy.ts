import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'
import { DEMO_SESSION_COOKIE, verifyDemoSessionToken } from '@/lib/demo-session'

export async function proxy(request: NextRequest) {
  const path = request.nextUrl.pathname
  const demoSession = verifyDemoSessionToken(request.cookies.get(DEMO_SESSION_COOKIE)?.value)

  // Demo credentials only open the isolated, sample-data demo route.
  if (path.startsWith('/dashboard/demo')) {
    if (demoSession) return NextResponse.next()
    const url = request.nextUrl.clone()
    url.pathname = '/login'
    url.searchParams.set('demo', '1')
    return NextResponse.redirect(url)
  }

  if (demoSession && path === '/login') {
    const url = request.nextUrl.clone()
    url.pathname = '/dashboard/demo'
    url.search = ''
    return NextResponse.redirect(url)
  }

  let supabaseResponse = NextResponse.next({
    request,
  })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet: Array<{ name: string; value: string; options?: any }>) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          supabaseResponse = NextResponse.next({
            request,
          })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  const {
    data: { user },
  } = await supabase.auth.getUser()

  // If logged in and trying to access /login, redirect to /dashboard
  if (user && request.nextUrl.pathname === '/login') {
    const url = request.nextUrl.clone()
    url.pathname = '/dashboard'
    return NextResponse.redirect(url)
  }

  // If not logged in and trying to access /dashboard, redirect to /login
  if (!user && request.nextUrl.pathname.startsWith('/dashboard')) {
    const url = request.nextUrl.clone()
    url.pathname = '/login'
    return NextResponse.redirect(url)
  }

  return supabaseResponse
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|assets|prototypes|api).*)',
  ],
}
