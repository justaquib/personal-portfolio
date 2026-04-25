import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

// GET: Get blocked users and IPs
export async function GET(request: NextRequest) {
  try {
    const supabase = await createAdminClient()
    const { searchParams } = new URL(request.url)
    const type = searchParams.get('type') // 'visitors', 'ips', or 'all'

    // Check if user is admin
    const { data: { user }, error: userError } = await supabase.auth.getUser()
    if (userError || !user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const { data: userRole } = await supabase
      .from('user_roles')
      .select('role')
      .eq('user_id', user.id)
      .single()

    if (!userRole || !['admin', 'super_admin'].includes(userRole.role)) {
      return NextResponse.json({ error: 'Admin access required' }, { status: 403 })
    }

    let blockedVisitors = []
    let blockedIPs = []

    if (type === 'visitors' || type === 'all') {
      const { data: visitors, error: visitorsError } = await supabase
        .from('blocked_visitors')
        .select(`
          *,
          blocker:user_roles!blocked_visitors_blocked_by_fkey(name, email)
        `)
        .order('created_at', { ascending: false })

      if (visitorsError) throw visitorsError
      blockedVisitors = visitors || []
    }

    if (type === 'ips' || type === 'all') {
      const { data: ips, error: ipsError } = await supabase
        .from('blocked_ips')
        .select(`
          *,
          blocker:user_roles!blocked_ips_blocked_by_fkey(name, email)
        `)
        .order('created_at', { ascending: false })

      if (ipsError) throw ipsError
      blockedIPs = ips || []
    }

    return NextResponse.json({
      blocked_visitors: blockedVisitors,
      blocked_ips: blockedIPs
    })

  } catch (error) {
    console.error('Blocked items GET error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch blocked items' },
      { status: 500 }
    )
  }
}