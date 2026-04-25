import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

// GET: Fetch analytics data
export async function GET(request: NextRequest) {
  try {
    const supabase = await createAdminClient()
    const { searchParams } = new URL(request.url)
    const period = searchParams.get('period') || 'daily' // 'daily' or 'weekly'

    let data: any[]

    if (period === 'weekly') {
      // Group by week for last 4 weeks
      const { data: weeklyData, error } = await supabase
        .from('analytics_visits')
        .select('visit_date, visitor_id')
        .gte('visit_date', new Date(Date.now() - 4 * 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0])
        .order('visit_date', { ascending: false })

      if (error) throw error

      // Group by week
      const weeklyMap = new Map<string, Set<string>>()
      weeklyData?.forEach(visit => {
        const date = new Date(visit.visit_date)
        const year = date.getFullYear()
        const week = Math.ceil((date.getDate() - date.getDay() + 1) / 7)
        const weekKey = `${year}-W${week.toString().padStart(2, '0')}`

        if (!weeklyMap.has(weekKey)) {
          weeklyMap.set(weekKey, new Set())
        }
        weeklyMap.get(weekKey)!.add(visit.visitor_id)
      })

      data = Array.from(weeklyMap.entries())
        .map(([period, visitors]) => ({
          period,
          unique_users: visitors.size
        }))
        .sort((a, b) => b.period.localeCompare(a.period))
        .slice(0, 4)

    } else {
      // Daily data for last 7 days
      const { data: dailyData, error } = await supabase
        .from('analytics_visits')
        .select('visit_date, visitor_id')
        .gte('visit_date', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0])
        .order('visit_date', { ascending: true })

      if (error) throw error

      // Group by date
      const dailyMap = new Map<string, { unique_users: Set<string>; page_views: number }>()
      dailyData?.forEach(visit => {
        const date = visit.visit_date
        if (!dailyMap.has(date)) {
          dailyMap.set(date, { unique_users: new Set(), page_views: 0 })
        }
        dailyMap.get(date)!.unique_users.add(visit.visitor_id)
        dailyMap.get(date)!.page_views++
      })

      data = Array.from(dailyMap.entries())
        .map(([date, stats]) => ({
          date,
          unique_users: stats.unique_users.size,
          page_views: stats.page_views
        }))
        .sort((a, b) => a.date.localeCompare(b.date))
    }

    // Get total stats for last 30 days
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]

    const { data: totalData, error: totalError } = await supabase
      .from('analytics_visits')
      .select('visitor_id')
      .gte('visit_date', thirtyDaysAgo)

    if (totalError) throw totalError

    const uniqueVisitors = new Set(totalData?.map(d => d.visitor_id) || [])
    const totals = {
      total_unique_users: uniqueVisitors.size,
      total_page_views: totalData?.length || 0
    }

    // Get recent visits with details for the recent visitors section
    const { data: recentVisits, error: recentError } = await supabase
      .from('analytics_visits')
      .select('*')
      .order('timestamp', { ascending: false })
      .limit(10)

    if (recentError) throw recentError

    return NextResponse.json({
      data,
      totals,
      period,
      recent_visits: recentVisits || []
    })

  } catch (error) {
    console.error('Analytics GET error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch analytics data' },
      { status: 500 }
    )
  }
}

// PUT: Block or unblock a visitor/IP
export async function PUT(request: NextRequest) {
  try {
    const supabase = await createAdminClient()
    const { searchParams } = new URL(request.url)
    const action = searchParams.get('action') // 'block_visitor', 'unblock_visitor', 'block_ip', 'unblock_ip'
    const body = await request.json()

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

    if (action === 'block_visitor') {
      const { visitor_id, reason } = body

      // Add to blocked_visitors table
      const { error } = await supabase
        .from('blocked_visitors')
        .insert({
          visitor_id,
          reason,
          blocked_by: user.id
        })

      if (error && !error.message.includes('duplicate key value')) {
        throw error
      }

      // Update existing visits
      await supabase
        .from('analytics_visits')
        .update({
          is_blocked: true,
          blocked_reason: reason,
          blocked_by: user.id,
          blocked_at: new Date().toISOString()
        })
        .eq('visitor_id', visitor_id)

    } else if (action === 'unblock_visitor') {
      const { visitor_id } = body

      // Remove from blocked_visitors table
      await supabase
        .from('blocked_visitors')
        .delete()
        .eq('visitor_id', visitor_id)

      // Update existing visits
      await supabase
        .from('analytics_visits')
        .update({
          is_blocked: false,
          blocked_reason: null,
          blocked_by: null,
          blocked_at: null
        })
        .eq('visitor_id', visitor_id)

    } else if (action === 'block_ip') {
      const { ip_address, reason } = body

      // Add to blocked_ips table
      const { error } = await supabase
        .from('blocked_ips')
        .insert({
          ip_address,
          reason,
          blocked_by: user.id
        })

      if (error && !error.message.includes('duplicate key value')) {
        throw error
      }

      // Update existing visits
      await supabase
        .from('analytics_visits')
        .update({
          is_blocked: true,
          blocked_reason: reason,
          blocked_by: user.id,
          blocked_at: new Date().toISOString()
        })
        .eq('ip_address', ip_address)

    } else if (action === 'unblock_ip') {
      const { ip_address } = body

      // Remove from blocked_ips table
      await supabase
        .from('blocked_ips')
        .delete()
        .eq('ip_address', ip_address)

      // Update existing visits
      await supabase
        .from('analytics_visits')
        .update({
          is_blocked: false,
          blocked_reason: null,
          blocked_by: null,
          blocked_at: null
        })
        .eq('ip_address', ip_address)
    }

    return NextResponse.json({ success: true })

  } catch (error) {
    console.error('Analytics PUT error:', error)
    return NextResponse.json(
      { error: 'Failed to update block status' },
      { status: 500 }
    )
  }
}

// DELETE: Remove analytics data (for GDPR compliance)
export async function DELETE(request: NextRequest) {
  try {
    const supabase = await createAdminClient()
    const { searchParams } = new URL(request.url)
    const visitorId = searchParams.get('visitor_id')

    if (!visitorId) {
      return NextResponse.json({ error: 'Visitor ID required' }, { status: 400 })
    }

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

    // Delete all visits for this visitor
    await supabase
      .from('analytics_visits')
      .delete()
      .eq('visitor_id', visitorId)

    return NextResponse.json({ success: true })

  } catch (error) {
    console.error('Analytics DELETE error:', error)
    return NextResponse.json(
      { error: 'Failed to delete analytics data' },
      { status: 500 }
    )
  }
}

// Helper function to parse user agent
function parseUserAgent(userAgent: string) {
  const ua = userAgent.toLowerCase()

  // Device type detection
  let deviceType = 'desktop'
  if (/mobile|android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(ua)) {
    deviceType = /ipad|tablet/i.test(ua) ? 'tablet' : 'mobile'
  }

  // Browser detection
  let browser = 'Unknown'
  if (ua.includes('chrome') && !ua.includes('edg')) browser = 'Chrome'
  else if (ua.includes('firefox')) browser = 'Firefox'
  else if (ua.includes('safari') && !ua.includes('chrome')) browser = 'Safari'
  else if (ua.includes('edg')) browser = 'Edge'
  else if (ua.includes('opera')) browser = 'Opera'
  else if (ua.includes('msie') || ua.includes('trident')) browser = 'Internet Explorer'

  // OS detection
  let os = 'Unknown'
  if (ua.includes('windows')) os = 'Windows'
  else if (ua.includes('macintosh') || ua.includes('mac os x')) os = 'macOS'
  else if (ua.includes('linux')) os = 'Linux'
  else if (ua.includes('android')) os = 'Android'
  else if (ua.includes('ios') || ua.includes('iphone') || ua.includes('ipad')) os = 'iOS'

  return { deviceType, browser, os }
}

// Helper function to get location data from IP
async function getLocationFromIP(ip: string) {
  try {
    if (ip === 'unknown' || ip === '127.0.0.1' || ip === '::1') {
      return { country: 'Local', city: 'Local', region: 'Local', latitude: null, longitude: null }
    }

    const response = await fetch(`http://ip-api.com/json/${ip}?fields=country,city,region,lat,lon`, {
      timeout: 3000 // 3 second timeout
    })

    if (!response.ok) {
      throw new Error('Geolocation API failed')
    }

    const data = await response.json()

    if (data.status === 'fail') {
      return { country: 'Unknown', city: 'Unknown', region: 'Unknown', latitude: null, longitude: null }
    }

    return {
      country: data.country || 'Unknown',
      city: data.city || 'Unknown',
      region: data.region || 'Unknown',
      latitude: data.lat,
      longitude: data.lon
    }
  } catch (error) {
    console.warn('Failed to get location data:', error)
    return { country: 'Unknown', city: 'Unknown', region: 'Unknown', latitude: null, longitude: null }
  }
}

// POST: Track a page visit
export async function POST(request: NextRequest) {
  try {
    const supabase = await createAdminClient()
    const body = await request.json()
    const { page = '/', referrer } = body

    // Get IP address from headers
    const forwarded = request.headers.get('x-forwarded-for')
    const ip = forwarded?.split(',')[0] || request.headers.get('x-real-ip') || 'unknown'

    // Get user agent
    const userAgent = request.headers.get('user-agent') || 'unknown'

    // Check if IP is blocked
    if (ip !== 'unknown') {
      const { data: blockedIP } = await supabase
        .from('blocked_ips')
        .select('id')
        .eq('ip_address', ip)
        .single()

      if (blockedIP) {
        return NextResponse.json(
          { error: 'Access blocked' },
          { status: 403 }
        )
      }
    }

    // Generate visitor ID based on IP and user agent
    const visitorId = Buffer.from(`${ip}-${userAgent.slice(0, 50)}`).toString('base64').slice(0, 32)

    // Check if visitor is blocked
    const { data: blockedVisitor } = await supabase
      .from('blocked_visitors')
      .select('id')
      .eq('visitor_id', visitorId)
      .single()

    if (blockedVisitor) {
      return NextResponse.json(
        { error: 'Access blocked' },
        { status: 403 }
      )
    }

    // Parse user agent for device/browser info
    const { deviceType, browser, os } = parseUserAgent(userAgent)

    // Get location data
    const locationData = await getLocationFromIP(ip)

    // Insert visit record (will fail silently if duplicate for same day due to unique constraint)
    const { error } = await supabase
      .from('analytics_visits')
      .insert({
        visitor_id: visitorId,
        page,
        user_agent: userAgent,
        ip_address: ip !== 'unknown' ? ip : null,
        referrer,
        visit_date: new Date().toISOString().split('T')[0],
        ...locationData,
        device_type: deviceType,
        browser,
        os
      })

    // Ignore unique constraint violations (visitor already tracked today)
    if (error && !error.message.includes('duplicate key value')) {
      throw error
    }

    return NextResponse.json({ success: true })

  } catch (error) {
    console.error('Analytics POST error:', error)
    return NextResponse.json(
      { error: 'Failed to track visit' },
      { status: 500 }
    )
  }
}