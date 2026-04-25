import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

const supabase = createClient()

// GET: Fetch analytics data
export async function GET(request: NextRequest) {
  try {
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

    return NextResponse.json({
      data,
      totals,
      period
    })

  } catch (error) {
    console.error('Analytics GET error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch analytics data' },
      { status: 500 }
    )
  }
}

// POST: Track a page visit
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { page = '/', referrer } = body

    // Get IP address from headers
    const forwarded = request.headers.get('x-forwarded-for')
    const ip = forwarded?.split(',')[0] || request.headers.get('x-real-ip') || 'unknown'

    // Get user agent
    const userAgent = request.headers.get('user-agent') || 'unknown'

    // Generate visitor ID based on IP and user agent (simplified approach)
    // In production, you'd want a more sophisticated tracking method
    const visitorId = Buffer.from(`${ip}-${userAgent.slice(0, 50)}`).toString('base64').slice(0, 32)

    // Insert visit record (will fail silently if duplicate for same day due to unique constraint)
    const { error } = await supabase
      .from('analytics_visits')
      .insert({
        visitor_id: visitorId,
        page,
        user_agent: userAgent,
        ip_address: ip !== 'unknown' ? ip : null,
        referrer,
        visit_date: new Date().toISOString().split('T')[0]
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