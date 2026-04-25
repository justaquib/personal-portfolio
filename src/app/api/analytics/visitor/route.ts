import { NextRequest, NextResponse } from 'next/server'
import { createAdminClient } from '@/lib/supabase/admin'

// GET: Get detailed information about a specific visitor
export async function GET(request: NextRequest) {
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

    // Get all visits for this visitor
    const { data: visits, error: visitsError } = await supabase
      .from('analytics_visits')
      .select('*')
      .eq('visitor_id', visitorId)
      .order('timestamp', { ascending: false })

    if (visitsError) throw visitsError

    // Get session information
    const { data: sessions, error: sessionsError } = await supabase
      .from('analytics_sessions')
      .select('*')
      .eq('visitor_id', visitorId)
      .order('start_time', { ascending: false })

    if (sessionsError) throw sessionsError

    // Calculate session durations and page times
    const enrichedSessions = sessions?.map(session => {
      const sessionVisits = visits?.filter(v => v.session_id === session.session_id) || []
      const sortedVisits = sessionVisits.sort((a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      )

      // Calculate time spent on each page
      const visitsWithDuration = sortedVisits.map((visit, index) => {
        if (index < sortedVisits.length - 1) {
          const nextVisit = sortedVisits[index + 1]
          const timeSpent = Math.floor(
            (new Date(nextVisit.timestamp).getTime() - new Date(visit.timestamp).getTime()) / 1000
          )
          return { ...visit, time_on_page: Math.min(timeSpent, 3600) } // Cap at 1 hour
        }
        return { ...visit, time_on_page: null } // Last page, no next visit
      })

      // Calculate total session duration
      const firstVisit = sortedVisits[0]
      const lastVisit = sortedVisits[sortedVisits.length - 1]
      const sessionDuration = firstVisit && lastVisit ?
        Math.floor((new Date(lastVisit.timestamp).getTime() - new Date(firstVisit.timestamp).getTime()) / 1000) : 0

      return {
        ...session,
        duration_seconds: sessionDuration,
        visits: visitsWithDuration
      }
    }) || []

    // Get visitor summary
    const totalVisits = visits?.length || 0
    const uniquePages = new Set(visits?.map(v => v.page) || []).size
    const totalSessionTime = enrichedSessions.reduce((sum, s) => sum + (s.duration_seconds || 0), 0)
    const avgSessionTime = enrichedSessions.length > 0 ? Math.floor(totalSessionTime / enrichedSessions.length) : 0

    // Get latest visit info for summary
    const latestVisit = visits?.[0]
    const locationInfo = latestVisit ? {
      country: latestVisit.country,
      city: latestVisit.city,
      region: latestVisit.region,
      ip_address: latestVisit.ip_address
    } : null

    // Get user info if this is an authenticated visitor
    const userInfo = latestVisit?.user_info || null

    const summary = {
      visitor_id: visitorId,
      visitor_type: latestVisit?.visitor_type || 'anonymous',
      total_visits: totalVisits,
      total_sessions: enrichedSessions.length,
      unique_pages: uniquePages,
      total_time_spent: totalSessionTime,
      average_session_time: avgSessionTime,
      first_visit: visits?.[visits.length - 1]?.timestamp,
      last_visit: latestVisit?.timestamp,
      location: locationInfo,
      device_info: latestVisit ? {
        device_type: latestVisit.device_type,
        browser: latestVisit.browser,
        os: latestVisit.os
      } : null,
      user_info: userInfo,
      is_blocked: latestVisit?.is_blocked || false
    }

    return NextResponse.json({
      summary,
      sessions: enrichedSessions,
      all_visits: visits
    })

  } catch (error) {
    console.error('Visitor details GET error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch visitor details' },
      { status: 500 }
    )
  }
}