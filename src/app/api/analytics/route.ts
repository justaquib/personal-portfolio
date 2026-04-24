import { NextRequest, NextResponse } from 'next/server'
import Database from 'better-sqlite3'
import path from 'path'

// Database setup
const dbPath = path.join(process.cwd(), 'taskflow.db')
const db = new Database(dbPath)

// Initialize analytics table if it doesn't exist
db.exec(`
  CREATE TABLE IF NOT EXISTS analytics_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id TEXT NOT NULL,
    page TEXT NOT NULL,
    user_agent TEXT,
    ip_address TEXT,
    referrer TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(visitor_id, DATE(timestamp))
  )
`)

// GET: Fetch analytics data
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const period = searchParams.get('period') || 'daily' // 'daily' or 'weekly'

    let query: string
    let params: any[] = []

    if (period === 'weekly') {
      // Group by week
      query = `
        SELECT
          strftime('%Y-%W', timestamp) as period,
          COUNT(DISTINCT visitor_id) as unique_users
        FROM analytics_visits
        WHERE timestamp >= date('now', '-4 weeks')
        GROUP BY strftime('%Y-%W', timestamp)
        ORDER BY period DESC
        LIMIT 4
      `
    } else {
      // Daily data for last 7 days
      query = `
        SELECT
          DATE(timestamp) as date,
          COUNT(DISTINCT visitor_id) as unique_users,
          COUNT(*) as page_views
        FROM analytics_visits
        WHERE timestamp >= date('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date ASC
      `
    }

    const stmt = db.prepare(query)
    const data = stmt.all(...params)

    // Get total stats
    const totalStmt = db.prepare(`
      SELECT
        COUNT(DISTINCT visitor_id) as total_unique_users,
        COUNT(*) as total_page_views
      FROM analytics_visits
      WHERE timestamp >= date('now', '-30 days')
    `)
    const totals = totalStmt.get() as { total_unique_users: number; total_page_views: number }

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

    // Insert visit record (with UNIQUE constraint to avoid duplicates per day)
    const stmt = db.prepare(`
      INSERT OR REPLACE INTO analytics_visits
      (visitor_id, page, user_agent, ip_address, referrer, timestamp)
      VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `)

    stmt.run(visitorId, page, userAgent, ip, referrer)

    return NextResponse.json({ success: true })

  } catch (error) {
    console.error('Analytics POST error:', error)
    return NextResponse.json(
      { error: 'Failed to track visit' },
      { status: 500 }
    )
  }
}