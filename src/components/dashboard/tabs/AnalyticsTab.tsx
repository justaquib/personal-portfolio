'use client'

import { useState, useEffect } from 'react'
import { Card, LoadingState, EmptyState, Alert } from '../ui'
import { useAnalytics } from '@/hooks/useDashboardData'
import { useAuth } from '@/context/AuthContext'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { MapPin, Monitor, Smartphone, Tablet, Globe, Shield, ShieldOff, Trash2 } from 'lucide-react'

export function AnalyticsTab() {
  const [timeRange, setTimeRange] = useState<'daily' | 'weekly'>('daily')
  const [activeView, setActiveView] = useState<'analytics' | 'blocked'>('analytics')
  const [blockedData, setBlockedData] = useState<any>(null)
  const [loadingBlocked, setLoadingBlocked] = useState(false)
  const { analyticsData, loading, error, fetchAnalytics } = useAnalytics()
  const { isAdmin, isSuperAdmin } = useAuth()

  useEffect(() => {
    fetchAnalytics(timeRange)
  }, [fetchAnalytics, timeRange])

  useEffect(() => {
    if (activeView === 'blocked' && (isAdmin || isSuperAdmin)) {
      fetchBlockedData()
    }
  }, [activeView, isAdmin, isSuperAdmin])

  const fetchBlockedData = async () => {
    setLoadingBlocked(true)
    try {
      const response = await fetch('/api/analytics/blocked?type=all')
      if (response.ok) {
        const data = await response.json()
        setBlockedData(data)
      }
    } catch (error) {
      console.error('Failed to fetch blocked data:', error)
    } finally {
      setLoadingBlocked(false)
    }
  }

  const handleBlockAction = async (action: string, data: any) => {
    try {
      const response = await fetch(`/api/analytics?action=${action}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      if (response.ok) {
        // Refresh data
        fetchAnalytics(timeRange)
        if (activeView === 'blocked') {
          fetchBlockedData()
        }
        alert(`${action.replace('_', ' ')} successful`)
      } else {
        alert('Action failed')
      }
    } catch (error) {
      console.error('Block action failed:', error)
      alert('Action failed')
    }
  }

  const handleDeleteData = async (visitorId: string) => {
    if (!confirm('Are you sure you want to delete all analytics data for this visitor? This action cannot be undone.')) {
      return
    }

    try {
      const response = await fetch(`/api/analytics?visitor_id=${visitorId}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        fetchAnalytics(timeRange)
        if (activeView === 'blocked') {
          fetchBlockedData()
        }
        alert('Data deleted successfully')
      } else {
        alert('Delete failed')
      }
    } catch (error) {
      console.error('Delete failed:', error)
      alert('Delete failed')
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  const formatWeek = (periodStr: string) => {
    const [year, week] = periodStr.split('-')
    return `Week ${week}, ${year}`
  }

  // Calculate totals from real data
  const data = analyticsData?.data || []
  const totalUniqueUsers = data.reduce((sum, item) => sum + item.unique_users, 0)
  const averageUniqueUsers = data.length > 0 ? Math.round(totalUniqueUsers / data.length) : 0
  const totals = analyticsData?.totals || { total_unique_users: 0, total_page_views: 0 }

  if (loading) {
    return <LoadingState message="Loading analytics data..." />
  }

  if (error) {
    return <EmptyState message={`Error loading analytics: ${error}`} />
  }

  const canManageBlocks = isAdmin || isSuperAdmin

  return (
    <div className="space-y-6">
      {/* View Switcher */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setActiveView('analytics')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeView === 'analytics'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Analytics
        </button>
        {canManageBlocks && (
          <button
            onClick={() => setActiveView('blocked')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeView === 'blocked'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Blocked Users/IPs
          </button>
        )}
      </div>

      {activeView === 'analytics' ? (
        <>
          {/* Key Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Total Unique Users">
          <div className="text-2xl font-bold text-blue-600">{totalUniqueUsers}</div>
          <p className="text-sm text-gray-500 mt-1">
            {timeRange === 'daily' ? 'Last 7 days' : 'Last 4 weeks'}
          </p>
        </Card>

        <Card title="Average Users">
          <div className="text-2xl font-bold text-green-600">{averageUniqueUsers}</div>
          <p className="text-sm text-gray-500 mt-1">
            {timeRange === 'daily' ? 'Per day' : 'Per week'}
          </p>
        </Card>

        <Card title="Peak Period">
          <div className="text-2xl font-bold text-purple-600">
            {data.length > 0 ? Math.max(...data.map(d => d.unique_users)) : 0}
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Highest {timeRange === 'daily' ? 'daily' : 'weekly'} count
          </p>
        </Card>
      </div>

      {/* Chart Controls */}
      <Card>
        <div className="flex items-center gap-4 mb-4">
          <h3 className="text-lg font-semibold">User Analytics</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setTimeRange('daily')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                timeRange === 'daily'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Daily
            </button>
            <button
              onClick={() => setTimeRange('weekly')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                timeRange === 'weekly'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Weekly
            </button>
          </div>
        </div>

        {/* Charts */}
        <div className="h-80">
          {data.length === 0 ? (
            <EmptyState message="No analytics data available yet. Data will appear as users visit your site." />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              {timeRange === 'daily' ? (
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={formatDate}
                    fontSize={12}
                  />
                  <YAxis fontSize={12} />
                  <Tooltip
                    labelFormatter={(label) => `Date: ${formatDate(label)}`}
                    formatter={(value, name) => [
                      value,
                      name === 'unique_users' ? 'Unique Users' : 'Page Views'
                    ]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="unique_users"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    name="Unique Users"
                  />
                  {data.some(d => d.page_views !== undefined) && (
                    <Line
                      type="monotone"
                      dataKey="page_views"
                      stroke="#10b981"
                      strokeWidth={2}
                      name="Page Views"
                    />
                  )}
                </LineChart>
              ) : (
                <BarChart data={data.map(item => ({
                  ...item,
                  week: item.period ? formatWeek(item.period) : 'Unknown'
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip />
                  <Legend />
                  <Bar
                    dataKey="unique_users"
                    fill="#3b82f6"
                    name="Unique Users"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              )}
            </ResponsiveContainer>
          )}
        </div>
      </Card>

      {/* Additional Stats */}
      <Card title="Recent Activity">
        <div className="space-y-3">
          {timeRange === 'daily' && data.length >= 2 && (
            <>
              <div className="flex justify-between items-center py-2 border-b">
                <span className="text-sm text-gray-600">Today</span>
                <span className="text-sm font-medium">
                  {data[data.length - 1]?.unique_users || 0} unique users
                </span>
              </div>
              <div className="flex justify-between items-center py-2 border-b">
                <span className="text-sm text-gray-600">Yesterday</span>
                <span className="text-sm font-medium">
                  {data[data.length - 2]?.unique_users || 0} unique users
                </span>
              </div>
            </>
          )}
          <div className="flex justify-between items-center py-2">
            <span className="text-sm text-gray-600">
              {timeRange === 'daily' ? 'This Week' : 'Last 30 Days'}
            </span>
            <span className="text-sm font-medium">
              {timeRange === 'daily' ? totalUniqueUsers : totals.total_unique_users} unique users
            </span>
          </div>
          <div className="flex justify-between items-center py-2 border-t pt-2">
            <span className="text-sm text-gray-600">Total Page Views</span>
            <span className="text-sm font-medium">
              {timeRange === 'daily' ? (data.reduce((sum, d) => sum + (d.page_views || 0), 0)) : totals.total_page_views}
            </span>
          </div>
        </div>
      </Card>

      {/* Recent Visitors with Location/Device Info */}
      <Card title="Recent Visitors">
        {!analyticsData?.recent_visits || analyticsData.recent_visits.length === 0 ? (
          <EmptyState message="No visitor data available yet." />
        ) : (
          <div className="space-y-3">
            {analyticsData.recent_visits.slice(0, 5).map((visit: any) => (
              <div key={visit.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  {visit.device_type === 'mobile' && <Smartphone className="w-4 h-4 text-gray-500" />}
                  {visit.device_type === 'tablet' && <Tablet className="w-4 h-4 text-gray-500" />}
                  {visit.device_type === 'desktop' && <Monitor className="w-4 h-4 text-gray-500" />}
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Visitor {visit.visitor_id.slice(0, 8)}...</span>
                      {visit.country && visit.country !== 'Unknown' && (
                        <span className="flex items-center gap-1 text-sm text-gray-500">
                          <MapPin className="w-3 h-3" />
                          {visit.city !== 'Unknown' ? `${visit.city}, ` : ''}{visit.country}
                        </span>
                      )}
                      {visit.is_blocked && (
                        <span className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded">
                          Blocked
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-500">
                      {visit.browser} on {visit.os} • {new Date(visit.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                </div>
                {canManageBlocks && (
                  <div className="flex gap-2">
                    {!visit.is_blocked ? (
                      <button
                        onClick={() => handleBlockAction('block_visitor', {
                          visitor_id: visit.visitor_id,
                          reason: 'Blocked by admin'
                        })}
                        className="p-1 text-red-600 hover:bg-red-50 rounded"
                        title="Block this visitor"
                      >
                        <Shield className="w-4 h-4" />
                      </button>
                    ) : (
                      <button
                        onClick={() => handleBlockAction('unblock_visitor', {
                          visitor_id: visit.visitor_id
                        })}
                        className="p-1 text-green-600 hover:bg-green-50 rounded"
                        title="Unblock this visitor"
                      >
                        <ShieldOff className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={() => handleDeleteData(visit.visitor_id)}
                      className="p-1 text-red-600 hover:bg-red-50 rounded"
                      title="Delete all data for this visitor"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>
      </>
      ) : (
        // Blocked Users/IPs View
        <div className="space-y-6">
          <Card title="Blocked Visitors">
            {loadingBlocked ? (
              <LoadingState message="Loading blocked data..." />
            ) : blockedData?.blocked_visitors?.length === 0 ? (
              <EmptyState message="No blocked visitors." />
            ) : (
              <div className="space-y-3">
                {blockedData?.blocked_visitors?.map((visitor: any) => (
                  <div key={visitor.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">{visitor.visitor_id}</div>
                      <div className="text-sm text-gray-500">
                        {visitor.reason || 'No reason provided'} • Blocked by {visitor.blocker?.name || visitor.blocker?.email}
                      </div>
                      <div className="text-xs text-gray-400">
                        {new Date(visitor.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleBlockAction('unblock_visitor', { visitor_id: visitor.visitor_id })}
                        className="p-1 text-green-600 hover:bg-green-50 rounded"
                        title="Unblock this visitor"
                      >
                        <ShieldOff className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDeleteData(visitor.visitor_id)}
                        className="p-1 text-red-600 hover:bg-red-50 rounded"
                        title="Delete all data for this visitor"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card title="Blocked IP Addresses">
            {loadingBlocked ? (
              <LoadingState message="Loading blocked data..." />
            ) : blockedData?.blocked_ips?.length === 0 ? (
              <EmptyState message="No blocked IP addresses." />
            ) : (
              <div className="space-y-3">
                {blockedData?.blocked_ips?.map((ip: any) => (
                  <div key={ip.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">{ip.ip_address}</div>
                      <div className="text-sm text-gray-500">
                        {ip.reason || 'No reason provided'} • Blocked by {ip.blocker?.name || ip.blocker?.email}
                      </div>
                      <div className="text-xs text-gray-400">
                        {new Date(ip.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleBlockAction('unblock_ip', { ip_address: ip.ip_address })}
                        className="p-1 text-green-600 hover:bg-green-50 rounded"
                        title="Unblock this IP"
                      >
                        <ShieldOff className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}