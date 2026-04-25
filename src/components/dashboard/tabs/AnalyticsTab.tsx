'use client'

import { useState, useEffect } from 'react'
import { Card, LoadingState, EmptyState, Alert } from '../ui'
import { useAnalytics } from '@/hooks/useDashboardData'
import { useAuth } from '@/context/AuthContext'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { MapPin, Monitor, Smartphone, Tablet, Globe, Shield, ShieldOff, Trash2, Clock, Eye, Calendar, X } from 'lucide-react'

export function AnalyticsTab() {
  const [timeRange, setTimeRange] = useState<'daily' | 'weekly'>('daily')
  const [activeView, setActiveView] = useState<'analytics' | 'blocked'>('analytics')
  const [blockedData, setBlockedData] = useState<any>(null)
  const [loadingBlocked, setLoadingBlocked] = useState(false)
  const [visitorDetails, setVisitorDetails] = useState<any>(null)
  const [loadingVisitorDetails, setLoadingVisitorDetails] = useState(false)
  const [showVisitorModal, setShowVisitorModal] = useState(false)
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
        setShowVisitorModal(false)
        alert('Data deleted successfully')
      } else {
        alert('Delete failed')
      }
    } catch (error) {
      console.error('Delete failed:', error)
      alert('Delete failed')
    }
  }

  const handleVisitorClick = async (visitorId: string) => {
    setLoadingVisitorDetails(true)
    setShowVisitorModal(true)

    try {
      const response = await fetch(`/api/analytics/visitor?visitor_id=${visitorId}`)
      if (response.ok) {
        const data = await response.json()
        setVisitorDetails(data)
      } else {
        alert('Failed to load visitor details')
        setShowVisitorModal(false)
      }
    } catch (error) {
      console.error('Failed to fetch visitor details:', error)
      alert('Failed to load visitor details')
      setShowVisitorModal(false)
    } finally {
      setLoadingVisitorDetails(false)
    }
  }

  const formatDuration = (seconds: number) => {
    if (!seconds) return 'N/A'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
    if (minutes > 0) return `${minutes}m ${secs}s`
    return `${secs}s`
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
                      <button
                        onClick={() => handleVisitorClick(visit.visitor_id)}
                        className="font-medium text-blue-600 hover:text-blue-800 hover:underline cursor-pointer"
                      >
                        Visitor {visit.visitor_id.slice(0, 8)}...
                      </button>
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

      {/* Visitor Details Side Drawer */}
      {showVisitorModal && (
        <div className="fixed inset-0 bg-black bg-opacity-10 backdrop-blur-sm z-50">
          <div className="absolute right-0 top-0 h-full w-[40%] bg-white shadow-xl overflow-y-auto border-l border-gray-200">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Visitor Details</h2>
                <button
                  onClick={() => setShowVisitorModal(false)}
                  className="p-2 hover:bg-gray-100 rounded-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {loadingVisitorDetails ? (
                <LoadingState message="Loading visitor details..." />
              ) : visitorDetails ? (
                <div className="space-y-6">
                  {/* Visitor Summary */}
                  <Card title="Visitor Summary">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-3">
                        <Eye className="w-5 h-5 text-blue-500" />
                        <div>
                          <div className="text-2xl font-bold">{visitorDetails.summary.total_visits}</div>
                          <div className="text-sm text-gray-500">Total Visits</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Calendar className="w-5 h-5 text-green-500" />
                        <div>
                          <div className="text-2xl font-bold">{visitorDetails.summary.total_sessions}</div>
                          <div className="text-sm text-gray-500">Sessions</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Clock className="w-5 h-5 text-purple-500" />
                        <div>
                          <div className="text-2xl font-bold">{formatDuration(visitorDetails.summary.total_time_spent)}</div>
                          <div className="text-sm text-gray-500">Total Time</div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-semibold mb-2">Location & Device</h4>
                        <div className="space-y-1 text-sm">
                          {visitorDetails.summary.location && (
                            <>
                              <div><strong>IP:</strong> {visitorDetails.summary.location.ip_address || 'Unknown'}</div>
                              <div><strong>Location:</strong> {visitorDetails.summary.location.city}, {visitorDetails.summary.location.region}, {visitorDetails.summary.location.country}</div>
                            </>
                          )}
                          {visitorDetails.summary.device_info && (
                            <>
                              <div><strong>Device:</strong> {visitorDetails.summary.device_info.device_type}</div>
                              <div><strong>Browser:</strong> {visitorDetails.summary.device_info.browser}</div>
                              <div><strong>OS:</strong> {visitorDetails.summary.device_info.os}</div>
                            </>
                          )}
                        </div>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Activity Summary</h4>
                        <div className="space-y-1 text-sm">
                          <div><strong>Unique Pages:</strong> {visitorDetails.summary.unique_pages}</div>
                          <div><strong>Avg Session:</strong> {formatDuration(visitorDetails.summary.average_session_time)}</div>
                          <div><strong>First Visit:</strong> {new Date(visitorDetails.summary.first_visit).toLocaleString()}</div>
                          <div><strong>Last Visit:</strong> {new Date(visitorDetails.summary.last_visit).toLocaleString()}</div>
                          {visitorDetails.summary.is_blocked && (
                            <div className="text-red-600 font-semibold">🚫 BLOCKED</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </Card>

                  {/* Sessions */}
                  <Card title="Sessions">
                    <div className="space-y-4">
                      {visitorDetails.sessions.map((session: any) => (
                        <div key={session.id} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center gap-4">
                              <div className="text-sm text-gray-500">
                                <Calendar className="w-4 h-4 inline mr-1" />
                                {new Date(session.start_time).toLocaleString()}
                              </div>
                              <div className="text-sm text-gray-500">
                                <Clock className="w-4 h-4 inline mr-1" />
                                Duration: {formatDuration(session.duration_seconds)}
                              </div>
                              <div className="text-sm text-gray-500">
                                <Eye className="w-4 h-4 inline mr-1" />
                                {session.page_views} pages
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <h5 className="font-medium">Page Flow:</h5>
                            <div className="flex flex-wrap gap-2">
                              {session.visits.map((visit: any, index: number) => (
                                <div key={visit.id} className="flex items-center gap-2 bg-gray-50 px-3 py-2 rounded">
                                  <span className="text-sm font-medium">{visit.page}</span>
                                  {visit.time_on_page && (
                                    <span className="text-xs text-gray-500">
                                      ({formatDuration(visit.time_on_page)})
                                    </span>
                                  )}
                                  {index < session.visits.length - 1 && (
                                    <span className="text-gray-400">→</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>

                  {/* All Visits Table */}
                  <Card title="All Visits">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2">Time</th>
                            <th className="text-left py-2">Page</th>
                            <th className="text-left py-2">Time on Page</th>
                            <th className="text-left py-2">IP</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visitorDetails.all_visits.map((visit: any) => (
                            <tr key={visit.id} className="border-b">
                              <td className="py-2">{new Date(visit.timestamp).toLocaleString()}</td>
                              <td className="py-2">{visit.page}</td>
                              <td className="py-2">{visit.time_on_page ? formatDuration(visit.time_on_page) : 'N/A'}</td>
                              <td className="py-2">{visit.ip_address || 'Unknown'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>

                  {/* Admin Actions */}
                  {canManageBlocks && (
                    <Card title="Admin Actions">
                      <div className="flex gap-4">
                        {!visitorDetails.summary.is_blocked ? (
                          <button
                            onClick={() => handleBlockAction('block_visitor', {
                              visitor_id: visitorDetails.summary.visitor_id,
                              reason: 'Blocked by admin from visitor details'
                            })}
                            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                          >
                            <Shield className="w-4 h-4" />
                            Block Visitor
                          </button>
                        ) : (
                          <button
                            onClick={() => handleBlockAction('unblock_visitor', {
                              visitor_id: visitorDetails.summary.visitor_id
                            })}
                            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                          >
                            <ShieldOff className="w-4 h-4" />
                            Unblock Visitor
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteData(visitorDetails.summary.visitor_id)}
                          className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete All Data
                        </button>
                      </div>
                    </Card>
                  )}
                </div>
              ) : (
                <EmptyState message="Failed to load visitor details" />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Click outside to close drawer */}
      {showVisitorModal && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowVisitorModal(false)}
        />
      )}
    </div>
  )
}