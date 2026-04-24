'use client'

import { useState, useEffect } from 'react'
import { Card, LoadingState, EmptyState } from '../ui'
import { useAnalytics } from '@/hooks/useDashboardData'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts'

export function AnalyticsTab() {
  const [timeRange, setTimeRange] = useState<'daily' | 'weekly'>('daily')
  const { analyticsData, loading, error, fetchAnalytics } = useAnalytics()

  useEffect(() => {
    fetchAnalytics(timeRange)
  }, [fetchAnalytics, timeRange])

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

  return (
    <div className="space-y-6">
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
    </div>
  )
}