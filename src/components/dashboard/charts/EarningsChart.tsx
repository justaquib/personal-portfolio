'use client'

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'

interface EarningsChartProps {
  data: Record<string, number>
  type: 'yearly' | 'monthly'
  filterYear: string
}

const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function EarningsChart({ data, type, filterYear }: EarningsChartProps) {
  // Transform data into array format for Recharts
  const chartData = type === 'yearly' 
    ? (() => {
        const displayYears = []
        for (let i = 0; i < 12; i++) {
          const year = String(2025 + i)
          displayYears.push({
            name: year,
            value: data[year] || 0
          })
        }
        return displayYears
      })()
    : months.map((month, index) => {
        const monthKey = `${filterYear}-${String(index + 1).padStart(2, '0')}`
        return {
          name: month,
          value: data[monthKey] || 0
        }
      })

  const formatYAxis = (value: number) => {
    if (value >= 100000) return `₹${(value / 100000).toFixed(1)}L`
    if (value >= 1000) return `₹${(value / 1000).toFixed(0)}K`
    return `₹${value}`
  }

  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number }>; label?: string }) => {
    if (active && payload && payload.length && payload[0].value > 0) {
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
          <p className="text-sm font-medium text-gray-700">{label}</p>
          <p className="text-sm font-semibold text-gray-900">
            ₹{payload[0].value.toLocaleString()}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart
        data={chartData}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <CartesianGrid 
          strokeDasharray="3 3" 
          vertical={false} 
          stroke="#e5e7eb"
        />
        <XAxis 
          dataKey="name" 
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 12, fill: '#6b7280' }}
          dy={10}
        />
        <YAxis 
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 12, fill: '#6b7280' }}
          tickFormatter={formatYAxis}
          dx={-10}
        />
        <Tooltip 
          content={<CustomTooltip />}
          cursor={{ fill: 'rgba(0, 0, 0, 0.05)' }}
        />
        <Bar 
          dataKey="value" 
          radius={[4, 4, 0, 0]}
          maxBarSize={50}
        >
          {chartData.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.value > 0 ? '#212529' : '#e5e7eb'} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
