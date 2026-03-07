'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Badge } from '../ui'
import { usePayments, useSubscriptions, useServices } from '@/hooks/useDashboardData'
import dayjs from 'dayjs'
import customParseFormat from 'dayjs/plugin/customParseFormat'

dayjs.extend(customParseFormat)

interface EarningsTabProps {
  userId: string
}

export function EarningsTab({ userId }: EarningsTabProps) {
  const { payments, loading: paymentsLoading, fetchPayments } = usePayments()
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions } = useSubscriptions()
  const { services, loading: servicesLoading, fetchServices } = useServices()
  
  const [filterYear, setFilterYear] = useState<string>('all')

  useEffect(() => {
    fetchPayments()
    fetchSubscriptions()
    fetchServices()
  }, [fetchPayments, fetchSubscriptions, fetchServices])

  const loading = paymentsLoading || subscriptionsLoading || servicesLoading

  const getServiceName = (serviceId: string) => {
    const service = services.find(s => s.id === serviceId)
    return service?.name || 'Unknown Service'
  }

  // Helper to parse date strings in DD/MM/YYYY or YYYY-MM-DD format
  const parseDate = (dateStr: string | undefined | null): dayjs.Dayjs | null => {
    if (!dateStr) return null
    // Try parsing as DD/MM/YYYY first (Indian format)
    let parsed = dayjs(dateStr, 'DD/MM/YYYY', true)
    if (parsed.isValid()) return parsed
    // Try parsing as YYYY-MM-DD (ISO format)
    parsed = dayjs(dateStr, 'YYYY-MM-DD', true)
    if (parsed.isValid()) return parsed
    // Fallback to default parsing
    return dayjs(dateStr)
  }

  // Calculate earnings by year (only for payments with actual payment dates)
  const yearlyEarnings = payments.reduce((acc: Record<string, number>, payment) => {
    const parsedDate = parseDate(payment.payment_date)
    if (parsedDate && parsedDate.isValid()) {
      const year = parsedDate.year().toString()
      acc[year] = (acc[year] || 0) + (payment.amount_paid || 0)
    }
    return acc
  }, {})

  // Calculate monthly earnings for the selected year (or all years)
  const monthlyEarnings = payments.reduce((acc: Record<string, number>, payment) => {
    const parsedDate = parseDate(payment.payment_date)
    if (parsedDate && parsedDate.isValid()) {
      const paymentYear = parsedDate.year().toString()
      if (filterYear === 'all' || paymentYear === filterYear.toString()) {
        const monthKey = parsedDate.format('YYYY-MM')
        acc[monthKey] = (acc[monthKey] || 0) + (payment.amount_paid || 0)
      }
    }
    return acc
  }, {})

  // Calculate earnings by service (for selected year or all years)
  const earningsByService = payments.reduce((acc: Record<string, number>, payment) => {
    const parsedDate = parseDate(payment.payment_date)
    if (parsedDate && parsedDate.isValid() && (filterYear === 'all' || parsedDate.year().toString() === filterYear.toString())) {
      const subscription = subscriptions.find(s => s.id === payment.subscription_id)
      if (subscription) {
        const serviceName = getServiceName(subscription.service_id)
        acc[serviceName] = (acc[serviceName] || 0) + (payment.amount_paid || 0)
      }
    }
    return acc
  }, {})

  // Calculate profit (revenue - costs)
  const calculateProfit = (serviceId: string) => {
    const service = services.find(s => s.id === serviceId)
    const revenue = earningsByService[service?.name || ''] || 0
    const cost = service?.actual_cost || 0
    return revenue - cost
  }

  // Total earnings for selected year (or all years)
  const totalEarnings = Object.values(monthlyEarnings).reduce((sum, val) => sum + val, 0)
  
  // Total Revenue - sum of all amount_paid for selected year (or all years if no filter)
  const yearFilteredPayments = filterYear === 'all' 
    ? payments 
    : payments.filter(p => {
        const parsedDate = parseDate(p.payment_date)
        return parsedDate && parsedDate.isValid() && parsedDate.year().toString() === filterYear.toString()
      })
  
  const totalRevenue = yearFilteredPayments.reduce((sum, p) => sum + (p.amount_paid || 0), 0)
  
  // Total Cost - sum of actual_cost from unique services that have payments (subscriptions)
  const uniqueServiceIds = new Set<string>()
  yearFilteredPayments.forEach(payment => {
    const subscription = subscriptions.find(s => s.id === payment.subscription_id)
    if (subscription) {
      uniqueServiceIds.add(subscription.service_id)
    }
  })
  
  const totalCost = Array.from(uniqueServiceIds).reduce((sum, serviceId) => {
    const service = services.find(s => s.id === serviceId)
    return sum + (service?.actual_cost || 0)
  }, 0)
  
  // Net Profit - Revenue minus Cost
  const netProfit = totalRevenue - totalCost
  
  // Total Received - sum of all amount_paid for selected year
  const totalReceivedFromPayments = yearFilteredPayments.reduce((sum, p) => sum + (p.amount_paid || 0), 0)
  
  // Remaining Due - properly calculate including sub-invoices
  // For each main invoice: remaining = amount_due - (main_invoice_paid + all_sub_invoices_paid)
  const remainingDue = payments
    .filter(p => !p.sub_invoice_id) // Only main invoices
    .reduce((sum, mainInvoice) => {
      // Find all sub-invoices for this main invoice
      const subInvoices = payments.filter(p => p.invoice_id === mainInvoice.invoice_id && p.sub_invoice_id)
      const subInvoicesTotal = subInvoices.reduce((s, p) => s + (p.amount_paid || 0), 0)
      const totalPaidForInvoice = (mainInvoice.amount_paid || 0) + subInvoicesTotal
      const remainingForInvoice = Math.max(0, (mainInvoice.amount_due || 0) - totalPaidForInvoice)
      return sum + remainingForInvoice
    }, 0)
  
  // Outstanding - same as remaining due (for backwards compatibility)
  const outstanding = remainingDue

  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
  ]

  if (loading) {
    return (
      <div className="p-6">
        <LoadingState />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="bg-green-500 text-white p-4">
          <div className="text-xs opacity-90">Total Revenue</div>
          <div className="text-xl font-bold mt-1">₹{totalRevenue.toLocaleString()}</div>
        </Card>
        <Card className="bg-red-500 text-white p-4">
          <div className="text-xs opacity-90">Total Cost</div>
          <div className="text-xl font-bold mt-1">₹{totalCost.toLocaleString()}</div>
        </Card>
        <Card className="bg-purple-500 text-white p-4">
          <div className="text-xs opacity-90">Net Profit</div>
          <div className="text-xl font-bold mt-1">₹{netProfit.toLocaleString()}</div>
        </Card>
        <Card className="bg-amber-500 text-white p-4">
          <div className="text-xs opacity-90">Total Due</div>
          <div className="text-xl font-bold mt-1">₹{remainingDue.toLocaleString()}</div>
        </Card>
        <Card className="bg-blue-500 text-white p-4">
          <div className="text-xs opacity-90">Outstanding</div>
          <div className="text-xl font-bold mt-1">₹{outstanding.toLocaleString()}</div>
        </Card>
      </div>

      {/* Year Filter */}
      <Card>
        <div className="flex items-center gap-4 mb-6">
          <label className="text-sm text-gray-600">Year:</label>
          <select
            value={filterYear}
            onChange={(e) => setFilterYear(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500"
          >
            <option value="all">All Years</option>
            {Object.keys(yearlyEarnings).sort().reverse().map(year => (
              <option key={year} value={year}>{year}</option>
            ))}
            {Object.keys(yearlyEarnings).length === 0 && (
              <option value={new Date().getFullYear()}>{new Date().getFullYear()}</option>
            )}
          </select>
        </div>

        {/* Monthly Chart */}
        <div className="mb-8">
          <h4 className="text-sm font-medium text-gray-600 mb-4">Monthly Earnings ({filterYear === 'all' ? 'All Years' : filterYear})</h4>
          <div className="flex items-end justify-between gap-2 h-48">
            {months.map((month, index) => {
              const monthKey = `${filterYear}-${String(index + 1).padStart(2, '0')}`
              const amount = monthlyEarnings[monthKey] || 0
              const maxAmount = Math.max(...Object.values(monthlyEarnings), 1)
              // Use a minimum height of 4px for visibility, and scale to fill the container better
              const height = maxAmount > 0 ? Math.max((amount / maxAmount) * 100, amount > 0 ? 8 : 0) : 0
              
              return (
                <div key={month} className="flex-1 flex flex-col items-center">
                  {amount > 0 && (
                    <span className="text-xs font-medium text-gray-700 mb-1">₹{amount.toLocaleString()}</span>
                  )}
                  <div 
                    className="w-full bg-purple-500 rounded-t transition-all hover:bg-purple-600"
                    style={{ height: `${height}%`, minHeight: amount > 0 ? '8px' : '0' }}
                    title={`₹${amount.toLocaleString()}`}
                  />
                  <span className="text-xs text-gray-500 mt-2">{month}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Earnings by Service */}
        <div>
          <h4 className="text-sm font-medium text-gray-600 mb-4">Earnings by Service</h4>
          {Object.keys(earningsByService).length === 0 ? (
            <EmptyState message="No earnings data available" />
          ) : (
            <div className="space-y-3">
              {Object.entries(earningsByService).map(([service, earnings]) => {
                const serviceData = services.find(s => s.name === service)
                const profit = earnings - (serviceData?.actual_cost || 0)
                
                return (
                  <div key={service} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium text-gray-900">{service}</span>
                      <p className="text-xs text-gray-500">
                        Cost: ₹{serviceData?.actual_cost || 0} | Revenue: ₹{earnings}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-gray-900">₹{earnings.toLocaleString()}</div>
                      <div className={`text-xs ${profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        Profit: ₹{profit.toLocaleString()}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </Card>

      {/* Recent Payments */}
      <Card title="Recent Transactions">
        {payments.length === 0 ? (
          <EmptyState message="No transactions yet" />
        ) : (
          <div className="space-y-2">
            {payments
              .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
              .slice(0, 10)
              .map((payment) => {
                const subscription = subscriptions.find(s => s.id === payment.subscription_id)
                return (
                  <div key={payment.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${payment.amount_paid >= payment.amount_due ? 'bg-green-500' : 'bg-amber-500'}`} />
                      <div>
                        <p className="font-medium text-gray-900">
                          {subscription ? getServiceName(subscription.service_id) : 'Unknown'}
                        </p>
                        <p className="text-xs text-gray-500">
                          {payment.payment_date ? new Date(payment.payment_date).toLocaleDateString() : 'N/A'}
                          {payment.invoice_id && ` • Invoice #${payment.invoice_id}`}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold text-gray-900">₹{payment.amount_paid.toLocaleString()}</p>
                      <Badge variant={payment.remaining_due > 0 ? 'warning' : 'success'}>
                        {payment.remaining_due > 0 ? `Due: ₹${payment.remaining_due}` : 'Paid'}
                      </Badge>
                    </div>
                  </div>
                )
              })}
          </div>
        )}
      </Card>
    </div>
  )
}
