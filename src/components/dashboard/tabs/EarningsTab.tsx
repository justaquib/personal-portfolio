'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Badge } from '../ui'
import { usePayments, useSubscriptions, useServices, useContacts } from '@/hooks/useDashboardData'
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
  const { contacts, loading: contactsLoading, fetchContacts } = useContacts()
  
  const [filterYear, setFilterYear] = useState<string>('all')
  const [earningsTab, setEarningsTab] = useState<'service' | 'contact'>('service')

  useEffect(() => {
    fetchPayments()
    fetchSubscriptions()
    fetchServices()
    fetchContacts()
  }, [fetchPayments, fetchSubscriptions, fetchServices, fetchContacts])

  const loading = paymentsLoading || subscriptionsLoading || servicesLoading || contactsLoading

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

  // Calculate monthly earnings for the selected year
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

  // Total earnings for selected year (or all years)
  const totalEarnings = Object.values(monthlyEarnings).reduce((sum, val) => sum + val, 0)
  
  // Total Revenue, Cost, and Due - calculated from main invoice records (including sub-invoices)
  // Using the old logic: each main invoice has one cost (service.actual_cost)
  const mainInvoices = payments.filter(p => !p.sub_invoice_id)
  
  // Filter by year if not "all"
  const filteredMainInvoices = filterYear === 'all' 
    ? mainInvoices 
    : mainInvoices.filter(p => {
        const parsedDate = parseDate(p.payment_date)
        return parsedDate && parsedDate.isValid() && parsedDate.year().toString() === filterYear.toString()
      })
  
  let totalRevenue = 0
  let totalCost = 0
  let remainingDue = 0
  
  // Calculate earnings by service using same logic as summary cards
  const serviceEarningsMap: Record<string, { revenue: number; cost: number; due: number; lastInvoiceDate: string | null }> = {}
  
  // Calculate earnings by contact using same logic as summary cards
  const contactEarningsMap: Record<string, { revenue: number; cost: number; due: number; lastInvoiceDate: string | null }> = {}
  
  filteredMainInvoices.forEach(payment => {
    const subscription = subscriptions.find(s => s.id === payment.subscription_id)
    if (!subscription) return
    
    const service = services.find(s => s.id === subscription.service_id)
    if (!service) return
    
    const contact = contacts.find(c => c.id === subscription.contact_id)
    const serviceName = service.name || 'Unknown Service'
    const contactName = contact?.name || 'Unknown Contact'
    
    // Get all sub-invoices for this main invoice
    const subInvoices = payments.filter(p => p.invoice_id === payment.invoice_id && p.sub_invoice_id)
    
    // Calculate total paid: main invoice + all sub-invoices
    const mainPaid = payment.amount_paid || 0
    const subInvoicesPaid = subInvoices.reduce((sum, sub) => sum + (sub.amount_paid || 0), 0)
    const totalPaid = mainPaid + subInvoicesPaid
    
    // Use main invoice's amount_due as the total due
    const mainDue = payment.amount_due || 0
    
    // Cost applies to the main invoice (service cost)
    const cost = service.actual_cost || 0
    
    // Revenue = total paid
    const revenue = totalPaid
    
    // Calculate remaining due
    const invoiceDue = totalPaid >= mainDue ? 0 : (mainDue - totalPaid)
    
    // Initialize and add to service totals
    if (!serviceEarningsMap[serviceName]) {
      serviceEarningsMap[serviceName] = { revenue: 0, cost: 0, due: 0, lastInvoiceDate: null }
    }
    serviceEarningsMap[serviceName].revenue += revenue
    serviceEarningsMap[serviceName].cost += cost
    serviceEarningsMap[serviceName].due += invoiceDue
    // Update last invoice date if this payment is newer
    if (payment.payment_date) {
      const currentLastDate = serviceEarningsMap[serviceName].lastInvoiceDate
      if (!currentLastDate || payment.payment_date > currentLastDate) {
        serviceEarningsMap[serviceName].lastInvoiceDate = payment.payment_date
      }
    }
    
    // Initialize and add to contact totals
    if (!contactEarningsMap[contactName]) {
      contactEarningsMap[contactName] = { revenue: 0, cost: 0, due: 0, lastInvoiceDate: null }
    }
    contactEarningsMap[contactName].revenue += revenue
    contactEarningsMap[contactName].cost += cost
    contactEarningsMap[contactName].due += invoiceDue
    // Update last invoice date if this payment is newer
    if (payment.payment_date) {
      const currentLastDate = contactEarningsMap[contactName].lastInvoiceDate
      if (!currentLastDate || payment.payment_date > currentLastDate) {
        contactEarningsMap[contactName].lastInvoiceDate = payment.payment_date
      }
    }
    
    // Add to totals
    totalRevenue += revenue
    totalCost += cost
    remainingDue += invoiceDue
  })
  
  // Net Profit - Revenue minus Cost
  const netProfit = totalRevenue - totalCost
  
  // Outstanding - same as remaining due
  const outstanding = remainingDue
  
  // Total Received (same as totalRevenue for backwards compatibility)
  const totalReceivedFromPayments = totalRevenue

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

        {/* Chart - Yearly for All Years, Monthly for specific year */}
        <div className="mb-8">
          <h4 className="text-sm font-medium text-gray-600 mb-4">
            {filterYear === 'all' ? 'Yearly Earnings' : `Monthly Earnings (${filterYear})`}
          </h4>
          <div className="flex items-end justify-between gap-2 h-48">
            {filterYear === 'all' ? (
              // Show yearly chart (12 years from 2025 forwards)
              (() => {
                const displayYears = []
                for (let i = 0; i < 12; i++) {
                  const year = String(2025 + i)
                  displayYears.push(year)
                }
                return displayYears.map(year => {
                  const amount = yearlyEarnings[year] || 0
                  const maxAmount = Math.max(...Object.values(yearlyEarnings), 1)
                  const height = maxAmount > 0 ? Math.max((amount / maxAmount) * 100, amount > 0 ? 8 : 0) : 0
                  
                  return (
                    <div key={year} className="flex-1 flex flex-col items-center">
                      {amount > 0 && (
                        <span className="text-xs font-medium text-gray-700 mb-1">₹{amount.toLocaleString()}</span>
                      )}
                      <div 
                        className="w-full bg-purple-500 rounded-t transition-all hover:bg-purple-600"
                        style={{ height: `${height}%`, minHeight: amount > 0 ? '8px' : '0' }}
                        title={`₹{amount.toLocaleString()}`}
                      />
                      <span className="text-xs text-gray-500 mt-2">{year}</span>
                    </div>
                  )
                })
              })()
            ) : (
              // Show monthly chart for selected year
              months.map((month, index) => {
                const monthKey = `${filterYear}-${String(index + 1).padStart(2, '0')}`
                const amount = monthlyEarnings[monthKey] || 0
                const maxAmount = Math.max(...Object.values(monthlyEarnings), 1)
                const height = maxAmount > 0 ? Math.max((amount / maxAmount) * 100, amount > 0 ? 8 : 0) : 0
                
                return (
                  <div key={month} className="flex-1 flex flex-col items-center">
                    {amount > 0 && (
                      <span className="text-xs font-medium text-gray-700 mb-1">₹{amount.toLocaleString()}</span>
                    )}
                    <div 
                      className="w-full bg-purple-500 rounded-t transition-all hover:bg-purple-600"
                      style={{ height: `${height}%`, minHeight: amount > 0 ? '8px' : '0' }}
                      title={`₹{amount.toLocaleString()}`}
                    />
                    <span className="text-xs text-gray-500 mt-2">{month}</span>
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Earnings Tab Navigation */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setEarningsTab('service')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              earningsTab === 'service'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Service
          </button>
          <button
            onClick={() => setEarningsTab('contact')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              earningsTab === 'contact'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Contact
          </button>
        </div>

        {/* Earnings by Service */}
        {earningsTab === 'service' && (
        <div>
          <h4 className="text-sm font-medium text-gray-600 mb-4">Earnings by Service</h4>
          {Object.keys(serviceEarningsMap).length === 0 ? (
            <EmptyState message="No earnings data available" />
          ) : (
            <div className="space-y-3">
              {Object.entries(serviceEarningsMap).map(([service, data]) => {
                const profit = data.revenue - data.cost
                
                return (
                  <div key={service} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium text-gray-900">{service}</span>
                      <p className="text-xs text-gray-500">
                        Cost: ₹{data.cost.toLocaleString()} | Revenue: ₹{data.revenue.toLocaleString()}
                        {data.lastInvoiceDate && ` | Last: ${data.lastInvoiceDate}`}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-gray-900">₹{data.revenue.toLocaleString()}</div>
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
        )}

        {/* Earnings by Contact */}
        {earningsTab === 'contact' && (
        <div>
          <h4 className="text-sm font-medium text-gray-600 mb-4">Earnings by Contact</h4>
          {Object.keys(contactEarningsMap).length === 0 ? (
            <EmptyState message="No earnings data available" />
          ) : (
            <div className="space-y-3">
              {Object.entries(contactEarningsMap).map(([contact, data]) => {
                const profit = data.revenue - data.cost
                
                return (
                  <div key={contact} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <span className="font-medium text-gray-900">{contact}</span>
                      <p className="text-xs text-gray-500">
                        Cost: ₹{data.cost.toLocaleString()} | Revenue: ₹{data.revenue.toLocaleString()}
                        {data.lastInvoiceDate && ` | Last: ${data.lastInvoiceDate}`}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold text-gray-900">₹{data.revenue.toLocaleString()}</div>
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
        )}
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
