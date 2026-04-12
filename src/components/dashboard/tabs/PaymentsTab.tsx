'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Badge, Button, Modal } from '../ui'
import { useSubscriptions, usePayments, useServices, useContacts } from '@/hooks/useDashboardData'
import { PaymentForm } from '../PaymentForm'
import { Subscription, Payment, Service } from '@/types/database'
import { truncateText } from '@/utils/misc/stringUtils'
import dayjs from 'dayjs'
import customParseFormat from 'dayjs/plugin/customParseFormat'
import relativeTime from 'dayjs/plugin/relativeTime'

dayjs.extend(customParseFormat)
dayjs.extend(relativeTime)

interface PaymentsTabProps {
  userId: string
}

export function PaymentsTab({ userId }: PaymentsTabProps) {
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions, deleteSubscription } = useSubscriptions()
  const { payments, loading: paymentsLoading, fetchPayments, savePayment, deletePayment } = usePayments()
  const { services, loading: servicesLoading, fetchServices } = useServices()
  const { contacts, loading: contactsLoading, fetchContacts } = useContacts()
  
  const [showPaymentModal, setShowPaymentModal] = useState(false)
  const [showSubscriptionModal, setShowSubscriptionModal] = useState(false)
  const [editingSubscription, setEditingSubscription] = useState<Subscription | null>(null)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [subscriptionToDelete, setSubscriptionToDelete] = useState<Subscription | null>(null)
  const [infoSubscriptionId, setInfoSubscriptionId] = useState<string | null>(null)
  const [paymentsPage, setPaymentsPage] = useState(1)
  const paymentsPerPage = 10
  const [viewingPayment, setViewingPayment] = useState<Payment | null>(null)
  const [editingPayment, setEditingPayment] = useState<Payment | null>(null)
  const [deletingPayment, setDeletingPayment] = useState<Payment | null>(null)
  const [parentInvoice, setParentInvoice] = useState<Payment | null>(null)
  const [showSubInvoices, setShowSubInvoices] = useState(true)
  const [viewingFromSubInvoice, setViewingFromSubInvoice] = useState<Payment | null>(null)
  const [statusFilter, setStatusFilter] = useState<'all' | 'paid' | 'partial' | 'unpaid'>('all')

  useEffect(() => {
    fetchSubscriptions()
    fetchPayments()
    fetchServices()
    fetchContacts()
  }, [fetchSubscriptions, fetchPayments, fetchServices, fetchContacts])

  const loading = subscriptionsLoading || paymentsLoading || servicesLoading || contactsLoading

  const getServiceName = (serviceId: string) => {
    const service = services.find((s: Service) => s.id === serviceId)
    return service?.name || 'Unknown Service'
  }

  const getService = (serviceId: string) => {
    return services.find((s: Service) => s.id === serviceId)
  }

  const getContactName = (contactId: string) => {
    const sub = subscriptions.find(s => s.contact_id === contactId)
    return sub?.contact?.name || 'Unknown Contact'
  }

  const calculateTotalPaid = (subscriptionId: string) => {
    return payments
      .filter(p => p.subscription_id === subscriptionId)
      .reduce((sum, p) => sum + (p.amount_paid || 0), 0)
  }

  const getPaymentCycle = (subscription: Subscription): 'monthly' | 'quarterly' | 'yearly' => {
    const service = getService(subscription.service_id)
    return service?.payment_cycle || 'monthly'
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

  // Calculate when the next payment is due
  const getNextPaymentDate = (subscription: Subscription): Date => {
    const parsedStartDate = parseDate(subscription.started_at)
    if (!parsedStartDate) return new Date()
    
    const startDate = parsedStartDate
    const now = dayjs()
    const paymentCycle = getPaymentCycle(subscription)
    
    // Calculate the next payment date from start date
    let nextPayment = startDate
    
    while (nextPayment.isBefore(now) || nextPayment.isSame(now)) {
      if (paymentCycle === 'monthly') {
        nextPayment = nextPayment.add(1, 'month')
      } else if (paymentCycle === 'quarterly') {
        nextPayment = nextPayment.add(3, 'month')
      } else if (paymentCycle === 'yearly') {
        nextPayment = nextPayment.add(1, 'year')
      }
    }
    
    return nextPayment.toDate()
  }

  // Check if payment is due based on payment cycle
  const isPaymentDue = (subscription: Subscription): boolean => {
    const paymentCycle = getPaymentCycle(subscription)

    // For monthly and quarterly, check if current period is paid
    if (paymentCycle === 'monthly' || paymentCycle === 'quarterly') {
      const service = getService(subscription.service_id)
      const serviceAmount = service?.amount || 0

      const startDate = parseDate(subscription.started_at)
      if (!startDate) return false

      const now = dayjs()

      // Calculate expected amount based on payment cycle
      const expectedAmount = paymentCycle === 'monthly' ? serviceAmount :
                            paymentCycle === 'quarterly' ? serviceAmount * 3 :
                            serviceAmount * 12

      // Check if payment was made for current period
      const subPayments = payments.filter(p => p.subscription_id === subscription.id)

      // Find current billing period
      let periodStart = startDate
      let periodEnd = startDate.add(paymentCycle === 'monthly' ? 1 : paymentCycle === 'quarterly' ? 3 : 1, paymentCycle === 'monthly' ? 'month' : paymentCycle === 'quarterly' ? 'month' : 'year')

      while (periodEnd.isBefore(now)) {
        periodStart = periodEnd
        periodEnd = periodEnd.add(paymentCycle === 'monthly' ? 1 : paymentCycle === 'quarterly' ? 3 : 1, paymentCycle === 'monthly' ? 'month' : paymentCycle === 'quarterly' ? 'month' : 'year')
      }

      const periodPayment = subPayments.find(p => {
        if (!p.payment_month) return false
        const paymentDate = dayjs(p.payment_month)
        return (paymentDate.isAfter(periodStart) || paymentDate.isSame(periodStart, 'day')) &&
               paymentDate.isBefore(periodEnd)
      })

      if (!periodPayment) {
        return true // No payment for current period
      }

      // Payment made for this period
      return periodPayment.amount_paid < expectedAmount
    }

    // For yearly, due only when past the due date
    const startDate = parseDate(subscription.started_at)
    if (!startDate) return false

    const now = dayjs()

    // Calculate the next payment date
    let nextPayment = startDate
    while (nextPayment.isBefore(now)) {
      nextPayment = nextPayment.add(1, 'year')
    }

    const daysUntilDue = nextPayment.diff(now, 'day')

    // For yearly, due if past due date
    return daysUntilDue <= 0
  }

  const calculateRemaining = (subscription: Subscription) => {
    const paymentCycle = getPaymentCycle(subscription)
    const service = getService(subscription.service_id)
    const serviceAmount = service?.amount || 0
    
    // Calculate expected amount based on payment cycle
    let expectedAmount = serviceAmount
    if (paymentCycle === 'quarterly') {
      expectedAmount = serviceAmount * 3
    } else if (paymentCycle === 'yearly') {
      expectedAmount = serviceAmount * 12
    }
    
    const totalPaid = calculateTotalPaid(subscription.id)
    return expectedAmount - totalPaid
  }

  const handleSavePayment = async (data: any, id?: string) => {
    await savePayment({
      ...data,
      user_id: userId,
      // For new payments, generate invoice_id if not provided
      ...(id ? {} : { invoice_id: data.invoice_id || `INV-${Date.now()}` })
    }, id)
    await fetchSubscriptions() // Refresh subscriptions to update total_due
    setShowPaymentModal(false)
    setEditingPayment(null)
    setParentInvoice(null)
  }

  const handleSaveSubscription = async (data: any) => {
    // Subscription management would require additional API endpoint
    setShowSubscriptionModal(false)
    setEditingSubscription(null)
  }

  const handleEditSubscription = (subscription: Subscription) => {
    setEditingSubscription(subscription)
    setShowSubscriptionModal(true)
  }

  const handleDeleteClick = (subscription: Subscription) => {
    setSubscriptionToDelete(subscription)
    setDeleteModalOpen(true)
  }

  const handleConfirmDelete = async () => {
    if (subscriptionToDelete) {
      try {
        await deleteSubscription(subscriptionToDelete.id)
      } catch (err) {
        console.error('Failed to delete subscription:', err)
      }
    }
    setDeleteModalOpen(false)
    setSubscriptionToDelete(null)
  }

  const handleDeletePayment = async () => {
    if (deletingPayment) {
      try {
        await deletePayment(deletingPayment.id)
      } catch (err) {
        console.error('Failed to delete payment:', err)
      }
    }
    setDeletingPayment(null)
  }

  const currentMonth = new Date().toISOString().slice(0, 7)

  // Helper to get sub-invoice count for an invoice
  const getSubInvoiceCount = (invoiceId: string | undefined) => {
    if (!invoiceId) return 0
    return payments.filter(p => p.invoice_id === invoiceId && p.sub_invoice_id).length
  }

  // Helper to get sub-invoices for an invoice
  const getSubInvoices = (invoiceId: string | undefined) => {
    if (!invoiceId) return []
    return payments.filter(p => p.invoice_id === invoiceId && p.sub_invoice_id)
  }

  // Helper to get the main invoice for a sub-invoice
  const getMainInvoice = (invoiceId: string | undefined, subInvoiceId: string | null | undefined) => {
    if (!invoiceId || !subInvoiceId) return null
    return payments.find(p => p.invoice_id === invoiceId && !p.sub_invoice_id)
  }

  // Helper to calculate total paid for an invoice (including all sub-invoices)
  const getTotalPaidForInvoice = (invoiceId: string | undefined) => {
    if (!invoiceId) return 0
    const mainInvoice = payments.find(p => p.invoice_id === invoiceId && !p.sub_invoice_id)
    const subInvoices = getSubInvoices(invoiceId)
    const mainPaid = mainInvoice?.amount_paid || 0
    const subPaid = subInvoices.reduce((sum, p) => sum + (p.amount_paid || 0), 0)
    return mainPaid + subPaid
  }

  // Helper to get combined status for an invoice
  const getCombinedStatus = (invoiceId: string | undefined) => {
    if (!invoiceId) return 'unpaid'
    const mainInvoice = payments.find(p => p.invoice_id === invoiceId && !p.sub_invoice_id)
    const totalPaid = getTotalPaidForInvoice(invoiceId)
    const amountDue = mainInvoice?.amount_due || 0
    
    if (totalPaid >= amountDue && amountDue > 0) return 'paid'
    if (totalPaid > 0) return 'partial'
    return 'unpaid'
  }

  // Sort payments by invoice number descending (newest first)
  const sortedPayments = [...payments].sort((a: Payment, b: Payment) => {
    const aNum = parseInt(a.invoice_id || '0') || 0
    const bNum = parseInt(b.invoice_id || '0') || 0
    return bNum - aNum
  })

  // Filter payments based on showSubInvoices toggle
  const filteredPayments = showSubInvoices 
    ? sortedPayments 
    : sortedPayments.filter(p => !p.sub_invoice_id)

  // Apply status filter
  const statusFilteredPayments = statusFilter === 'all' 
    ? filteredPayments 
    : filteredPayments.filter(p => {
        // For main invoices with sub-invoices, use combined status
        const subInvoiceCount = !p.sub_invoice_id ? getSubInvoiceCount(p.invoice_id) : 0
        const hasSubInvoices = subInvoiceCount > 0
        const isHidingSubInvoices = !showSubInvoices && !p.sub_invoice_id && hasSubInvoices
        const displayStatus = isHidingSubInvoices ? getCombinedStatus(p.invoice_id) : p.payment_status
        return displayStatus === statusFilter
      })

  // Reset page when filter changes
  useEffect(() => {
    setPaymentsPage(1)
  }, [showSubInvoices, statusFilter])

  if (loading) {
    return (
      <div className="p-6">
        <LoadingState />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Subscriptions */}
      <Card 
        title="Active Subscriptions"
        actions={
          <Button onClick={() => { setEditingSubscription(null); setShowSubscriptionModal(true); }}>
            Add Subscription
          </Button>
        }
      >
        {subscriptions.length === 0 ? (
          <EmptyState 
            message="No active subscriptions"
            action={
              <Button onClick={() => setShowSubscriptionModal(true)}>Add Subscription</Button>
            }
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {subscriptions.map((subscription: Subscription) => {
              const service = getService(subscription.service_id)
              const totalPaid = calculateTotalPaid(subscription.id)
              const remaining = calculateRemaining(subscription)
               // Calculate expected total paid based on time since start
               const startDate = parseDate(subscription.started_at)
               const monthsSinceStart = startDate ? dayjs().diff(startDate, 'month') + 1 : 1 // +1 to include current month
               const svc = getService(subscription.service_id)
               const amount = svc?.amount || 0
               const cycle = svc?.payment_cycle || 'monthly'
               let periodsPassed = monthsSinceStart
               if (cycle === 'quarterly') {
                 periodsPassed = Math.floor(monthsSinceStart / 3)
               } else if (cycle === 'yearly') {
                 periodsPassed = Math.floor(monthsSinceStart / 12)
               }
               const expectedTotal = periodsPassed * amount
               const paymentPercentage = expectedTotal > 0 ? Math.round((totalPaid / expectedTotal) * 100) : 0
              const paymentCycle = getPaymentCycle(subscription)
              const isDue = isPaymentDue(subscription)
              
              return (
                <div key={subscription.id} className="p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="font-medium text-gray-900 truncate" title={getServiceName(subscription.service_id)}>{truncateText(getServiceName(subscription.service_id), 25)}</h4>
                      <p className="text-xs text-gray-500 mt-1">
                        {getContactName(subscription.contact_id)}
                      </p>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() => handleEditSubscription(subscription)}
                        className="p-1.5 rounded"
                        style={{ color: '#6c757d' }}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </button>
                      <button
                        onClick={() => handleDeleteClick(subscription)}
                        className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">{paymentCycle.charAt(0).toUpperCase() + paymentCycle.slice(1)}</span>
                      <span className="text-gray-600">Paid: ₹{totalPaid}</span>
                    </div>
                     <div className="w-full bg-gray-200 rounded-full h-2">
                       <div
                         className={`h-2 rounded-full transition-all ${
                           isDue ? 'bg-orange-500' : 'bg-green-500'
                         }`}
                         style={{ width: isDue ? `${paymentPercentage}%` : '100%' }}
                       />
                     </div>
                     <div className="flex justify-between text-xs items-center">
                       <span className={isDue ? 'text-red-600' : 'text-green-600'}>
                         {isDue ? 'Payment Due' : 'Paid'}
                       </span>
                      <button
                        onClick={() => setInfoSubscriptionId(infoSubscriptionId === subscription.id ? null : subscription.id)}
                        className="p-1 text-gray-400 hover:text-gray-600"
                        title="More info"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </button>
                    </div>
                    {infoSubscriptionId === subscription.id && (
                      <div className="mt-2 p-2 bg-blue-50 rounded-lg text-xs text-blue-700">
                        {isDue ? (
                          <p>
                            <strong>Payment Due:</strong> Payment for {paymentCycle} cycle is pending.
                            {paymentCycle === 'monthly' && ` Expected: ₹${service?.amount || 0}`}
                            {paymentCycle === 'quarterly' && ` Expected: ₹${(service?.amount || 0) * 3}`}
                            {paymentCycle === 'yearly' && ` Expected: ₹${(service?.amount || 0) * 12}`}
                          </p>
                        ) : (
                          <p>
                            <strong>Paid:</strong> {paymentCycle.charAt(0).toUpperCase() + paymentCycle.slice(1)} payment received.
                          </p>
                        )}
                        {subscription.started_at && (
                          <p className="mt-1">Started: {parseDate(subscription.started_at)?.format('DD/MM/YYYY') || subscription.started_at}</p>
                        )}
                        <p className="mt-1">
                          <strong>Next payment:</strong> {dayjs(getNextPaymentDate(subscription)).format('DD/MM/YYYY')}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Card>

      {/* Recent Payments */}
      <Card 
        title="Recent Payments"
        actions={
          <div className="flex items-center gap-3">
            {/* Status Filter */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as 'all' | 'paid' | 'partial' | 'unpaid')}
              className="px-3 py-1.5 text-sm border rounded-lg focus:ring-2"
              style={{ borderColor: '#ced4da', backgroundColor: '#f8f9fa', color: '#212529' }}
            >
              <option value="all">All Status</option>
              <option value="paid">Paid</option>
              <option value="partial">Partial</option>
              <option value="unpaid">Unpaid</option>
            </select>

            {/* Sub-invoices Toggle */}
            <label className="flex items-center gap-2 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={showSubInvoices}
                onChange={(e) => setShowSubInvoices(e.target.checked)}
                className="rounded border"
                style={{ borderColor: '#ced4da', color: '#6c757d' }}
              />
              Show Sub-Invoices
            </label>
            <Button onClick={() => setShowPaymentModal(true)}>Record Payment</Button>
          </div>
        }
      >
        {payments.length === 0 ? (
          <EmptyState message="No payments recorded yet" />
        ) : (
          <div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-gray-500 border-b">
                    <th className="pb-3 font-medium">Date</th>
                    <th className="pb-3 font-medium">Service</th>
                    <th className="pb-3 font-medium">Amount</th>
                    <th className="pb-3 font-medium">Invoice ID</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {statusFilteredPayments
                    .slice((paymentsPage - 1) * paymentsPerPage, paymentsPage * paymentsPerPage)
                    .map((payment: Payment) => {
                      const subscription = subscriptions.find(s => s.id === payment.subscription_id)
                      const paymentDate = payment.payment_date ? parseDate(payment.payment_date) : null
                      const isCurrentMonth = payment.payment_month?.slice(0, 7) === currentMonth
                      const subInvoiceCount = !payment.sub_invoice_id ? getSubInvoiceCount(payment.invoice_id) : 0
                      const hasSubInvoices = subInvoiceCount > 0
                      
                      // When hiding sub-invoices (showSubInvoices = false), show combined values for main invoices
                      // When showing sub-invoices, show original values
                      const isHidingSubInvoices = !showSubInvoices && !payment.sub_invoice_id && hasSubInvoices
                      const displayStatus = isHidingSubInvoices ? getCombinedStatus(payment.invoice_id) : payment.payment_status
                      const displayAmount = isHidingSubInvoices ? getTotalPaidForInvoice(payment.invoice_id) : payment.amount_paid
                      
                      return (
                        <tr key={payment.id} className="border-b border-gray-100">
                          <td className="py-3 text-sm text-gray-600">
                            {paymentDate ? paymentDate.format('DD/MM/YYYY') : '-'}
                          </td>
                          <td className="py-3 text-sm text-gray-900">
                            {subscription ? getServiceName(subscription.service_id) : '-'}
                          </td>
                          <td className="py-3 text-sm font-medium text-gray-900">
                            ₹{displayAmount}
                            {hasSubInvoices && !payment.sub_invoice_id && (
                              <span className="text-xs text-gray-500 ml-1">
                                (incl. S{subInvoiceCount})
                              </span>
                            )}
                          </td>
                          <td className="py-3 text-sm text-gray-500 font-mono">
                            {payment.invoice_id || '-'}{payment.sub_invoice_id ? `-${payment.sub_invoice_id}` : ''}
                            {!payment.sub_invoice_id && subInvoiceCount > 0 && (
                              <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                                style={{ backgroundColor: '#e9ecef', color: '#495057' }}
                              >
                                S{subInvoiceCount}
                              </span>
                            )}
                          </td>
                          <td className="py-3">
                            <Badge 
                              variant={
                                displayStatus === 'paid' ? 'success' : 
                                displayStatus === 'partial' ? 'warning' : 'error'
                              }
                            >
                              {displayStatus === 'paid' ? 'Paid' : 
                               displayStatus === 'partial' ? 'Partial' : 'Unpaid'}
                            </Badge>
                          </td>
                          <td className="py-3">
                            <div className="flex gap-2">
                              <button
                                onClick={() => setViewingPayment(payment)}
                                className="p-1.5 text-blue-600 hover:bg-blue-50 rounded"
                                title="View"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                </svg>
                              </button>
                              <button
                                onClick={() => setEditingPayment(payment)}
                                className="p-1.5 rounded"
                                style={{ color: '#6c757d' }}
                                title="Edit"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                </svg>
                              </button>
                              <button
                                onClick={() => setDeletingPayment(payment)}
                                className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                                title="Delete"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
            
            {/* Pagination */}
            {statusFilteredPayments.length > paymentsPerPage && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <p className="text-sm text-gray-500">
                  Showing {(paymentsPage - 1) * paymentsPerPage + 1} to {Math.min(paymentsPage * paymentsPerPage, statusFilteredPayments.length)} of {statusFilteredPayments.length} payments
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setPaymentsPage(p => Math.max(1, p - 1))}
                    disabled={paymentsPage === 1}
                    className="px-3 py-1 text-sm rounded-lg border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setPaymentsPage(p => Math.min(Math.ceil(statusFilteredPayments.length / paymentsPerPage), p + 1))}
                    disabled={paymentsPage >= Math.ceil(statusFilteredPayments.length / paymentsPerPage)}
                    className="px-3 py-1 text-sm rounded-lg border disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Payment Modal - for creating or editing payments */}
      <Modal
        isOpen={showPaymentModal || !!editingPayment || !!parentInvoice}
        onClose={() => { setShowPaymentModal(false); setEditingPayment(null); setParentInvoice(null); }}
        title={editingPayment ? 'Edit Payment' : parentInvoice ? 'Create Sub-Invoice' : 'Record Payment'}
        size="lg"
        variant="sidebar"
      >
        <PaymentForm
          subscriptions={subscriptions}
          contacts={contacts}
          existingPayment={editingPayment || undefined}
          parentInvoice={parentInvoice || undefined}
          onSave={handleSavePayment}
          onCancel={() => { setShowPaymentModal(false); setEditingPayment(null); setParentInvoice(null); }}
        />
      </Modal>

      {/* View Payment Modal */}
      {viewingPayment && (
        <Modal
          isOpen={!!viewingPayment}
          onClose={() => { setViewingPayment(null); setViewingFromSubInvoice(null); }}
          title="Payment Details"
        >
          <div className="space-y-4">
            {/* Back button when viewing from sub-invoice */}
            {viewingFromSubInvoice && (
              <button
                onClick={() => {
                  const mainInvoice = getMainInvoice(viewingPayment.invoice_id, viewingPayment.sub_invoice_id)
                  if (mainInvoice) {
                    setViewingPayment(mainInvoice)
                    setViewingFromSubInvoice(null)
                  }
                }}
                className="flex items-center gap-2 text-sm mb-2"
                style={{ color: '#6c757d' }}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Main Invoice
              </button>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-gray-500">Service</label>
                <p className="font-medium">{getServiceName(viewingPayment.subscription_id ? subscriptions.find(s => s.id === viewingPayment.subscription_id)?.service_id || '-' : '-')}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500 mr-2">Status</label>
                <Badge
                  variant={
                    viewingPayment.payment_status === 'paid' ? 'success' : 
                    viewingPayment.payment_status === 'partial' ? 'warning' : 'error'
                  }
                >
                  {viewingPayment.payment_status === 'paid' ? 'Paid' : 
                   viewingPayment.payment_status === 'partial' ? 'Partial' : 'Unpaid'}
                </Badge>
                {/* Show combined status for main invoices with sub-invoices */}
                {!viewingPayment.sub_invoice_id && getSubInvoiceCount(viewingPayment.invoice_id) > 0 && (
                  <div className="mt-1">
                    <span className="text-xs text-gray-500">Combined: </span>
                    <Badge 
                      variant={getCombinedStatus(viewingPayment.invoice_id) === 'paid' ? 'success' : getCombinedStatus(viewingPayment.invoice_id) === 'partial' ? 'warning' : 'error'}
                    >
                      {getCombinedStatus(viewingPayment.invoice_id) === 'paid' ? 'Paid' : 
                       getCombinedStatus(viewingPayment.invoice_id) === 'partial' ? 'Partial' : 'Unpaid'}
                    </Badge>
                  </div>
                )}
              </div>
              <div>
                <label className="text-sm text-gray-500">Amount Due</label>
                <p className="font-medium">₹{viewingPayment.amount_due}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Amount Paid</label>
                <p className="font-medium">
                  ₹{viewingPayment.amount_paid}
                  {/* Show total for main invoices with sub-invoices */}
                  {!viewingPayment.sub_invoice_id && getSubInvoiceCount(viewingPayment.invoice_id) > 0 && (
                    <span className="text-xs text-gray-500 ml-1">
                      (Remaining: ₹{viewingPayment.remaining_due?.toFixed(2) || '0.00'})
                    </span>
                  )}
                </p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Payment Month</label>
                <p className="font-medium">{viewingPayment.payment_month ? parseDate(viewingPayment.payment_month)?.format('MMMM YYYY') : '-'}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Payment Date</label>
                <p className="font-medium">{viewingPayment.payment_date ? parseDate(viewingPayment.payment_date)?.format('DD/MM/YYYY') : '-'}</p>
              </div>
              <div>
                <label className="text-sm text-gray-500">Payment Method</label>
                <p className="font-medium">{viewingPayment.payment_method || '-'}</p>
              </div>
              <div className="col-span-2">
                <label className="text-sm text-gray-500">Invoice ID</label>
                <p className="font-medium font-mono">{viewingPayment.invoice_id || '-'}{viewingPayment.sub_invoice_id ? `-${viewingPayment.sub_invoice_id}` : ''}</p>
              </div>
              {viewingPayment.notes && (
                <div className="col-span-2">
                  <label className="text-sm text-gray-500">Notes</label>
                  <p className="font-medium">{viewingPayment.notes}</p>
                </div>
              )}
            </div>

            {/* Show sub-invoices if viewing a main invoice */}
            {!viewingPayment.sub_invoice_id && getSubInvoices(viewingPayment.invoice_id).length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="font-medium text-gray-900 mb-3">Sub-Invoices</h4>
                <div className="space-y-2">
                  {getSubInvoices(viewingPayment.invoice_id).map((subInvoice) => (
                    <div 
                      key={subInvoice.id} 
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100"
                      onClick={() => {
                        setViewingFromSubInvoice(viewingPayment)
                        setViewingPayment(subInvoice)
                      }}
                    >
                      <div>
                        <p className="font-medium font-mono text-sm">
                          {viewingPayment.invoice_id}-{subInvoice.sub_invoice_id}
                        </p>
                        <p className="text-xs text-gray-500">
                          {subInvoice.payment_date ? parseDate(subInvoice.payment_date)?.format('DD/MM/YYYY') : 'No date'}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">₹{subInvoice.amount_paid}</p>
                        <Badge 
                          variant={
                            subInvoice.payment_status === 'paid' ? 'success' : 
                            subInvoice.payment_status === 'partial' ? 'warning' : 'error'
                          }
                        >
                          {subInvoice.payment_status === 'paid' ? 'Paid' : 
                           subInvoice.payment_status === 'partial' ? 'Partial' : 'Unpaid'}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex justify-end gap-3 pt-4">
              {viewingPayment.payment_status === 'partial' && (
                <button
                  onClick={() => {
                    setParentInvoice(viewingPayment)
                    setViewingPayment(null)
                  }}
                  className="px-4 py-2 rounded-lg"
                  style={{ backgroundColor: '#6c757d', color: '#ffffff' }}
                >
                  Create Sub-Invoice
                </button>
              )}
              <button
                onClick={() => setViewingPayment(null)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
              >
                Close
              </button>
            </div>
          </div>
        </Modal>
      )}

      {/* Delete Payment Confirmation Modal */}
      {deletingPayment && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-center justify-center p-4">
            <div className="fixed inset-0 bg-black/50" onClick={() => setDeletingPayment(null)} />
            <div className="relative bg-white rounded-2xl shadow-xl p-6 max-w-md w-full">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Payment</h3>
              <p className="text-gray-600 mb-4">
                Are you sure you want to delete this payment record? This action cannot be undone.
              </p>
              <div className="bg-gray-50 p-3 rounded-lg mb-4">
                <p className="text-sm"><strong>Amount:</strong> ₹{deletingPayment.amount_paid}</p>
                <p className="text-sm"><strong>Date:</strong> {deletingPayment.payment_date ? parseDate(deletingPayment.payment_date)?.format('DD/MM/YYYY') : '-'}</p>
              </div>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDeletingPayment(null)}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  onClick={handleDeletePayment}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Subscription Modal - simplified */}
      <Modal
        isOpen={showSubscriptionModal}
        onClose={() => { setShowSubscriptionModal(false); setEditingSubscription(null); }}
        title={editingSubscription ? 'Edit Subscription' : 'Add Subscription'}
      >
        <div className="p-4 text-center text-gray-500">
          Subscription management requires additional setup. Please use the Services and Contacts tabs to manage subscriptions.
        </div>
      </Modal>

      {/* Delete Confirmation Modal */}
      {deleteModalOpen && subscriptionToDelete && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-center justify-center p-4">
            <div className="fixed inset-0 bg-black/50" onClick={() => setDeleteModalOpen(false)} />
            <div className="relative bg-white rounded-2xl shadow-xl p-6 max-w-md w-full">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Subscription</h3>
              <p className="text-gray-600 mb-4">
                Are you sure you want to delete the subscription for <strong>{getServiceName(subscriptionToDelete.service_id)}</strong> for <strong>{getContactName(subscriptionToDelete.contact_id)}</strong>? This will also delete all associated payment records. This action cannot be undone.
              </p>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => setDeleteModalOpen(false)}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmDelete}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
