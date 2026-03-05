'use client'

import { useState, useEffect } from 'react'
import type { Subscription, Payment, Service, Contact } from '@/types/database'
import { CURRENCY_SYMBOL } from '@/constants'

interface PaymentFormProps {
  subscriptions: Subscription[]
  contacts: Contact[]
  existingPayment?: Payment
  parentInvoice?: Payment
  onSave: (data: {
    subscription_id: string
    payment_month: string
    amount_due: number
    amount_paid: number
    payment_date: string | null
    payment_method: string
    notes: string
  }, id?: string) => void
  onCancel: () => void
}

export function PaymentForm({ subscriptions, contacts, existingPayment, parentInvoice, onSave, onCancel }: PaymentFormProps) {
  const [subscriptionId, setSubscriptionId] = useState(existingPayment?.subscription_id || parentInvoice?.subscription_id || '')
  const [paymentMonth, setPaymentMonth] = useState(
    existingPayment?.payment_month 
      ? new Date(existingPayment.payment_month).toISOString().slice(0, 7)
      : parentInvoice?.payment_month
      ? new Date(parentInvoice.payment_month).toISOString().slice(0, 7)
      : new Date().toISOString().slice(0, 7)
  )
  const [amountDue, setAmountDue] = useState(parentInvoice ? parentInvoice.amount_due - parentInvoice.amount_paid : existingPayment?.amount_due || 0)
  const [amountPaid, setAmountPaid] = useState(parentInvoice ? 0 : existingPayment?.amount_paid || 0)
  const [paymentType, setPaymentType] = useState<'full' | 'partial'>(parentInvoice ? 'partial' : (existingPayment ? (existingPayment.amount_paid >= existingPayment.amount_due ? 'full' : 'partial') : 'full'))
  const [paymentDate, setPaymentDate] = useState(
    existingPayment?.payment_date 
      ? new Date(existingPayment.payment_date).toISOString().split('T')[0]
      : new Date().toISOString().split('T')[0]
  )
  const [paymentMethod, setPaymentMethod] = useState(existingPayment?.payment_method || 'cash')
  const [notes, setNotes] = useState(existingPayment?.notes || '')

  // Get selected subscription
  const selectedSubscription = subscriptions.find(s => s.id === subscriptionId)

  // Update amount due when subscription is selected or when editing
  useEffect(() => {
    if (existingPayment) {
      // When editing, use existing payment values
      setAmountDue(existingPayment.amount_due || 0)
      setAmountPaid(existingPayment.amount_paid || 0)
      setPaymentType(existingPayment.amount_paid >= existingPayment.amount_due ? 'full' : 'partial')
    } else if (parentInvoice) {
      // When adding sub-invoice, use remaining amount
      setAmountDue(parentInvoice.amount_due - parentInvoice.amount_paid)
      setAmountPaid(0)
    } else if (selectedSubscription) {
      // When creating new, use subscription amount
      setAmountDue(selectedSubscription.service?.amount || 0)
      // For partial, start at 0 so user can enter partial amount
      setAmountPaid(0)
    }
  }, [selectedSubscription, existingPayment, parentInvoice])

  // Update amount paid based on payment type
  useEffect(() => {
    if (paymentType === 'full') {
      setAmountPaid(amountDue)
    } else if (paymentType === 'partial') {
      // When switching to partial for new payments, reset amount to 0
      // For editing existing, keep the existing value
      if (!existingPayment) {
        setAmountPaid(0)
      }
    }
  }, [paymentType, amountDue, existingPayment])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    onSave({
      subscription_id: subscriptionId,
      payment_month: `${paymentMonth}-01`,
      amount_due: amountDue,
      amount_paid: paymentType === 'full' ? amountDue : amountPaid,
      payment_date: paymentType === 'full' ? paymentDate : (amountPaid > 0 ? paymentDate : null),
      payment_method: paymentMethod,
      notes
    }, existingPayment?.id)
  }

  const paymentMethods = [
    { value: 'cash', label: 'Cash' },
    { value: 'bank_transfer', label: 'Bank Transfer' },
    { value: 'upi', label: 'UPI' },
    { value: 'card', label: 'Card' },
    { value: 'cheque', label: 'Cheque' },
    { value: 'other', label: 'Other' },
  ]

  return (
    <form onSubmit={handleSubmit} className="bg-gray-50 p-4 rounded-lg space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Subscription Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Subscription</label>
          <select
            value={subscriptionId}
            onChange={(e) => setSubscriptionId(e.target.value)}
            required
            disabled={!!existingPayment || !!parentInvoice}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            <option value="">Select subscription...</option>
            {subscriptions.map(sub => {
              const contact = contacts.find(c => c.id === sub.contact_id)
              return (
                <option key={sub.id} value={sub.id}>
                  {contact?.name || 'Unknown'} - {sub.service?.name || 'Unknown Service'} ({CURRENCY_SYMBOL}{sub.service?.amount}/{sub.service?.payment_cycle || 'month'})
                </option>
              )
            })}
          </select>
        </div>

        {/* Payment Month */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Payment Month</label>
          <input
            type="month"
            value={paymentMonth}
            onChange={(e) => setPaymentMonth(e.target.value)}
            required
            disabled={!!parentInvoice}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100"
          />
        </div>

        {/* Amount Due */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Amount Due ({CURRENCY_SYMBOL})</label>
          <input
            type="number"
            value={amountDue}
            onChange={(e) => setAmountDue(Number(e.target.value))}
            required
            min="0"
            step="0.01"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* Payment Type - Hide when adding sub-invoice */}
        {!parentInvoice && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Payment Type</label>
            <div className="flex gap-4 mt-2">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="paymentType"
                  value="full"
                  checked={paymentType === 'full'}
                  onChange={(e) => setPaymentType(e.target.value as 'full')}
                  className="mr-2"
                />
                Full Payment
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="paymentType"
                  value="partial"
                  checked={paymentType === 'partial'}
                  onChange={(e) => setPaymentType(e.target.value as 'partial')}
                  className="mr-2"
                />
                Partial Payment
              </label>
            </div>
          </div>
        )}

        {/* Amount Paid (for partial payments or sub-invoices) */}
        {(paymentType === 'partial' || parentInvoice) && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Amount Paid ({CURRENCY_SYMBOL})</label>
            <input
              type="number"
              value={amountPaid}
              onChange={(e) => setAmountPaid(Number(e.target.value))}
              required
              min="0"
              max={amountDue}
              step="0.01"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              {parentInvoice 
                ? `Remaining to pay: ${CURRENCY_SYMBOL}${(amountDue - amountPaid).toFixed(2)}`
                : `Remaining: ${CURRENCY_SYMBOL}${(amountDue - amountPaid).toFixed(2)} (This will be added to any existing partial payment for this month)`
              }
            </p>
          </div>
        )}

        {/* Payment Date */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Payment Date</label>
          <input
            type="date"
            value={paymentDate}
            onChange={(e) => setPaymentDate(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* Payment Method */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Payment Method</label>
          <select
            value={paymentMethod}
            onChange={(e) => setPaymentMethod(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            {paymentMethods.map(method => (
              <option key={method.value} value={method.value}>{method.label}</option>
            ))}
          </select>
        </div>

        {/* Notes */}
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={2}
            placeholder="Any additional notes..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end gap-3 pt-4">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
        >
          Cancel
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
        >
          {existingPayment ? 'Update Payment' : 'Add Payment'}
        </button>
      </div>
    </form>
  )
}
