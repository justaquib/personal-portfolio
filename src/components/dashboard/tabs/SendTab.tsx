'use client'

import { useState, useEffect } from 'react'
import { Card, EmptyState, LoadingState, Alert } from '../ui'
import { Contact, MessageTemplate, Subscription, Payment } from '@/types/database'
import { CURRENCY_SYMBOL } from '@/constants'
import { useContacts, useTemplates, useSubscriptions, usePayments } from '@/hooks/useDashboardData'
import { createClient } from '@/lib/supabase/client'

interface SubscriptionWithPayments extends Subscription {
  payments?: Payment[]
}

interface SendTabProps {
  userId: string
}

const getCurrentMonth = () => {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
}

export function SendTab({ userId }: SendTabProps) {
  const { contacts, loading: contactsLoading, fetchContacts } = useContacts()
  const { templates, loading: templatesLoading, fetchTemplates } = useTemplates()
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions } = useSubscriptions()
  const { fetchPaymentsForSubscription } = usePayments()
  
  // Fetch data on mount
  useEffect(() => {
    fetchContacts()
    fetchTemplates()
    fetchSubscriptions()
  }, [fetchContacts, fetchTemplates, fetchSubscriptions])
  
  const [phoneNumber, setPhoneNumber] = useState('')
  const [message, setMessage] = useState('')
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null)
  const [selectedTemplate, setSelectedTemplate] = useState<MessageTemplate | null>(null)
  const [selectedSubscription, setSelectedSubscription] = useState<SubscriptionWithPayments | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [sendError, setSendError] = useState<string | null>(null)
  const [sendSuccess, setSendSuccess] = useState(false)

  const getSubscriptions = (contactId: string): Subscription[] => {
    return subscriptions.filter(s => s.contact_id === contactId)
  }

  const calculateOutstandingBalance = async (subscription: SubscriptionWithPayments): Promise<number> => {
    if (!subscription) return 0
    
    const serviceAmount = subscription.service?.amount || 0
    const subscriptionStart = subscription.started_at || new Date().toISOString().split('T')[0]
    
    const startDate = new Date(subscriptionStart)
    const startYear = startDate.getFullYear()
    const startMonth = startDate.getMonth() + 1
    const currentYear = new Date().getFullYear()
    const currentMonthNum = new Date().getMonth() + 1
    
    let outstanding = 0
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    
    for (let year = startYear; year <= currentYear; year++) {
      const monthStart = (year === startYear) ? startMonth : 1
      const monthEnd = (year === currentYear) ? currentMonthNum : 12
      
      for (let month = monthStart; month <= monthEnd; month++) {
        const paymentForMonth = subPayments.find((p: Payment) => {
          const paymentDate = new Date(p.payment_month)
          return paymentDate.getFullYear() === year && paymentDate.getMonth() + 1 === month
        })
        
        const isPaidForMonth = paymentForMonth && paymentForMonth.amount_paid >= paymentForMonth.amount_due
        if (!isPaidForMonth) {
          outstanding += serviceAmount
        }
      }
    }
    
    return outstanding
  }

  const replaceTemplatePlaceholders = (template: string, contact: Contact | null, subscription: SubscriptionWithPayments | null): string => {
    if (!contact) return template
    
    let result = template
    result = result.replace(/\{\{name\}\}/g, contact.name)
    
    const serviceName = subscription?.service?.name || ''
    const serviceAmount = subscription?.service?.amount || 0
    
    result = result.replace(/\{\{amount\}\}/g, String(serviceAmount))
    result = result.replace(/\{\{service\}\}/g, serviceName)
    
    let outstandingBalance = 0
    let dueMonths: string[] = []
    
    if (subscription && subscription.payments) {
      const mainInvoices = subscription.payments.filter((p: Payment) => !p.sub_invoice_id)
      
      mainInvoices.forEach((payment: Payment) => {
        const subInvoices = subscription.payments!.filter((p: Payment) => 
          p.invoice_id === payment.invoice_id && p.sub_invoice_id
        )
        
        const mainPaid = payment.amount_paid || 0
        const subInvoicesPaid = subInvoices.reduce((sum: number, sub: Payment) => sum + (sub.amount_paid || 0), 0)
        const totalPaid = mainPaid + subInvoicesPaid
        const mainDue = payment.amount_due || 0
        const remainingDue = totalPaid >= mainDue ? 0 : (mainDue - totalPaid)
        
        if (remainingDue > 0) {
          outstandingBalance += remainingDue
          const paymentMonth = payment.payment_month ? new Date(payment.payment_month).toISOString().slice(0, 7) : ''
          if (paymentMonth && !dueMonths.includes(paymentMonth)) {
            dueMonths.push(paymentMonth)
          }
        }
      })
    }
    
    const displayMonth = dueMonths.length > 0 ? dueMonths.sort().join(', ') : getCurrentMonth()
    result = result.replace(/\{\{month\}\}/g, displayMonth)
    result = result.replace(/\{\{outstanding\}\}/g, String(outstandingBalance))
    
    return result
  }

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault()
    setSendError(null)
    setSendSuccess(false)

    const cleanedPhone = phoneNumber.replace(/[^\d+]/g, '')
    if (cleanedPhone.length < 10 || !message.trim()) {
      setSendError('Please enter a valid phone number and message')
      return
    }

    const outstandingBalance = selectedSubscription ? await calculateOutstandingBalance(selectedSubscription) : 0
    const currentMonth = getCurrentMonth()
    const currentPayments = await fetchPaymentsForSubscription(selectedSubscription?.id || '')
    const isCurrentMonthPaid = currentPayments.some((p: Payment) => {
      const paymentDate = new Date(p.payment_month)
      const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
      return paymentMonthStr === currentMonth && p.amount_paid >= p.amount_due
    })
    
    if (isCurrentMonthPaid && outstandingBalance === 0) {
      setSendError('Payment already received for this month and no outstanding balance.')
      return
    }

    let finalMessage = message.trim()
    if (outstandingBalance > 0) {
      finalMessage = `Outstanding Balance: ${CURRENCY_SYMBOL}${outstandingBalance}\n\n${finalMessage}`
    }

    const encodedMessage = encodeURIComponent(finalMessage)
    const whatsappUrl = `https://wa.me/${cleanedPhone}?text=${encodedMessage}`
    window.open(whatsappUrl, '_blank')
    
    try {
      const supabase = createClient()
      await supabase.from('notifications').insert({
        phone_number: cleanedPhone,
        message: finalMessage,
        timestamp: new Date().toISOString(),
        status: 'sent',
      })
    } catch (err) {
      console.error('Failed to log notification:', err)
    }
    
    setSendSuccess(true)
    setPhoneNumber('')
    setMessage('')
    setSelectedContact(null)
    setSelectedTemplate(null)
    setSelectedSubscription(null)
    setTimeout(() => setSendSuccess(false), 5000)
  }

  const handleSelectContact = (contact: Contact) => {
    setSelectedContact(contact)
    setPhoneNumber(contact.phone_number)
    setSelectedSubscription(null)
    
    if (selectedTemplate) {
      const filledMessage = replaceTemplatePlaceholders(selectedTemplate.content, contact, null)
      setMessage(filledMessage)
    }
  }

  const handleSelectSubscription = async (subscription: Subscription) => {
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    const subscriptionWithPayments: SubscriptionWithPayments = {
      ...subscription,
      payments: subPayments
    }
    setSelectedSubscription(subscriptionWithPayments)
    if (selectedTemplate && selectedContact) {
      const filledMessage = replaceTemplatePlaceholders(selectedTemplate.content, selectedContact, subscriptionWithPayments)
      setMessage(filledMessage)
    }
  }

  const handleSelectTemplate = (template: MessageTemplate) => {
    setSelectedTemplate(template)
    const filledMessage = replaceTemplatePlaceholders(template.content, selectedContact, selectedSubscription)
    setMessage(filledMessage)
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Quick Select - Contacts */}
      <Card title="Select Contact">
        {contactsLoading ? (
          <LoadingState />
        ) : contacts.length === 0 ? (
          <EmptyState message="No contacts yet. Add contacts in the Contacts tab." />
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {contacts.map((contact) => (
              <button
                key={contact.id}
                onClick={() => handleSelectContact(contact)}
                disabled={!contact.is_active}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedContact?.id === contact.id
                    ? 'bg-purple-100 border-purple-500'
                    : !contact.is_active
                    ? 'bg-gray-50 cursor-not-allowed opacity-50'
                    : 'bg-gray-50 hover:bg-gray-100'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-medium text-gray-900">{contact.name}</p>
                    <p className="text-sm text-gray-500">{contact.company || contact.phone_number}</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Quick Select - Subscriptions */}
      <Card title="Select Service">
        {!selectedContact ? (
          <EmptyState message="Select a contact first to see their services." />
        ) : subscriptionsLoading ? (
          <LoadingState />
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {getSubscriptions(selectedContact.id).length === 0 ? (
              <EmptyState message="No services linked to this contact. Go to Services tab to link services." />
            ) : (
              getSubscriptions(selectedContact.id).map((sub) => (
                <button
                  key={sub.id}
                  onClick={() => handleSelectSubscription(sub)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedSubscription?.id === sub.id
                      ? 'bg-purple-100 border-purple-500'
                      : 'bg-gray-50 hover:bg-gray-100'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium text-gray-900">{sub.service?.name || 'Unknown Service'}</p>
                      <p className="text-sm text-gray-500">{CURRENCY_SYMBOL}{sub.service?.amount || 0} <span className="text-xs">/ {sub.service?.payment_cycle || 'month'}</span></p>
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        )}
      </Card>

      {/* Quick Select - Templates */}
      <Card title="Select Template">
        {templatesLoading ? (
          <LoadingState />
        ) : templates.length === 0 ? (
          <EmptyState message="No templates yet. Create templates in the Templates tab." />
        ) : (
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {templates.map((template) => (
              <button
                key={template.id}
                onClick={() => handleSelectTemplate(template)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedTemplate?.id === template.id
                    ? 'bg-green-100 border-green-500'
                    : 'bg-gray-50 hover:bg-gray-100'
                }`}
              >
                <p className="font-medium text-gray-900">{template.name}</p>
                <p className="text-sm text-gray-500 truncate">{template.content}</p>
              </button>
            ))}
          </div>
        )}
      </Card>

      {/* Send Form */}
      <div className="lg:col-span-3">
        <Card title="Send Notification">
          <form onSubmit={handleSend} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Phone Number</label>
              <input
                type="tel"
                value={phoneNumber}
                onChange={(e) => setPhoneNumber(e.target.value)}
                placeholder="+1234567890"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={isSending}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Message</label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={4}
                placeholder="Enter your payment notification message..."
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
                disabled={isSending}
              />
              <p className="text-xs text-gray-500 mt-1">
                Available placeholders: {'{{name}}'}, {'{{amount}}'}, {'{{service}}'}, {'{{month}}'}, {'{{outstanding}}'}
              </p>
            </div>
            {sendError && <Alert type="error" message={sendError} />}
            {sendSuccess && <Alert type="success" message="✓ Notification sent successfully!" />}
            <button
              type="submit"
              disabled={!phoneNumber.trim() || !message.trim()}
              className="w-full bg-green-600 text-white font-semibold py-3 rounded-xl hover:bg-green-700 disabled:opacity-50"
            >
              Send Notification
            </button>
          </form>
        </Card>
      </div>
    </div>
  )
}
