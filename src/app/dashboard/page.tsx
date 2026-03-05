'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/context/AuthContext'
import { useContacts, useTemplates, useNotifications, useServices, useSubscriptions, usePayments } from '@/hooks/useDashboardData'
import { ContactForm } from '@/components/dashboard/ContactForm'
import { TemplateForm } from '@/components/dashboard/TemplateForm'
import { ServiceForm } from '@/components/dashboard/ServiceForm'
import { PaymentForm } from '@/components/dashboard/PaymentForm'
import { createClient } from '@/lib/supabase/client'
import type { TabType, ContactFormData, TemplateFormData, Contact, MessageTemplate, Subscription, Payment, ServiceFormData, Service, PaymentFormData } from '@/types/database'
import { CURRENCY_SYMBOL } from '@/constants'
import './dashboard.css'

// Icons as components for reusability
const SendIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
)

const UserIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
  </svg>
)

const TemplateIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
)

const HistoryIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const ServiceIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
  </svg>
)

const DollarIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const CreditCardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
  </svg>
)

const LogoutIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
  </svg>
)

const RefreshIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
)

const EditIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
  </svg>
)

const DeleteIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
  </svg>
)

// Helper function to get current month
const getCurrentMonth = () => {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
}

const getYearOptions = () => {
  const years = []
  const currentYear = new Date().getFullYear()
  for (let year = currentYear; year >= currentYear - 5; year--) {
    years.push(year)
  }
  return years
}

const getMonthOptions = () => {
  return [
    { value: '01', label: 'January' },
    { value: '02', label: 'February' },
    { value: '03', label: 'March' },
    { value: '04', label: 'April' },
    { value: '05', label: 'May' },
    { value: '06', label: 'June' },
    { value: '07', label: 'July' },
    { value: '08', label: 'August' },
    { value: '09', label: 'September' },
    { value: '10', label: 'October' },
    { value: '11', label: 'November' },
    { value: '12', label: 'December' },
  ]
}

// Combined subscription with payment info
interface SubscriptionWithPayments extends Subscription {
  payments?: Payment[]
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, loading: authLoading, signOut } = useAuth()
  const { contacts, loading: contactsLoading, fetchContacts, saveContact, deleteContact } = useContacts()
  const { templates, loading: templatesLoading, fetchTemplates, saveTemplate, deleteTemplate } = useTemplates()
  const { notifications, loading: notificationsLoading, fetchNotifications, sendNotification } = useNotifications()
  const { services, loading: servicesLoading, fetchServices, saveService, deleteService } = useServices()
  const { subscriptions, loading: subscriptionsLoading, fetchSubscriptions, saveSubscription, deleteSubscription, getSubscriptionsForContact } = useSubscriptions()
  const { payments, loading: paymentsLoading, fetchPayments, savePayment, deletePayment, fetchPaymentsForSubscription } = usePayments()
  
  const [activeTab, setActiveTab] = useState<TabType>('send')
  const [phoneNumber, setPhoneNumber] = useState('')
  const [message, setMessage] = useState('')
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null)
  const [selectedTemplate, setSelectedTemplate] = useState<MessageTemplate | null>(null)
  const [editingContact, setEditingContact] = useState<Contact | null>(null)
  const [editingTemplate, setEditingTemplate] = useState<MessageTemplate | null>(null)
  const [showContactForm, setShowContactForm] = useState(false)
  const [showTemplateForm, setShowTemplateForm] = useState(false)
  const [showServiceForm, setShowServiceForm] = useState(false)
  const [showPaymentForm, setShowPaymentForm] = useState(false)
  const [editingPayment, setEditingPayment] = useState<Payment | null>(null)
  const [editingService, setEditingService] = useState<Service | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [sendError, setSendError] = useState<string | null>(null)
  const [sendSuccess, setSendSuccess] = useState(false)

  // Service linking state
  const [selectedContactForService, setSelectedContactForService] = useState<string>('')
  const [selectedServiceToLink, setSelectedServiceToLink] = useState<string>('')
  const [linkServiceLoading, setLinkServiceLoading] = useState(false)

  // Send message state - selected subscription for the contact
  const [selectedSubscription, setSelectedSubscription] = useState<SubscriptionWithPayments | null>(null)

  // History tab filters
  const [filterPaymentStatus, setFilterPaymentStatus] = useState<'all' | 'paid' | 'unpaid'>('all')
  const [filterMonth, setFilterMonth] = useState<string>(getCurrentMonth())

  // Earnings tab filters
  const [earningsFilter, setEarningsFilter] = useState<'all' | 'service' | 'contact'>('all')
  const [servicesTab, setServicesTab] = useState<'services' | 'linked'>('services')
  const [paymentMonth, setPaymentMonth] = useState<string>(getCurrentMonth())
  const [subscriptionStartMonth, setSubscriptionStartMonth] = useState<string>(getCurrentMonth())
  const [earningsMonth, setEarningsMonth] = useState<string>(getCurrentMonth())

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/login')
    }
  }, [user, authLoading, router])

  // Fetch data on mount
  useEffect(() => {
    if (user) {
      fetchContacts()
      fetchTemplates()
      fetchNotifications()
      fetchServices()
      fetchSubscriptions()
      fetchPayments()
    }
  }, [user])

  const handleSignOut = async () => {
    await signOut()
    router.push('/login')
  }

  // Calculate outstanding balance for a subscription
  const calculateOutstandingBalance = async (subscription: SubscriptionWithPayments): Promise<number> => {
    if (!subscription) return 0
    
    const serviceAmount = subscription.service?.amount || 0
    const subscriptionStart = subscription.started_at || new Date().toISOString().split('T')[0]
    
    // Get subscription start year and month
    const startDate = new Date(subscriptionStart)
    const startYear = startDate.getFullYear()
    const startMonth = startDate.getMonth() + 1
    
    const currentYear = new Date().getFullYear()
    const currentMonthNum = new Date().getMonth() + 1
    
    let outstanding = 0
    
    // Fetch all payments for this subscription
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    
    // Check from subscription start month to current month
    for (let year = startYear; year <= currentYear; year++) {
      const monthStart = (year === startYear) ? startMonth : 1
      const monthEnd = (year === currentYear) ? currentMonthNum : 12
      
      for (let month = monthStart; month <= monthEnd; month++) {
        const checkMonth = `${year}-${String(month).padStart(2, '0')}-01`
        
        // Check if payment was received for this month
        const paymentForMonth = subPayments.find(p => {
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

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault()
    setSendError(null)
    setSendSuccess(false)

    const cleanedPhone = phoneNumber.replace(/[^\d+]/g, '')
    if (cleanedPhone.length < 10 || !message.trim()) {
      setSendError('Please enter a valid phone number and message')
      return
    }

    // Calculate outstanding balance
    const outstandingBalance = selectedSubscription ? await calculateOutstandingBalance(selectedSubscription) : 0
    
    // Check if payment is received for current month
    const currentMonth = getCurrentMonth()
    const currentPayments = await fetchPaymentsForSubscription(selectedSubscription?.id || '')
    const isCurrentMonthPaid = currentPayments.some(p => {
      const paymentDate = new Date(p.payment_month)
      const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
      return paymentMonthStr === currentMonth && p.amount_paid >= p.amount_due
    })
    
    // Allow sending if: current month not paid OR there's outstanding balance
    if (isCurrentMonthPaid && outstandingBalance === 0) {
      setSendError('Payment already received for this month and no outstanding balance.')
      return
    }

    // Add outstanding balance to message if applicable
    let finalMessage = message.trim()
    if (outstandingBalance > 0) {
      finalMessage = `Outstanding Balance: ${CURRENCY_SYMBOL}${outstandingBalance}\n\n${finalMessage}`
    }

    // Open WhatsApp with pre-filled message
    const encodedMessage = encodeURIComponent(finalMessage)
    const whatsappUrl = `https://wa.me/${cleanedPhone}?text=${encodedMessage}`
    
    // Open WhatsApp in new tab
    window.open(whatsappUrl, '_blank')
    
    // Log the notification to database for history
    try {
      const supabase = createClient()
      await supabase.from('notifications').insert({
        phone_number: cleanedPhone,
        message: finalMessage,
        timestamp: new Date().toISOString(),
        status: 'sent',
      })
      fetchNotifications()
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

  // Get subscriptions for a specific contact
  const getSubscriptions = (contactId: string): Subscription[] => {
    return subscriptions.filter(s => s.contact_id === contactId)
  }

  // Check if payment is received for a subscription for current month
  const isPaymentReceivedForSubscription = async (subscription: Subscription) => {
    const currentMonth = getCurrentMonth()
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    return subPayments.some(p => {
      const paymentDate = new Date(p.payment_month)
      const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
      return paymentMonthStr === currentMonth && p.amount_paid >= p.amount_due
    })
  }

  // Toggle payment status for a subscription
  const togglePaymentStatusForSubscription = async (subscription: Subscription) => {
    const currentMonth = getCurrentMonth()
    const currentDate = new Date()
    const paymentMonthDate = new Date(currentDate.getFullYear(), currentDate.getMonth(), 1)
    
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    const existingPayment = subPayments.find(p => {
      const paymentDate = new Date(p.payment_month)
      const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
      return paymentMonthStr === currentMonth
    })
    
    const serviceAmount = subscription.service?.amount || 0
    
    if (existingPayment) {
      // Toggle payment
      await savePayment({
        subscription_id: subscription.id,
        payment_month: existingPayment.payment_month,
        amount_due: serviceAmount,
        amount_paid: existingPayment.amount_paid > 0 ? 0 : serviceAmount,
        payment_date: existingPayment.amount_paid > 0 ? null : currentDate.toISOString().split('T')[0],
        payment_method: existingPayment.amount_paid > 0 ? '' : 'manual',
        notes: ''
      }, existingPayment.id)
    } else if (user) {
      // Create new payment
      await savePayment({
        subscription_id: subscription.id,
        payment_month: paymentMonthDate.toISOString().split('T')[0],
        amount_due: serviceAmount,
        amount_paid: serviceAmount,
        payment_date: currentDate.toISOString().split('T')[0],
        payment_method: 'manual',
        notes: '',
        user_id: user.id
      })
    }
    
    fetchPayments()
  }

  const handleSelectContact = async (contact: Contact) => {
    setSelectedContact(contact)
    setPhoneNumber(contact.phone_number)
    setSelectedSubscription(null) // Reset selected subscription
    
    // If a template is already selected, re-apply placeholders
    if (selectedTemplate) {
      const filledMessage = replaceTemplatePlaceholders(selectedTemplate.content, contact, null)
      setMessage(filledMessage)
    }
  }

  // Handle selecting a specific subscription for a contact
  const handleSelectSubscription = async (subscription: Subscription) => {
    const subPayments = await fetchPaymentsForSubscription(subscription.id)
    const subscriptionWithPayments: SubscriptionWithPayments = {
      ...subscription,
      payments: subPayments
    }
    setSelectedSubscription(subscriptionWithPayments)
    // If a template is already selected, re-apply placeholders with the service data
    if (selectedTemplate && selectedContact) {
      const filledMessage = replaceTemplatePlaceholders(selectedTemplate.content, selectedContact, subscriptionWithPayments)
      setMessage(filledMessage)
    }
  }

  // Replace template placeholders with contact and subscription data
  const replaceTemplatePlaceholders = (template: string, contact: Contact | null, subscription: SubscriptionWithPayments | null): string => {
    if (!contact) return template
    
    let result = template
    result = result.replace(/\{\{name\}\}/g, contact.name)
    
    // Service-related placeholders
    const serviceName = subscription?.service?.name || ''
    const serviceAmount = subscription?.service?.amount || 0
    
    result = result.replace(/\{\{amount\}\}/g, String(serviceAmount))
    result = result.replace(/\{\{service\}\}/g, serviceName)
    result = result.replace(/\{\{month\}\}/g, getCurrentMonth())
    
    // Calculate outstanding balance for this service
    let outstandingBalance = 0
    if (subscription) {
      // Use sync calculation with existing payments if available
      const serviceAmount2 = subscription.service?.amount || 0
      const subscriptionStart = subscription.started_at || new Date().toISOString().split('T')[0]
      const startDate = new Date(subscriptionStart)
      const startYear = startDate.getFullYear()
      const startMonth = startDate.getMonth() + 1
      const currentYear = new Date().getFullYear()
      const currentMonthNum = new Date().getMonth() + 1
      
      let outstanding = 0
      for (let year = startYear; year <= currentYear; year++) {
        const monthStart = (year === startYear) ? startMonth : 1
        const monthEnd = (year === currentYear) ? currentMonthNum : 12
        
        for (let month = monthStart; month <= monthEnd; month++) {
          const checkMonth = `${year}-${String(month).padStart(2, '0')}`
          
          const paymentForMonth = subscription.payments?.find(p => {
            const paymentDate = new Date(p.payment_month)
            return paymentDate.getFullYear() === year && paymentDate.getMonth() + 1 === month
          })
          
          const isPaidForMonth = paymentForMonth && paymentForMonth.amount_paid >= paymentForMonth.amount_due
          if (!isPaidForMonth) {
            outstanding += serviceAmount2
          }
        }
      }
      outstandingBalance = outstanding
    }
    result = result.replace(/\{\{outstanding\}\}/g, String(outstandingBalance))
    
    return result
  }

  const handleSelectTemplate = (template: MessageTemplate) => {
    setSelectedTemplate(template)
    // Replace placeholders with selected contact data if available
    const filledMessage = replaceTemplatePlaceholders(template.content, selectedContact, selectedSubscription)
    setMessage(filledMessage)
  }

  const handleContactSave = async (data: ContactFormData) => {
    await saveContact(data, user?.id || '', editingContact?.id)
    setEditingContact(null)
    setShowContactForm(false)
  }

  const handleTemplateSave = async (data: TemplateFormData) => {
    await saveTemplate(data, user?.id || '', editingTemplate?.id)
    setEditingTemplate(null)
    setShowTemplateForm(false)
  }

  const handleServiceSave = async (data: ServiceFormData) => {
    await saveService(data, user?.id || '', editingService?.id)
    setShowServiceForm(false)
    setEditingService(null)
  }

  // Link a service to a contact
  const handleLinkServiceToContact = async () => {
    if (!selectedContactForService || !selectedServiceToLink) return
    
    setLinkServiceLoading(true)
    try {
      await saveSubscription({
        contact_id: selectedContactForService,
        service_id: selectedServiceToLink,
        started_at: `${subscriptionStartMonth}-01`
      })
      setSelectedContactForService('')
      setSelectedServiceToLink('')
      setSubscriptionStartMonth(getCurrentMonth())
      fetchSubscriptions()
    } catch (err: any) {
      console.error('Failed to link service:', err)
    } finally {
      setLinkServiceLoading(false)
    }
  }

  // Remove subscription from contact
  const handleUnlinkService = async (subscriptionId: string) => {
    try {
      await deleteSubscription(subscriptionId)
      fetchSubscriptions()
    } catch (err: any) {
      console.error('Failed to unlink service:', err)
    }
  }

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  if (!user) return null

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'send', label: 'Send', icon: <SendIcon /> },
    { id: 'contacts', label: 'Contacts', icon: <UserIcon /> },
    { id: 'services', label: 'Services', icon: <ServiceIcon /> },
    { id: 'payments', label: 'Payments', icon: <CreditCardIcon /> },
    { id: 'templates', label: 'Templates', icon: <TemplateIcon /> },
    { id: 'history', label: 'History', icon: <HistoryIcon /> },
    { id: 'earnings', label: 'Earnings', icon: <DollarIcon /> },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Payment Dashboard</h1>
              <p className="text-sm text-gray-500">
                Welcome, {user.user_metadata?.full_name || user.email}
              </p>
            </div>
            <button
              onClick={handleSignOut}
              className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
            >
              <LogoutIcon />
              Sign Out
            </button>
          </div>
        </div>
      </header>

      {/* Tabs */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8" aria-label="Tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors flex items-center gap-2 ${
                  activeTab === tab.id
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Send Message Tab */}
        {activeTab === 'send' && (
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

            {/* Quick Select - Subscriptions (for selected contact) */}
            <Card title="Select Service">
              {!selectedContact ? (
                <EmptyState message="Select a contact first to see their services." />
              ) : servicesLoading || subscriptionsLoading ? (
                <LoadingState />
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {getSubscriptions(selectedContact.id).length === 0 ? (
                    <EmptyState message="No services linked to this contact. Go to Services tab to link services." />
                  ) : (
                    getSubscriptions(selectedContact.id).map((sub) => (
                      <SubscriptionItem 
                        key={sub.id} 
                        subscription={sub} 
                        isSelected={selectedSubscription?.id === sub.id}
                        onClick={() => handleSelectSubscription(sub)}
                      />
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

            {/* Send Form - Full width below */}
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
        )}

        {/* Contacts Tab */}
        {activeTab === 'contacts' && (
          <Card title="Contacts">
            <div className="flex justify-end mb-4">
              <button
                onClick={() => { setEditingContact(null); setShowContactForm(true); }}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Add Contact
              </button>
            </div>

            {(showContactForm || editingContact) && (
              <ContactForm
                contact={editingContact}
                onSave={handleContactSave}
                onCancel={() => { setEditingContact(null); setShowContactForm(false); }}
              />
            )}

            {contactsLoading ? (
              <LoadingState />
            ) : contacts.length === 0 ? (
              <EmptyState message="No contacts yet. Click 'Add Contact' to create one." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Name</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Phone</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Services</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {contacts.map((contact) => (
                      <tr key={contact.id} className="border-b hover:bg-gray-50">
                        <td className="py-3 px-4">{contact.name}</td>
                        <td className="py-3 px-4">{contact.phone_number}</td>
                        <td className="py-3 px-4">
                          {getSubscriptions(contact.id).length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {getSubscriptions(contact.id).map((sub) => (
                                <span key={sub.id} className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded-full">
                                  {sub.service?.name} (${sub.service?.amount})
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-gray-400 text-sm">No services</span>
                          )}
                        </td>
                        <td className="py-3 px-4">
                          <span className={`px-2 py-1 rounded-full text-xs ${contact.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'}`}>
                            {contact.is_active ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <button onClick={() => { setEditingContact(contact); setShowContactForm(true); }} className="text-purple-600 hover:underline mr-3">Edit</button>
                          <button onClick={() => deleteContact(contact.id)} className="text-red-600 hover:underline">Delete</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}

        {/* Services Tab */}
        {activeTab === 'services' && (
          <div className="space-y-6">
            {/* Sub-tabs for Services */}
            <div className="flex bg-gray-100 rounded-lg p-1 w-fit">
              <button
                onClick={() => setServicesTab('services')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  servicesTab === 'services'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Services ({services.length})
              </button>
              <button
                onClick={() => setServicesTab('linked')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  servicesTab === 'linked'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Linked Services ({subscriptions.length})
              </button>
            </div>

            {/* Services Section */}
            {servicesTab === 'services' && (
              <Card title="Services">
                <div className="flex justify-end mb-4">
                  <button
                    onClick={() => { setEditingService(null); setShowServiceForm(true); }}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                  >
                    {editingService ? 'Edit Service' : 'Add Service'}
                  </button>
                </div>

                {showServiceForm && (
                  <ServiceForm
                    service={editingService}
                    onSave={handleServiceSave}
                    onCancel={() => { setShowServiceForm(false); setEditingService(null); }}
                  />
                )}

                {servicesLoading ? (
                  <LoadingState />
                ) : services.length === 0 ? (
                  <EmptyState message="No services yet. Click 'Add Service' to create one." />
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {services.map((service) => (
                      <div key={service.id} className="p-4 bg-gray-50 rounded-xl">
                        <div className="flex items-start justify-between">
                          <div>
                            <h4 className="font-medium text-gray-900">{service.name}</h4>
                            <p className="text-sm text-gray-600 mt-1">${service.amount}</p>
                            {service.actual_cost > 0 && (
                              <p className="text-xs text-red-500 mt-1">Cost: ${service.actual_cost}</p>
                            )}
                            <p className="text-xs text-gray-500 mt-1">{service.description}</p>
                          </div>
                          <div>
                            <button onClick={() => { setEditingService(service); setShowServiceForm(true); }} className="text-purple-600 hover:underline mr-3">Edit</button>
                            <button onClick={() => deleteService(service.id)} className="text-red-600 hover:underline">Delete</button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )}

            {/* Linked Services Section */}
            {servicesTab === 'linked' && (
              <Card title="Link Service to Contact">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Select Contact</label>
                    <select
                      value={selectedContactForService}
                      onChange={(e) => setSelectedContactForService(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    >
                      <option value="">Choose a contact...</option>
                      {contacts.map((contact) => (
                        <option key={contact.id} value={contact.id}>{contact.name}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Select Service</label>
                    <select
                      value={selectedServiceToLink}
                      onChange={(e) => setSelectedServiceToLink(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    >
                      <option value="">Choose a service...</option>
                      {services.map((service) => (
                        <option key={service.id} value={service.id}>{service.name} - ${service.amount}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Service Started</label>
                    <select
                      value={subscriptionStartMonth}
                      onChange={(e) => setSubscriptionStartMonth(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    >
                      {getYearOptions().map(year => 
                        getMonthOptions().map(month => (
                          <option key={`${year}-${month.value}`} value={`${year}-${month.value}`}>
                            {month.label} {year}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                  <button
                    onClick={handleLinkServiceToContact}
                    disabled={!selectedContactForService || !selectedServiceToLink || linkServiceLoading}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50"
                  >
                    {linkServiceLoading ? 'Linking...' : 'Link Service'}
                  </button>
                </div>

                {/* Show linked subscriptions with unlink option */}
                {subscriptions.length > 0 ? (
                  <div className="mt-6">
                    <h4 className="font-medium text-gray-900 mb-3">Linked Services</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-3 px-4 font-medium text-gray-600">Contact</th>
                            <th className="text-left py-3 px-4 font-medium text-gray-600">Service</th>
                            <th className="text-left py-3 px-4 font-medium text-gray-600">Amount</th>
                            <th className="text-left py-3 px-4 font-medium text-gray-600">Since</th>
                            <th className="text-left py-3 px-4 font-medium text-gray-600">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {subscriptions.map((sub) => {
                            const contact = contacts.find(c => c.id === sub.contact_id)
                            const startedAt = sub.started_at ? new Date(sub.started_at).toISOString().slice(0, 7) : getCurrentMonth()
                            return (
                              <tr key={sub.id} className="border-b hover:bg-gray-50">
                                <td className="py-3 px-4">{contact?.name || 'Unknown'}</td>
                                <td className="py-3 px-4">{sub.service?.name || 'Unknown'}</td>
                                <td className="py-3 px-4">${sub.service?.amount || 0}</td>
                                <td className="py-3 px-4 text-sm text-gray-600">
                                  {startedAt}
                                </td>
                                <td className="py-3 px-4">
                                  <button 
                                    onClick={() => handleUnlinkService(sub.id)} 
                                    className="text-red-600 hover:underline"
                                  >
                                    Unlink
                                  </button>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <EmptyState message="No services linked to contacts yet." />
                )}
              </Card>
            )}
          </div>
        )}

        {/* Payments Tab */}
        {activeTab === 'payments' && (
          <Card title="Payments">
            {/* Add Payment Record Form */}
            {(showPaymentForm || editingPayment) && (
              <div className="mb-6">
                <h4 className="font-medium text-gray-900 mb-3">
                  {editingPayment ? 'Edit Payment' : 'Add Payment Record'}
                </h4>
                <PaymentForm
                  subscriptions={subscriptions}
                  contacts={contacts}
                  existingPayment={editingPayment || undefined}
                  onSave={async (data) => {
                    await savePayment({ ...data, user_id: user.id }, editingPayment?.id)
                    setShowPaymentForm(false)
                    setEditingPayment(null)
                    fetchPayments()
                  }}
                  onCancel={() => {
                    setShowPaymentForm(false)
                    setEditingPayment(null)
                  }}
                />
              </div>
            )}

            {/* Month Selector and Add Button */}
            <div className="flex flex-wrap items-center gap-4 mb-6">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">View Month:</label>
                <input
                  type="month"
                  value={filterMonth}
                  onChange={(e) => setFilterMonth(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">Status:</label>
                <select
                  value={filterPaymentStatus}
                  onChange={(e) => setFilterPaymentStatus(e.target.value as 'all' | 'paid' | 'unpaid')}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value="all">All</option>
                  <option value="paid">Paid</option>
                  <option value="unpaid">Unpaid</option>
                </select>
              </div>
              <button
                onClick={() => { fetchSubscriptions(); fetchPayments(); }}
                className="p-2 text-gray-400 hover:text-gray-600"
              >
                <RefreshIcon />
              </button>
              <button
                onClick={() => { setEditingPayment(null); setShowPaymentForm(true); }}
                className="ml-auto px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                + Add Record
              </button>
            </div>

            {/* Payment Records from Database */}
            {payments.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Contact</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Service</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Month</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Amount Due</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Amount Paid</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Status</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Payment Date</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Method</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-600">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payments
                      .filter(payment => {
                        // Filter by month if selected
                        if (filterPaymentStatus !== 'all') {
                          const paymentDate = new Date(payment.payment_month)
                          const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
                          if (paymentMonthStr !== filterMonth) return false
                          
                          const isPaid = payment.amount_paid >= payment.amount_due
                          if (filterPaymentStatus === 'paid' && !isPaid) return false
                          if (filterPaymentStatus === 'unpaid' && isPaid) return false
                        }
                        return true
                      })
                      .map((payment) => {
                        const subscription = subscriptions.find(s => s.id === payment.subscription_id)
                        const contact = subscription ? contacts.find(c => c.id === subscription.contact_id) : null
                        const isPaid = payment.amount_paid >= payment.amount_due
                        
                        return (
                          <tr key={payment.id} className="border-b hover:bg-gray-50">
                            <td className="py-3 px-4">{contact?.name || 'Unknown'}</td>
                            <td className="py-3 px-4">{subscription?.service?.name || 'Unknown'}</td>
                            <td className="py-3 px-4">{payment.payment_month ? new Date(payment.payment_month).toISOString().slice(0, 7) : '-'}</td>
                            <td className="py-3 px-4 font-medium">{CURRENCY_SYMBOL}{payment.amount_due || 0}</td>
                            <td className="py-3 px-4">{CURRENCY_SYMBOL}{payment.amount_paid || 0}</td>
                            <td className="py-3 px-4">
                              <span className={`px-2 py-1 rounded-full text-xs ${isPaid ? 'bg-green-100 text-green-700' : payment.amount_paid > 0 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                                {isPaid ? 'Paid' : payment.amount_paid > 0 ? 'Partial' : 'Unpaid'}
                              </span>
                            </td>
                            <td className="py-3 px-4">{payment.payment_date ? new Date(payment.payment_date).toLocaleDateString() : '-'}</td>
                            <td className="py-3 px-4">{payment.payment_method || '-'}</td>
                            <td className="py-3 px-4">
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => { setEditingPayment(payment); setShowPaymentForm(true); }}
                                  className="p-1.5 text-purple-600 hover:bg-purple-50 rounded"
                                  title="Edit"
                                >
                                  <EditIcon />
                                </button>
                                <button
                                  onClick={async () => {
                                    if (confirm('Delete this payment record?')) {
                                      await deletePayment(payment.id)
                                      fetchPayments()
                                    }
                                  }}
                                  className="p-1.5 text-red-600 hover:bg-red-50 rounded"
                                  title="Delete"
                                >
                                  <DeleteIcon />
                                </button>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState message="No payment records found. Click 'Add Payment Record' to create one." />
            )}
          </Card>
        )}

        {/* Templates Tab */}
        {activeTab === 'templates' && (
          <Card title="Message Templates">
            <div className="flex justify-end mb-4">
              <button
                onClick={() => { setEditingTemplate(null); setShowTemplateForm(true); }}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Add Template
              </button>
            </div>

            {(showTemplateForm || editingTemplate) && (
              <TemplateForm
                template={editingTemplate}
                onSave={handleTemplateSave}
                onCancel={() => { setEditingTemplate(null); setShowTemplateForm(false); }}
              />
            )}

            {templatesLoading ? (
              <LoadingState />
            ) : templates.length === 0 ? (
              <EmptyState message="No templates yet. Click 'Add Template' to create one." />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {templates.map((template) => (
                  <div key={template.id} className="p-4 bg-gray-50 rounded-xl">
                    <div className="flex items-start justify-between">
                      <div>
                        <h4 className="font-medium text-gray-900">{template.name}</h4>
                        <p className="text-sm text-gray-600 mt-1">{template.content}</p>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={() => { setEditingTemplate(template); setShowTemplateForm(true); }} className="text-purple-600 hover:underline">Edit</button>
                        <button onClick={() => deleteTemplate(template.id)} className="text-red-600 hover:underline">Delete</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <Card title="Notification History">
            {/* Month Filter */}
            <div className="flex flex-wrap items-center gap-4 mb-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">Month:</label>
                <input
                  type="month"
                  value={filterMonth}
                  onChange={(e) => setFilterMonth(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              <button
                onClick={fetchNotifications}
                className="p-2 text-gray-400 hover:text-gray-600 ml-auto"
              >
                <RefreshIcon />
              </button>
            </div>

            {/* Notifications List */}
            {notificationsLoading ? (
              <LoadingState />
            ) : notifications.length === 0 ? (
              <EmptyState message="No notifications sent yet." />
            ) : (
              <div className="space-y-3">
                {notifications
                  .filter((n) => {
                    if (filterMonth) {
                      const notificationDate = new Date(n.timestamp).toISOString().slice(0, 7)
                      return notificationDate === filterMonth
                    }
                    return true
                  })
                  .map((notification) => (
                    <div key={notification.id} className="p-4 bg-gray-50 rounded-xl">
                      <div className="flex items-start justify-between mb-2">
                        <span className="font-medium text-gray-900">{notification.phone_number}</span>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          notification.status === 'sent' ? 'bg-green-100 text-green-700' :
                          notification.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {notification.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{notification.message}</p>
                      <p className="text-xs text-gray-400">{new Date(notification.timestamp).toLocaleString()}</p>
                    </div>
                  ))}
              </div>
            )}
          </Card>
        )}

        {/* Earnings Tab */}
        {activeTab === 'earnings' && (
          <Card title="Earnings">
            {/* Filters */}
            <div className="flex flex-wrap items-center gap-4 mb-6">
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setEarningsFilter('all')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    earningsFilter === 'all'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  All
                </button>
                <button
                  onClick={() => setEarningsFilter('service')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    earningsFilter === 'service'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  By Service
                </button>
                <button
                  onClick={() => setEarningsFilter('contact')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    earningsFilter === 'contact'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  By Contact
                </button>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">Month:</label>
                <input
                  type="month"
                  value={earningsMonth}
                  onChange={(e) => setEarningsMonth(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
            </div>

            {/* Calculate Earnings */}
            {(() => {
              const earningsMonthDate = new Date(earningsMonth + '-01')
              const earningsYear = earningsMonthDate.getFullYear()
              const earningsMonthNum = earningsMonthDate.getMonth() + 1
              
              // Get all active subscriptions (started before or during the selected month)
              const activeSubscriptions = subscriptions.filter(sub => {
                const subscriptionStart = sub.started_at || new Date().toISOString().split('T')[0]
                const startDate = new Date(subscriptionStart)
                const startYear = startDate.getFullYear()
                const startMonth = startDate.getMonth() + 1
                // Include if subscription started in or before the selected month
                return (startYear < earningsYear) || (startYear === earningsYear && startMonth <= earningsMonthNum)
              })

              // Calculate totals
              let totalRevenue = 0
              let totalCost = 0
              let totalDue = 0  // Due for selected month only
              let totalOutstanding = 0  // Total unpaid across all months
              const serviceEarnings: Record<string, { revenue: number; cost: number; profit: number; due: number; outstanding: number }> = {}
              const contactEarnings: Record<string, { revenue: number; cost: number; profit: number; due: number; outstanding: number }> = {}

              activeSubscriptions.forEach(sub => {
                const amount = sub.service?.amount || 0
                const cost = sub.service?.actual_cost || 0
                
                // Calculate for selected month only (Due) and all months (Outstanding)
                const subscriptionStart = sub.started_at || new Date().toISOString().split('T')[0]
                const startDate = new Date(subscriptionStart)
                const startYear = startDate.getFullYear()
                const startMonth = startDate.getMonth() + 1
                
                let paidAmount = 0
                let outstandingAmount = 0
                
                // Calculate from subscription start to selected month
                for (let year = startYear; year <= earningsYear; year++) {
                  const monthStart = (year === startYear) ? startMonth : 1
                  const monthEnd = (year === earningsYear) ? earningsMonthNum : 12
                  
                  for (let month = monthStart; month <= monthEnd; month++) {
                    // Find all payments for this subscription
                    const subPayments = payments.filter(p => p.subscription_id === sub.id)
                    
                    // Find payment for this specific month
                    const paymentForMonth = subPayments.find(p => {
                      const paymentDate = new Date(p.payment_month)
                      return paymentDate.getFullYear() === year && paymentDate.getMonth() + 1 === month
                    })
                    
                    const isPaidForMonth = paymentForMonth && paymentForMonth.amount_paid >= paymentForMonth.amount_due
                    
                    // For selected month only, calculate due
                    if (year === earningsYear && month === earningsMonthNum) {
                      // Due for current month
                      if (!isPaidForMonth) {
                        totalDue += amount
                      }
                    }
                    
                    // Track outstanding across all months
                    if (!isPaidForMonth) {
                      outstandingAmount += amount
                    } else if (paymentForMonth) {
                      paidAmount += paymentForMonth.amount_paid
                    }
                  }
                }
                
                const revenue = paidAmount
                const profit = revenue - cost
                
                totalRevenue += revenue
                totalCost += cost
                totalOutstanding += outstandingAmount

                // By Service
                const serviceName = sub.service?.name || 'Unknown'
                if (!serviceEarnings[serviceName]) {
                  serviceEarnings[serviceName] = { revenue: 0, cost: 0, profit: 0, due: 0, outstanding: 0 }
                }
                serviceEarnings[serviceName].revenue += revenue
                serviceEarnings[serviceName].cost += cost
                serviceEarnings[serviceName].profit += profit
                // Due for selected month
                const subPayments = payments.filter(p => p.subscription_id === sub.id)
                const dueForMonth = subPayments.some(p => {
                  const pd = new Date(p.payment_month)
                  return pd.getFullYear() === earningsYear && pd.getMonth() + 1 === earningsMonthNum && p.amount_paid >= p.amount_due
                }) ? 0 : amount
                serviceEarnings[serviceName].due += dueForMonth
                serviceEarnings[serviceName].outstanding += outstandingAmount

                // By Contact
                const contact = contacts.find(c => c.id === sub.contact_id)
                const contactName = contact?.name || 'Unknown'
                if (!contactEarnings[contactName]) {
                  contactEarnings[contactName] = { revenue: 0, cost: 0, profit: 0, due: 0, outstanding: 0 }
                }
                contactEarnings[contactName].revenue += revenue
                contactEarnings[contactName].cost += cost
                contactEarnings[contactName].profit += profit
                contactEarnings[contactName].due += dueForMonth
                contactEarnings[contactName].outstanding += outstandingAmount
              })

              const totalProfit = totalRevenue - totalCost

              if (activeSubscriptions.length === 0) {
                return <EmptyState message={`No subscriptions found for ${earningsMonth}.`} />
              }

              return (
                <div className="space-y-6">
                  {/* Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="p-4 bg-green-50 rounded-xl">
                      <p className="text-sm text-green-600">Total Revenue</p>
                      <p className="text-2xl font-bold text-green-700">{CURRENCY_SYMBOL}{totalRevenue.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-red-50 rounded-xl">
                      <p className="text-sm text-red-600">Total Cost</p>
                      <p className="text-2xl font-bold text-red-700">{CURRENCY_SYMBOL}{totalCost.toFixed(2)}</p>
                    </div>
                    <div className={`p-4 rounded-xl ${totalProfit >= 0 ? 'bg-blue-50' : 'bg-orange-50'}`}>
                      <p className={`text-sm ${totalProfit >= 0 ? 'text-blue-600' : 'text-orange-600'}`}>Net Profit</p>
                      <p className={`text-2xl font-bold ${totalProfit >= 0 ? 'text-blue-700' : 'text-orange-700'}`}>
                        {CURRENCY_SYMBOL}{totalProfit.toFixed(2)}
                      </p>
                    </div>
                    <div className="p-4 bg-yellow-50 rounded-xl">
                      <p className="text-sm text-yellow-600">Total Due</p>
                      <p className="text-2xl font-bold text-yellow-700">{CURRENCY_SYMBOL}{totalDue.toFixed(2)}</p>
                    </div>
                    <div className="p-4 bg-orange-50 rounded-xl">
                      <p className="text-sm text-orange-600">Outstanding</p>
                      <p className="text-2xl font-bold text-orange-700">{CURRENCY_SYMBOL}{totalOutstanding.toFixed(2)}</p>
                    </div>
                  </div>

                  {/* By Service */}
                  {earningsFilter === 'all' || earningsFilter === 'service' ? (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Earnings by Service</h4>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Service</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Revenue</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Cost</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Profit</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Due</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Outstanding</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(serviceEarnings).map(([service, data]) => (
                              <tr key={service} className="border-b hover:bg-gray-50">
                                <td className="py-3 px-4 font-medium">{service}</td>
                                <td className="py-3 px-4 text-green-600">{CURRENCY_SYMBOL}{data.revenue.toFixed(2)}</td>
                                <td className="py-3 px-4 text-red-600">{CURRENCY_SYMBOL}{data.cost.toFixed(2)}</td>
                                <td className={`py-3 px-4 font-medium ${data.profit >= 0 ? 'text-blue-600' : 'text-orange-600'}`}>
                                  {CURRENCY_SYMBOL}{data.profit.toFixed(2)}
                                </td>
                                <td className="py-3 px-4 text-yellow-600">{CURRENCY_SYMBOL}{data.due.toFixed(2)}</td>
                                <td className={`py-3 px-4 font-medium ${data.outstanding > 0 ? 'text-orange-600' : 'text-green-600'}`}>
                                  {CURRENCY_SYMBOL}{data.outstanding.toFixed(2)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}

                  {/* By Contact */}
                  {earningsFilter === 'all' || earningsFilter === 'contact' ? (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Earnings by Contact</h4>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Contact</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Revenue</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Cost</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Profit</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Due</th>
                              <th className="text-left py-3 px-4 font-medium text-gray-600">Outstanding</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(contactEarnings).map(([contact, data]) => (
                              <tr key={contact} className="border-b hover:bg-gray-50">
                                <td className="py-3 px-4 font-medium">{contact}</td>
                                <td className="py-3 px-4 text-green-600">{CURRENCY_SYMBOL}{data.revenue.toFixed(2)}</td>
                                <td className="py-3 px-4 text-red-600">{CURRENCY_SYMBOL}{data.cost.toFixed(2)}</td>
                                <td className={`py-3 px-4 font-medium ${data.profit >= 0 ? 'text-blue-600' : 'text-orange-600'}`}>
                                  {CURRENCY_SYMBOL}{data.profit.toFixed(2)}
                                </td>
                                <td className="py-3 px-4 text-yellow-600">{CURRENCY_SYMBOL}{data.due.toFixed(2)}</td>
                                <td className={`py-3 px-4 font-medium ${data.outstanding > 0 ? 'text-orange-600' : 'text-green-600'}`}>
                                  {CURRENCY_SYMBOL}{data.outstanding.toFixed(2)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                </div>
              )
            })()}
          </Card>
        )}
      </main>
    </div>
  )
}

// Subscription item component for displaying in the list
async function SubscriptionItem({ 
  subscription, 
  isSelected, 
  onClick 
}: { 
  subscription: Subscription
  isSelected: boolean
  onClick: () => void
}) {
  const { fetchPaymentsForSubscription } = usePayments()
  
  const [paid, setPaid] = useState(false)
  
  useEffect(() => {
    const checkPayment = async () => {
      const currentMonth = getCurrentMonth()
      const currentDate = new Date()
      const subPayments = await fetchPaymentsForSubscription(subscription.id)
      const isPaid = subPayments.some(p => {
        const paymentDate = new Date(p.payment_month)
        const paymentMonthStr = `${paymentDate.getFullYear()}-${String(paymentDate.getMonth() + 1).padStart(2, '0')}`
        return paymentMonthStr === currentMonth && p.amount_paid >= p.amount_due
      })
      setPaid(isPaid)
    }
    checkPayment()
  }, [subscription.id, fetchPaymentsForSubscription])

  return (
    <button
      onClick={onClick}
      disabled={paid}
      className={`w-full text-left p-3 rounded-lg transition-colors ${
        isSelected
          ? 'bg-purple-100 border-purple-500'
          : paid
          ? 'bg-green-50 cursor-not-allowed opacity-60'
          : 'bg-gray-50 hover:bg-gray-100'
      }`}
    >
      <div className="flex justify-between items-start">
        <div>
          <p className="font-medium text-gray-900">{subscription.service?.name || 'Unknown Service'}</p>
          <p className="text-sm text-gray-500">${subscription.service?.amount || 0}</p>
        </div>
        {paid && (
          <span className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded-full">
            Paid
          </span>
        )}
      </div>
    </button>
  )
}

// Reusable sub-components
function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      {children}
    </div>
  )
}

function LoadingState() {
  return <p className="text-gray-500">Loading...</p>
}

function EmptyState({ message }: { message: string }) {
  return <p className="text-gray-500 text-sm">{message}</p>
}

function Alert({ type, message }: { type: 'error' | 'success'; message: string }) {
  return (
    <div className={`p-3 rounded-lg text-sm ${type === 'error' ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'}`}>
      {message}
    </div>
  )
}
