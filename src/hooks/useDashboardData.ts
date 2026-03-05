import { useState, useCallback } from 'react'
import { createClient } from '@/lib/supabase/client'
import type { Contact, MessageTemplate, Notification, ContactFormData, TemplateFormData, Service, ServiceFormData, Subscription, Payment, PaymentFormData } from '@/types/database'

const supabase = createClient()

// Subscription hook - for linking services to contacts (replaces contact_services)
export function useSubscriptions() {
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSubscriptions = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('service_subscriptions')
        .select('*, service:services(*), contact:contacts(*)')
        .order('created_at', { ascending: false })

      if (fetchError) throw fetchError
      setSubscriptions(data || [])
    } catch (err: any) {
      setError(err.message)
      setSubscriptions([])
    } finally {
      setLoading(false)
    }
  }, [])

  const saveSubscription = useCallback(async (
    data: { contact_id: string; service_id: string; started_at?: string; user_id?: string },
    id?: string
  ) => {
    if (id) {
      const { error } = await supabase
        .from('service_subscriptions')
        .update({
          contact_id: data.contact_id,
          service_id: data.service_id,
          started_at: data.started_at || new Date().toISOString().split('T')[0]
        })
        .eq('id', id)
      if (error) throw error
    } else {
      const { error } = await supabase
        .from('service_subscriptions')
        .insert({
          contact_id: data.contact_id,
          service_id: data.service_id,
          started_at: data.started_at || new Date().toISOString().split('T')[0],
          user_id: data.user_id
        })
      if (error) throw error
    }
    await fetchSubscriptions()
  }, [fetchSubscriptions])

  const deleteSubscription = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('service_subscriptions')
      .delete()
      .eq('id', id)
    if (error) throw error
    await fetchSubscriptions()
  }, [fetchSubscriptions])

  const getSubscriptionsForContact = useCallback(async (contactId: string) => {
    try {
      const { data, error } = await supabase
        .from('service_subscriptions')
        .select('*, service:services(*), contact:contacts(*)')
        .eq('contact_id', contactId)

      if (error) throw error
      return data || []
    } catch (err: any) {
      return []
    }
  }, [])

  return { subscriptions, loading, error, fetchSubscriptions, saveSubscription, deleteSubscription, getSubscriptionsForContact }
}

// Payment hook - for tracking payments
export function usePayments() {
  const [payments, setPayments] = useState<Payment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPayments = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('subscription_payments')
        .select('*')
        .order('payment_month', { ascending: false })

      if (fetchError) throw fetchError
      setPayments(data || [])
    } catch (err: any) {
      setError(err.message)
      setPayments([])
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchPaymentsForSubscription = useCallback(async (subscriptionId: string) => {
    try {
      const { data, error } = await supabase
        .from('subscription_payments')
        .select('*')
        .eq('subscription_id', subscriptionId)
        .order('payment_month', { ascending: false })

      if (error) throw error
      return data || []
    } catch (err: any) {
      return []
    }
  }, [])

  const savePayment = useCallback(async (data: PaymentFormData & { user_id?: string }, id?: string) => {
    try {
      console.log('Saving payment with data:', JSON.stringify(data, null, 2))
      
      // Get the service amount
      const { data: subData, error: subError } = await supabase
        .from('service_subscriptions')
        .select('service:services(amount), total_due')
        .eq('id', data.subscription_id)
        .single()
      
      if (subError) {
        console.error('Error fetching subscription:', subError)
        throw new Error('Subscription not found')
      }
      
      console.log('Subscription data:', JSON.stringify(subData, null, 2))
      
      const serviceAmount = (subData as any)?.service?.amount || data.amount_due || 0
      let currentTotalDue = (subData as any)?.total_due || 0
      
      // Helper function to generate invoice ID (0001, 0002, etc.)
      const generateInvoiceId = async (): Promise<string> => {
        const { data: lastInvoice } = await supabase
          .from('subscription_payments')
          .select('invoice_id')
          .not('invoice_id', 'is', null)
          .order('created_at', { ascending: false })
          .limit(1)
          
        let nextNum = 1
        if (lastInvoice && lastInvoice[0]?.invoice_id) {
          const lastNum = parseInt(lastInvoice[0].invoice_id) || 0
          nextNum = lastNum + 1
        }
        console.log('Generated invoice ID:', String(nextNum).padStart(4, '0'))
        return String(nextNum).padStart(4, '0')
      }
      
      // Helper function to generate sub-invoice ID (0001-a, 0001-b, etc.)
      const generateSubInvoiceId = async (invoiceId: string): Promise<string> => {
        const { data: existingSubInvoices } = await supabase
          .from('subscription_payments')
          .select('sub_invoice_id')
          .eq('invoice_id', invoiceId)
          .not('sub_invoice_id', 'is', null)
        
        const letterCode = 97 + (existingSubInvoices?.length || 0) // 97 = 'a'
        const subId = `${invoiceId}-${String.fromCharCode(letterCode)}`
        console.log('Generated sub-invoice ID:', subId, 'existing count:', existingSubInvoices?.length)
        return subId
      }
      
      if (id) {
        // Update existing payment by ID
        const remainingDue = data.amount_due - data.amount_paid
        const { error } = await supabase
          .from('subscription_payments')
          .update({
            amount_due: serviceAmount,
            amount_paid: data.amount_paid,
            remaining_due: Math.max(0, remainingDue),
            payment_date: data.payment_date,
            payment_method: data.payment_method,
            notes: data.notes
          })
          .eq('id', id)
        if (error) throw error
        
        // Update subscription's last_payment_date
        await supabase
          .from('service_subscriptions')
          .update({ last_payment_date: data.payment_date })
          .eq('id', data.subscription_id)
      } else {
        // Check if invoice already exists for this payment month
        let existingInvoice: any[] | null = null
        try {
          const result = await supabase
            .from('subscription_payments')
            .select('id, invoice_id, amount_due, amount_paid')
            .eq('subscription_id', data.subscription_id)
            .eq('payment_month', data.payment_month)
            .is('sub_invoice_id', null)
            .limit(1)
          existingInvoice = result.data
          if (result.error) throw result.error
        } catch (err) {
          console.log('Error checking existing invoice:', err)
          existingInvoice = null
        }

        if (existingInvoice && existingInvoice.length > 0) {
          // Partial payment - add sub_invoice to existing invoice
          const mainInvoice = existingInvoice[0]
          console.log('Found existing invoice for partial payment:', mainInvoice)
          const newSubInvoiceId = await generateSubInvoiceId(mainInvoice.invoice_id)
          
          // Calculate remaining amount to be paid (original amount - already paid)
          const remainingAmount = mainInvoice.amount_due - mainInvoice.amount_paid
          console.log('Remaining amount:', remainingAmount)
          
          // Try to insert sub-invoice
          let insertError: any = null
          const remainingDueSub = remainingAmount - data.amount_paid
          const { error } = await supabase
            .from('subscription_payments')
            .insert({
              subscription_id: data.subscription_id,
              payment_month: data.payment_month,
              invoice_id: mainInvoice.invoice_id,
              sub_invoice_id: newSubInvoiceId,
              amount_due: remainingAmount,
              amount_paid: data.amount_paid,
              remaining_due: Math.max(0, remainingDueSub),
              payment_date: data.payment_date,
              payment_method: data.payment_method,
              notes: data.notes || 'Partial payment',
              user_id: data.user_id
            })
          insertError = error
          
          if (insertError) {
            console.error('Error inserting sub-invoice:', insertError)
            console.error('Error code:', insertError?.code)
            console.error('Error message:', insertError?.message)
            throw insertError
          }
          
          // Success! (Don't update main invoice - keep original amount_paid unchanged)
          // The remaining_due tracking is just for display purposes
          
          // Decrease total_due by payment amount
          currentTotalDue = Math.max(0, currentTotalDue - data.amount_paid)
          
          await supabase
            .from('service_subscriptions')
            .update({
              total_due: currentTotalDue,
              last_payment_date: data.payment_date
            })
            .eq('id', data.subscription_id)
        } else {
          // New invoice - create new payment record with new invoice_id
          const newInvoiceId = await generateInvoiceId()
          console.log('Creating new invoice with ID:', newInvoiceId)
          const newRemainingDue = serviceAmount - data.amount_paid
          
          const { error } = await supabase
            .from('subscription_payments')
            .insert({ 
              subscription_id: data.subscription_id,
              payment_month: data.payment_month,
              invoice_id: newInvoiceId,
              sub_invoice_id: null,
              amount_due: serviceAmount,
              amount_paid: data.amount_paid,
              remaining_due: Math.max(0, newRemainingDue),
              payment_date: data.payment_date,
              payment_method: data.payment_method,
              notes: data.notes,
              user_id: data.user_id
            })
          if (error) {
            console.error('Error inserting new invoice:', error)
            throw error
          }
          
          // Increase total_due for new billing period
          currentTotalDue = currentTotalDue + serviceAmount
          
          await supabase
            .from('service_subscriptions')
            .update({
              total_due: currentTotalDue,
              last_payment_date: data.payment_date
            })
            .eq('id', data.subscription_id)
        }
      }
      await fetchPayments()
    } catch (err: any) {
      console.error('Error saving payment:', err)
      console.error('Error details:', JSON.stringify(err, Object.getOwnPropertyNames(err)))
      throw err
    }
  }, [fetchPayments])

  const deletePayment = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('subscription_payments')
      .delete()
      .eq('id', id)
    if (error) throw error
    await fetchPayments()
  }, [fetchPayments])

  return { payments, loading, error, fetchPayments, savePayment, deletePayment, fetchPaymentsForSubscription }
}

// Service hooks
export function useServices() {
  const [services, setServices] = useState<Service[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchServices = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('services')
        .select('*')
        .order('name', { ascending: true })

      if (fetchError) throw fetchError
      setServices(data || [])
    } catch (err: any) {
      setError(err.message)
      setServices([])
    } finally {
      setLoading(false)
    }
  }, [])

  const saveService = useCallback(async (data: ServiceFormData, userId: string, id?: string) => {
    if (id) {
      const { error } = await supabase
        .from('services')
        .update(data)
        .eq('id', id)
      if (error) throw error
    } else {
      const { error } = await supabase
        .from('services')
        .insert({ ...data, user_id: userId })
      if (error) throw error
    }
    await fetchServices()
  }, [fetchServices])

  const deleteService = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('services')
      .delete()
      .eq('id', id)
    if (error) throw error
    await fetchServices()
  }, [fetchServices])

  return { services, loading, error, fetchServices, saveService, deleteService }
}

// Contact hooks
export function useContacts() {
  const [contacts, setContacts] = useState<Contact[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchContacts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('contacts')
        .select('*')
        .order('name', { ascending: true })

      if (fetchError) throw fetchError
      setContacts(data || [])
    } catch (err: any) {
      setError(err.message)
      setContacts([])
    } finally {
      setLoading(false)
    }
  }, [])

  const saveContact = useCallback(async (data: ContactFormData, userId: string, id?: string) => {
    if (id) {
      const { error } = await supabase
        .from('contacts')
        .update(data)
        .eq('id', id)
      if (error) throw error
    } else {
      const { error } = await supabase
        .from('contacts')
        .insert({ ...data, user_id: userId })
      if (error) throw error
    }
    await fetchContacts()
  }, [fetchContacts])

  const deleteContact = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('contacts')
      .delete()
      .eq('id', id)
    if (error) throw error
    await fetchContacts()
  }, [fetchContacts])

  return { contacts, loading, error, fetchContacts, saveContact, deleteContact }
}

// Template hooks
export function useTemplates() {
  const [templates, setTemplates] = useState<MessageTemplate[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTemplates = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('message_templates')
        .select('*')
        .order('name', { ascending: true })

      if (fetchError) throw fetchError
      setTemplates(data || [])
    } catch (err: any) {
      setError(err.message)
      setTemplates([])
    } finally {
      setLoading(false)
    }
  }, [])

  const saveTemplate = useCallback(async (data: TemplateFormData, userId: string, id?: string) => {
    if (id) {
      const { error } = await supabase
        .from('message_templates')
        .update(data)
        .eq('id', id)
      if (error) throw error
    } else {
      const { error } = await supabase
        .from('message_templates')
        .insert({ ...data, user_id: userId })
      if (error) throw error
    }
    await fetchTemplates()
  }, [fetchTemplates])

  const deleteTemplate = useCallback(async (id: string) => {
    const { error } = await supabase
      .from('message_templates')
      .delete()
      .eq('id', id)
    if (error) throw error
    await fetchTemplates()
  }, [fetchTemplates])

  return { templates, loading, error, fetchTemplates, saveTemplate, deleteTemplate }
}

// Notification hooks
export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchNotifications = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data, error: fetchError } = await supabase
        .from('notifications')
        .select('*')
        .order('timestamp', { ascending: false })
        .limit(50)

      if (fetchError && fetchError.code !== '42P01') throw fetchError
      setNotifications(data || [])
    } catch (err: any) {
      setError(err.message)
      setNotifications([])
    } finally {
      setLoading(false)
    }
  }, [])

  const sendNotification = useCallback(async (phoneNumber: string, message: string) => {
    const response = await fetch('/api/notifications/send', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phoneNumber, message }),
    })

    const result = await response.json()
    if (!response.ok) throw new Error(result.error || 'Failed to send')

    // Save to database
    try {
      await supabase.from('notifications').insert({
        phone_number: phoneNumber,
        message,
        timestamp: new Date().toISOString(),
        status: result.success ? 'sent' : 'failed',
      })
    } catch {
      // Ignore database save errors
    }

    await fetchNotifications()
    return result
  }, [fetchNotifications])

  return { notifications, loading, error, fetchNotifications, sendNotification }
}
