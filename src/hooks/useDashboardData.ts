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
    data: { contact_id: string; service_id: string; started_at?: string },
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
          started_at: data.started_at || new Date().toISOString().split('T')[0]
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
    if (id) {
      const { error } = await supabase
        .from('subscription_payments')
        .update({
          amount_due: data.amount_due,
          amount_paid: data.amount_paid,
          payment_date: data.payment_date,
          payment_method: data.payment_method,
          notes: data.notes
        })
        .eq('id', id)
      if (error) throw error
    } else {
      const { error } = await supabase
        .from('subscription_payments')
        .insert({ ...data, user_id: data.user_id })
      if (error) throw error
    }
    await fetchPayments()
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
