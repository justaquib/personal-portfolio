// Database types for the application
export interface Contact {
  id: string
  user_id: string
  name: string
  company: string
  email: string
  phone_number: string
  address: string
  notes: string
  is_active: boolean
  created_at: string
  contact_services?: ContactService[]
}

export interface Service {
  id: string
  user_id: string
  name: string
  amount: number
  actual_cost: number
  description: string
  payment_cycle: 'monthly' | 'quarterly' | 'yearly'
  created_at: string
}

export interface ContactService {
  id: string
  contact_id: string
  service_id: string
  service?: Service
  payment_received: boolean
  payment_month: string | null
  payment_received_at: string | null
  created_at: string
}

// New Subscription type (replaces ContactService concept)
export interface Subscription {
  id: string
  user_id: string
  contact_id: string
  service_id: string
  service?: Service
  contact?: Contact
  is_active: boolean
  started_at: string
  ended_at: string | null
  created_at: string
  total_due?: number  // Track total amount due for the subscription
  last_payment_date?: string | null  // Track last payment date
  payments?: Payment[]
}

// Payment tracking for each month
export interface Payment {
  id: string
  user_id: string
  subscription_id: string
  payment_month: string
  billing_month: string
  invoice_id: string  // Main invoice: 0001, 0002, etc.
  sub_invoice_id: string | null  // Sub-invoice: 0001-a, 0001-b, etc.
  amount_due: number
  amount_paid: number
  remaining_due: number  // Remaining amount due after partial payments
  payment_date: string | null
  payment_method: string | null
  notes: string | null
  created_at: string
}

// Payment form data
export interface PaymentFormData {
  subscription_id: string
  payment_month: string
  billing_month?: string
  invoice_id?: string
  sub_invoice_id?: string
  amount_due: number
  amount_paid: number
  remaining_due?: number
  payment_date: string | null
  payment_method: string
  notes: string
}

export interface ContactFormData {
  name: string
  company: string
  email: string
  phone_number: string
  address: string
  notes: string
  is_active: boolean
}

export interface ServiceFormData {
  name: string
  amount: number
  actual_cost: number
  description: string
  payment_cycle: 'monthly' | 'quarterly' | 'yearly'
}

export interface ContactServiceFormData {
  contact_id: string
  service_id: string
  payment_received: boolean
  payment_month: string | null
}

export interface MessageTemplate {
  id: string
  user_id: string
  name: string
  content: string
  created_at: string
}

export interface Notification {
  id: string
  phone_number: string
  message: string
  timestamp: string
  status: 'pending' | 'sent' | 'failed'
}

export type TabType = 'send' | 'contacts' | 'services' | 'payments' | 'templates' | 'history' | 'earnings'

export interface TemplateFormData {
  name: string
  content: string
}
