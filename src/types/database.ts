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
  invoice_id: string  // Main invoice: 0001, 0002, etc.
  sub_invoice_id: string | null  // Sub-invoice: 0001-a, 0001-b, etc.
  amount_due: number
  amount_paid: number
  remaining_due: number  // Remaining amount due after partial payments
  payment_date: string | null
  payment_method: string | null
  notes: string | null
  payment_status: 'paid' | 'partial' | 'unpaid'  // Payment status: paid, partial, or unpaid
  created_at: string
}

// Payment form data
export interface PaymentFormData {
  subscription_id: string
  payment_month: string
  invoice_id?: string
  sub_invoice_id?: string
  amount_due: number
  amount_paid: number
  remaining_due?: number
  payment_date: string | null
  payment_method: string
  notes: string
  payment_status?: 'paid' | 'partial' | 'unpaid'
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

export type TabType = 'send' | 'contacts' | 'services' | 'payments' | 'templates' | 'history' | 'earnings' | 'resume-builder' | 'team' | 'profile' | 'analytics'

export interface TemplateFormData {
  name: string
  content: string
}

// Role-based access control types
export type UserRole = 'super_admin' | 'admin' | 'editor' | 'viewer'

export interface UserRoleRecord {
  id: string
  user_id: string
  email: string
  name: string | null
  role: UserRole
  created_by: string | null
  created_at: string
}

export interface CreateUserRoleInput {
  email: string
  name: string
  role: UserRole
}

// Analytics types
export interface AnalyticsVisit {
  id: string
  visitor_id: string
  page: string
  user_agent: string | null
  ip_address: string | null
  referrer: string | null
  visit_date: string
  timestamp: string
  created_at: string
}

export interface AnalyticsData {
  data: Array<{
    date?: string
    period?: string
    unique_users: number
    page_views?: number
  }>
  totals: {
    total_unique_users: number
    total_page_views: number
  }
  period: string
}
