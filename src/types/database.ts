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
  // Location and device tracking
  country: string | null
  city: string | null
  region: string | null
  latitude: number | null
  longitude: number | null
  device_type: string | null
  browser: string | null
  os: string | null
  // Session tracking
  session_id?: string | null
  session_start?: string | null
  time_on_page?: number | null
  visitor_type?: string | null
  user_info?: {
    user_id: string
    name: string | null
    email: string | null
  } | null
  // Admin controls
  is_blocked: boolean | null
  blocked_reason: string | null
  blocked_by: string | null
  blocked_at: string | null
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
  recent_visits?: AnalyticsVisit[]
}

export interface AnalyticsSession {
  id: string
  session_id: string
  visitor_id: string
  ip_address: string | null
  user_agent: string | null
  country: string | null
  city: string | null
  region: string | null
  device_type: string | null
  browser: string | null
  os: string | null
  start_time: string
  end_time: string | null
  duration_seconds: number | null
  page_views: number
  pages_visited: string[]
  referrer: string | null
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface VisitorDetails {
  summary: {
    visitor_id: string
    visitor_type: string
    total_visits: number
    total_sessions: number
    unique_pages: number
    total_time_spent: number
    average_session_time: number
    first_visit: string
    last_visit: string
    location: {
      country: string | null
      city: string | null
      region: string | null
      ip_address: string | null
    } | null
    device_info: {
      device_type: string | null
      browser: string | null
      os: string | null
    } | null
    user_info: {
      user_id: string
      name: string | null
      email: string | null
    } | null
    is_blocked: boolean
  }
  sessions: Array<AnalyticsSession & { visits: AnalyticsVisit[] }>
  all_visits: AnalyticsVisit[]
}

export interface BlockedVisitor {
  id: string
  visitor_id: string
  reason: string | null
  blocked_by: string
  created_at: string
  updated_at: string
  blocker?: {
    name: string | null
    email: string
  }
}

export interface BlockedIP {
  id: string
  ip_address: string
  reason: string | null
  blocked_by: string
  created_at: string
  updated_at: string
  blocker?: {
    name: string | null
    email: string
  }
}

export interface BlockedData {
  blocked_visitors: BlockedVisitor[]
  blocked_ips: BlockedIP[]
}
