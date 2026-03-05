import { getBaseUrl } from "@/utils/misc";

export const BASE_URL = getBaseUrl();

// Currency constants - Indian Rupee
export const CURRENCY_SYMBOL = '₹'

// Date format constants
export const DATE_FORMAT = {
  DISPLAY: 'en-US',
  DEFAULT_LOCALE: 'en-US'
}

// Payment status colors
export const PAYMENT_STATUS = {
  PAID: 'paid',
  PARTIAL: 'partial', 
  UNPAID: 'unpaid'
} as const
