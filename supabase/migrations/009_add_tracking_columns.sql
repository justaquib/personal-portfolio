-- Add columns to track total paid in service_subscriptions
ALTER TABLE service_subscriptions 
ADD COLUMN IF NOT EXISTS total_paid DECIMAL(10,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_payment_date DATE;
