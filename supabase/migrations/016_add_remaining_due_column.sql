-- Migration: Add remaining_due column to subscription_payments table
-- This column tracks the remaining amount due after partial payments

ALTER TABLE subscription_payments 
ADD COLUMN IF NOT EXISTS remaining_due DECIMAL(12, 2) DEFAULT 0;

-- Update existing records: set remaining_due = amount_due - amount_paid
UPDATE subscription_payments 
SET remaining_due = GREATEST(amount_due - amount_paid, 0)
WHERE remaining_due IS NULL OR remaining_due = 0;

-- Add NOT NULL constraint after updating existing data
ALTER TABLE subscription_payments 
ALTER COLUMN remaining_due SET DEFAULT 0;

COMMENT ON COLUMN subscription_payments.remaining_due IS 'Remaining amount due after partial payments';
