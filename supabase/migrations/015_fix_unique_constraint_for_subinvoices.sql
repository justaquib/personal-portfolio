-- Fix the unique constraint to allow sub-invoices
-- The original constraint only allows one payment per subscription per month
-- We need to allow multiple sub-invoices (partial payments) for the same month

-- Drop the existing unique constraint
ALTER TABLE subscription_payments DROP CONSTRAINT IF EXISTS subscription_payments_subscription_id_payment_month_key;

-- Add new constraint that includes sub_invoice_id
-- This allows multiple records when sub_invoice_id is NOT NULL (sub-invoices)
-- But still prevents duplicate main invoices (where sub_invoice_id is NULL)
ALTER TABLE subscription_payments ADD CONSTRAINT subscription_payments_subscription_id_payment_month_sub_key UNIQUE (subscription_id, payment_month, sub_invoice_id);
