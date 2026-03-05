-- Add billing_month field to subscription_payments to identify the billing period
ALTER TABLE subscription_payments
ADD COLUMN IF NOT EXISTS billing_month VARCHAR(7);

-- Update existing records with billing_month based on payment_month
UPDATE subscription_payments
SET billing_month = TO_CHAR(TO_DATE(payment_month, 'YYYY-MM-DD'), 'YYYY-MM')
WHERE billing_month IS NULL AND payment_month IS NOT NULL;

-- Make billing_month not null for new records
ALTER TABLE subscription_payments ALTER COLUMN billing_month SET NOT NULL;
