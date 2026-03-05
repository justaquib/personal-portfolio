-- Add payment_cycle column to services table
ALTER TABLE services 
ADD COLUMN IF NOT EXISTS payment_cycle VARCHAR(20) DEFAULT 'monthly';

-- Update existing records to have monthly as default
UPDATE services SET payment_cycle = 'monthly' WHERE payment_cycle IS NULL;
