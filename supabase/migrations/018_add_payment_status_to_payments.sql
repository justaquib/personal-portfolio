-- Add payment_status column to payments table
ALTER TABLE payments 
ADD COLUMN IF NOT EXISTS payment_status TEXT NOT NULL DEFAULT 'unpaid' 
CHECK (payment_status IN ('paid', 'partial', 'unpaid'));

-- Update existing records based on amount_paid vs amount_due
UPDATE payments 
SET payment_status = 'paid' 
WHERE amount_paid >= amount_due AND amount_due > 0;

UPDATE payments 
SET payment_status = 'partial' 
WHERE amount_paid > 0 AND amount_paid < amount_due;

UPDATE payments 
SET payment_status = 'unpaid' 
WHERE amount_paid = 0 OR amount_due = 0;
