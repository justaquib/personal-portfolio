-- Migration 013: Add invoice_id and sub_invoice_id as VARCHAR for readable IDs (0001, 0001-a, etc.)

-- Drop the function if exists
DROP FUNCTION IF EXISTS get_next_invoice_id();

-- Check if invoice_id column exists and is UUID type, then drop it first
DO $$
BEGIN
    -- Check if column exists and is UUID
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'subscription_payments' 
        AND column_name = 'invoice_id' 
        AND data_type = 'uuid'
    ) THEN
        -- Drop the old UUID column
        ALTER TABLE subscription_payments DROP COLUMN IF EXISTS invoice_id;
    END IF;
    
    -- Check if sub_invoice_id column exists and is UUID type, then drop it first
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'subscription_payments' 
        AND column_name = 'sub_invoice_id' 
        AND data_type = 'uuid'
    ) THEN
        -- Drop the old UUID column
        ALTER TABLE subscription_payments DROP COLUMN IF EXISTS sub_invoice_id;
    END IF;
END $$;

-- Add new VARCHAR columns
ALTER TABLE subscription_payments 
ADD COLUMN IF NOT EXISTS invoice_id VARCHAR(20),
ADD COLUMN IF NOT EXISTS sub_invoice_id VARCHAR(20);

-- Create a function to generate next invoice number
CREATE OR REPLACE FUNCTION get_next_invoice_id()
RETURNS VARCHAR(20) AS $$
DECLARE
  next_num INTEGER;
  result VARCHAR(20);
BEGIN
  SELECT COALESCE(MAX(CAST(invoice_id AS INTEGER)), 0) + 1 INTO next_num
  FROM subscription_payments
  WHERE invoice_id ~ '^[0-9]+$';
  
  result := LPAD(next_num::VARCHAR, 4, '0');
  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Update existing records with sequential invoice IDs
-- This assigns IDs in order of creation
UPDATE subscription_payments sp
SET invoice_id = (
    SELECT LPAD(ROW_NUMBER() OVER (ORDER BY created_at)::VARCHAR, 4, '0')
    FROM subscription_payments sp2
    WHERE sp2.created_at <= sp.created_at
    LIMIT 1
)
WHERE invoice_id IS NULL OR invoice_id = '';

-- Make invoice_id not null for records that have been updated
ALTER TABLE subscription_payments ALTER COLUMN invoice_id SET NOT NULL;
