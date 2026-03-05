-- Add amount and service columns to contacts table
ALTER TABLE contacts 
ADD COLUMN IF NOT EXISTS amount DECIMAL(10, 2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS service VARCHAR(255) DEFAULT '';

-- Update existing rows to set default values if needed
UPDATE contacts SET amount = 0 WHERE amount IS NULL;
UPDATE contacts SET service = '' WHERE service IS NULL;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_contacts_amount ON contacts(amount);
CREATE INDEX IF NOT EXISTS idx_contacts_service ON contacts(service);
