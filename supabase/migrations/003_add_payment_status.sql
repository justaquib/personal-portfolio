-- Add payment status columns to contacts table
ALTER TABLE contacts 
ADD COLUMN IF NOT EXISTS payment_received BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS payment_month VARCHAR(7) DEFAULT NULL,
ADD COLUMN IF NOT EXISTS payment_received_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_contacts_payment_month ON contacts(payment_month);
CREATE INDEX IF NOT EXISTS idx_contacts_payment_received ON contacts(payment_received);

-- Update RLS policy to include new columns
DROP POLICY IF EXISTS "Users can manage own contacts" ON contacts;

CREATE POLICY "Users can manage own contacts" ON contacts
  FOR ALL
  USING (auth.uid() = user_id);
