-- Create contact_services junction table for many-to-many relationship
CREATE TABLE IF NOT EXISTS contact_services (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  contact_id UUID REFERENCES contacts(id) ON DELETE CASCADE,
  service_id UUID REFERENCES services(id) ON DELETE CASCADE,
  payment_received BOOLEAN DEFAULT false,
  payment_month VARCHAR(7),
  payment_received_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(contact_id, service_id)
);

-- Enable RLS
ALTER TABLE contact_services ENABLE ROW LEVEL SECURITY;

-- Create policy for contact_services
DROP POLICY IF EXISTS "Users can manage own contact services" ON contact_services;

CREATE POLICY "Users can manage own contact services" ON contact_services
  FOR ALL
  USING (
    EXISTS (
      SELECT 1 FROM contacts 
      WHERE contacts.id = contact_services.contact_id 
      AND contacts.user_id = auth.uid()
    )
  );

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_contact_services_contact ON contact_services(contact_id);
CREATE INDEX IF NOT EXISTS idx_contact_services_service ON contact_services(service_id);

-- Remove the service_id column from contacts if it exists (we'll use junction table now)
-- ALTER TABLE contacts DROP COLUMN IF EXISTS service_id;
