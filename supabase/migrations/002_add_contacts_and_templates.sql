-- Supabase Database Migration
-- Run this SQL in your Supabase SQL Editor to add/update contacts and templates tables

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create contacts table if not exists (with all fields including new ones)
CREATE TABLE IF NOT EXISTS public.contacts (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  company VARCHAR(255),
  email VARCHAR(255),
  phone_number VARCHAR(50) NOT NULL,
  address TEXT,
  notes TEXT,
  is_active BOOLEAN DEFAULT true NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add new columns if they don't exist (for existing tables)
ALTER TABLE public.contacts ADD COLUMN IF NOT EXISTS company VARCHAR(255);
ALTER TABLE public.contacts ADD COLUMN IF NOT EXISTS email VARCHAR(255);
ALTER TABLE public.contacts ADD COLUMN IF NOT EXISTS address TEXT;
ALTER TABLE public.contacts ADD COLUMN IF NOT EXISTS notes TEXT;

-- Create message templates table if not exists
CREATE TABLE IF NOT EXISTS public.message_templates (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS) if not already enabled
ALTER TABLE public.contacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.message_templates ENABLE ROW LEVEL SECURITY;

-- Create RLS policies if they don't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Users can manage their own contacts'
  ) THEN
    CREATE POLICY "Users can manage their own contacts"
      ON public.contacts
      FOR ALL
      TO authenticated
      USING (auth.uid() = user_id)
      WITH CHECK (auth.uid() = user_id);
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Users can manage their own templates'
  ) THEN
    CREATE POLICY "Users can manage their own templates"
      ON public.message_templates
      FOR ALL
      TO authenticated
      USING (auth.uid() = user_id)
      WITH CHECK (auth.uid() = user_id);
  END IF;
END
$$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_contacts_user_id ON public.contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_contacts_is_active ON public.contacts(is_active);
CREATE INDEX IF NOT EXISTS idx_message_templates_user_id ON public.message_templates(user_id);

-- Create updated_at trigger function if not exists
CREATE OR REPLACE FUNCTION update_contacts_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Add or update the trigger
DROP TRIGGER IF EXISTS update_contacts_updated_at ON public.contacts;
CREATE TRIGGER update_contacts_updated_at
  BEFORE UPDATE ON public.contacts
  FOR EACH ROW
  EXECUTE FUNCTION update_contacts_updated_at_column();

-- Create updated_at trigger function for templates if not exists
CREATE OR REPLACE FUNCTION update_message_templates_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Add or update the trigger for templates
DROP TRIGGER IF EXISTS update_message_templates_updated_at ON public.message_templates;
CREATE TRIGGER update_message_templates_updated_at
  BEFORE UPDATE ON public.message_templates
  FOR EACH ROW
  EXECUTE FUNCTION update_message_templates_updated_at_column();

-- Add comments
COMMENT ON TABLE public.contacts IS 'Stores contact information for sending notifications';
COMMENT ON COLUMN public.contacts.name IS 'Contact display name';
COMMENT ON COLUMN public.contacts.company IS 'Company name';
COMMENT ON COLUMN public.contacts.email IS 'Email address';
COMMENT ON COLUMN public.contacts.phone_number IS 'Contact phone number in international format';
COMMENT ON COLUMN public.contacts.address IS 'Physical address';
COMMENT ON COLUMN public.contacts.notes IS 'Additional notes';
COMMENT ON COLUMN public.contacts.is_active IS 'Whether the contact is active for sending notifications';

COMMENT ON TABLE public.message_templates IS 'Stores message templates for quick sending';
COMMENT ON COLUMN public.message_templates.name IS 'Template name';
COMMENT ON COLUMN public.message_templates.content IS 'Template message content';
