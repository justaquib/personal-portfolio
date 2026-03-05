-- Supabase Database Migration
-- Run this SQL in your Supabase SQL Editor to create/update the notifications table

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create notifications table if not exists
CREATE TABLE IF NOT EXISTS public.notifications (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  phone_number VARCHAR(50) NOT NULL,
  message TEXT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  status VARCHAR(20) DEFAULT 'pending' NOT NULL CHECK (
    status IN ('pending', 'sent', 'failed')
  ),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance (IF NOT EXISTS for safety)
CREATE INDEX IF NOT EXISTS idx_notifications_phone_number 
  ON public.notifications(phone_number);

CREATE INDEX IF NOT EXISTS idx_notifications_timestamp 
  ON public.notifications(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_notifications_status 
  ON public.notifications(status);

-- Enable Row Level Security (RLS) if not already enabled
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

-- Create RLS policies if they don't exist
DO $$
BEGIN
  -- Check if policy exists
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to read notifications'
  ) THEN
    CREATE POLICY "Allow authenticated users to read notifications"
      ON public.notifications
      FOR SELECT
      TO authenticated
      USING (true);
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to insert notifications'
  ) THEN
    CREATE POLICY "Allow authenticated users to insert notifications"
      ON public.notifications
      FOR INSERT
      TO authenticated
      WITH CHECK (true);
  END IF;
END
$$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to update notifications'
  ) THEN
    CREATE POLICY "Allow authenticated users to update notifications"
      ON public.notifications
      FOR UPDATE
      TO authenticated
      USING (true)
      WITH CHECK (true);
  END IF;
END
$$;

-- Create updated_at trigger function if not exists
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Add or update the trigger
DROP TRIGGER IF EXISTS update_notifications_updated_at ON public.notifications;
CREATE TRIGGER update_notifications_updated_at
  BEFORE UPDATE ON public.notifications
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE public.notifications IS 'Stores WhatsApp payment notification records';
COMMENT ON COLUMN public.notifications.phone_number IS 'Recipient phone number in international format (e.g., +1234567890)';
COMMENT ON COLUMN public.notifications.message IS 'Content of the payment notification message';
COMMENT ON COLUMN public.notifications.timestamp IS 'When the notification was sent';
COMMENT ON COLUMN public.notifications.status IS 'Delivery status: pending, sent, or failed';
