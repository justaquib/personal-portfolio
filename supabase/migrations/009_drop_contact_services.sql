-- Migration: Drop the old contact_services table after migrating to new schema
-- Only run this after verifying the new service_subscriptions and subscription_payments tables are working

-- First, drop the RLS policy
DROP POLICY IF EXISTS "Users can manage own contact services" ON contact_services;

-- Drop the junction table
DROP TABLE IF EXISTS contact_services;

-- Optional: Add a note that this table was replaced by service_subscriptions and subscription_payments
COMMENT ON TABLE service_subscriptions IS 'Replaces the old contact_services table. Each record represents a service subscription for a contact.';
COMMENT ON TABLE subscription_payments IS 'Tracks payments for each subscription. Supports multiple payments per subscription and partial payments.';
