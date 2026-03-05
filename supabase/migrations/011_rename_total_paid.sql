-- Rename total_paid to total_due in service_subscriptions
ALTER TABLE service_subscriptions 
ADD COLUMN IF NOT EXISTS total_due DECIMAL(10,2) DEFAULT 0;

-- Copy values from total_paid if exists, otherwise calculate from service
UPDATE service_subscriptions ss
SET total_due = COALESCE(
  (SELECT SUM(sp.amount_due) FROM subscription_payments sp WHERE sp.subscription_id = ss.id),
  (SELECT s.amount FROM services s WHERE s.id = ss.service_id)
);

-- Drop old column if exists
ALTER TABLE service_subscriptions DROP COLUMN IF EXISTS total_paid;
