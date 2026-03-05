-- Migration: Create service_subscriptions table to replace contact_services
-- This supports: multiple payments per month, partial payments, payment history

-- Create the new table
CREATE TABLE IF NOT EXISTS service_subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  contact_id UUID REFERENCES contacts(id) ON DELETE CASCADE NOT NULL,
  service_id UUID REFERENCES services(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN DEFAULT true,
  notes TEXT,
  
  -- New fields for better payment tracking
  started_at DATE DEFAULT CURRENT_DATE,
  ended_at DATE
);

-- Create payments table to track each payment
CREATE TABLE IF NOT EXISTS subscription_payments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  subscription_id UUID REFERENCES service_subscriptions(id) ON DELETE CASCADE NOT NULL,
  payment_month DATE NOT NULL, -- First day of the payment month
  amount_due DECIMAL(10,2) NOT NULL,
  amount_paid DECIMAL(10,2) DEFAULT 0,
  payment_date DATE,
  payment_method TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(subscription_id, payment_month)
);

-- Enable RLS
ALTER TABLE service_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_payments ENABLE ROW LEVEL SECURITY;

-- RLS Policies for service_subscriptions
CREATE POLICY "Users can manage their own subscriptions"
  ON service_subscriptions FOR ALL
  USING (auth.uid() = user_id);

-- RLS Policies for subscription_payments
CREATE POLICY "Users can manage their own payments"
  ON subscription_payments FOR ALL
  USING (auth.uid() = user_id);

-- Create indexes for better performance
CREATE INDEX idx_subscriptions_user ON service_subscriptions(user_id);
CREATE INDEX idx_subscriptions_contact ON service_subscriptions(contact_id);
CREATE INDEX idx_payments_subscription ON subscription_payments(subscription_id);
CREATE INDEX idx_payments_month ON subscription_payments(payment_month);
CREATE INDEX idx_payments_user ON subscription_payments(user_id);

-- If contact_services exists, migrate data
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'contact_services') THEN
    -- Copy data to service_subscriptions (getting user_id from contacts table)
    INSERT INTO service_subscriptions (id, user_id, contact_id, service_id, created_at)
    SELECT cs.id, c.user_id, cs.contact_id, cs.service_id, cs.created_at
    FROM contact_services cs
    JOIN contacts c ON cs.contact_id = c.id
    ON CONFLICT DO NOTHING;
    
    -- Copy payment data to subscription_payments for current month payments
    -- Note: payment_month is stored as VARCHAR(7) format "YYYY-MM", so we append "-01" to make it a valid date
    INSERT INTO subscription_payments (id, user_id, subscription_id, payment_month, amount_due, amount_paid, payment_date, created_at)
    SELECT 
      gen_random_uuid(),
      c.user_id,
      cs.id,
      CASE 
        WHEN cs.payment_month IS NOT NULL AND length(cs.payment_month) >= 7 
        THEN (cs.payment_month || '-01')::date
        ELSE NULL
      END,
      s.amount,
      CASE WHEN cs.payment_received THEN s.amount ELSE 0 END,
      CASE WHEN cs.payment_received THEN CURRENT_DATE ELSE NULL END,
      NOW()
    FROM contact_services cs
    JOIN contacts c ON cs.contact_id = c.id
    JOIN services s ON cs.service_id = s.id
    WHERE cs.payment_received = true
    ON CONFLICT (subscription_id, payment_month) DO NOTHING;
  END IF;
END $$;

-- Grant permissions
GRANT ALL ON service_subscriptions TO authenticated;
GRANT ALL ON subscription_payments TO authenticated;
