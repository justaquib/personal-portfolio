-- Supabase Database Migration
-- Create analytics table for tracking unique user visits

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create analytics_visits table if not exists
CREATE TABLE IF NOT EXISTS public.analytics_visits (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  visitor_id VARCHAR(64) NOT NULL,
  page VARCHAR(255) NOT NULL DEFAULT '/',
  user_agent TEXT,
  ip_address INET,
  referrer TEXT,
  visit_date DATE NOT NULL DEFAULT CURRENT_DATE,
  timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  -- Location and device tracking
  country VARCHAR(100),
  city VARCHAR(100),
  region VARCHAR(100),
  latitude DECIMAL(10, 8),
  longitude DECIMAL(11, 8),
  device_type VARCHAR(50), -- 'desktop', 'mobile', 'tablet'
  browser VARCHAR(100),
  os VARCHAR(100),
  -- Admin controls
  is_blocked BOOLEAN DEFAULT FALSE,
  blocked_reason TEXT,
  blocked_by UUID REFERENCES auth.users(id),
  blocked_at TIMESTAMPTZ
);

-- Create unique index for visitor per day (this replaces the UNIQUE constraint)
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_visits_unique_daily
  ON public.analytics_visits(visitor_id, visit_date);

-- Create blocked_ips table for IP-based blocking
CREATE TABLE IF NOT EXISTS public.blocked_ips (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  ip_address INET NOT NULL UNIQUE,
  reason TEXT,
  blocked_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create blocked_visitors table for visitor ID-based blocking
CREATE TABLE IF NOT EXISTS public.blocked_visitors (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  visitor_id VARCHAR(64) NOT NULL UNIQUE,
  reason TEXT,
  blocked_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_analytics_visits_visitor_id
  ON public.analytics_visits(visitor_id);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_page
  ON public.analytics_visits(page);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_visit_date
  ON public.analytics_visits(visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_timestamp
  ON public.analytics_visits(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_ip_address
  ON public.analytics_visits(ip_address);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_country
  ON public.analytics_visits(country);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_device_type
  ON public.analytics_visits(device_type);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_is_blocked
  ON public.analytics_visits(is_blocked);

-- Enable Row Level Security (RLS)
ALTER TABLE public.analytics_visits ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for analytics_visits
DO $$
BEGIN
  -- Allow authenticated users to read analytics data
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to read analytics'
  ) THEN
    CREATE POLICY "Allow authenticated users to read analytics"
      ON public.analytics_visits
      FOR SELECT
      TO authenticated
      USING (true);
  END IF;
END
$$;

DO $$
BEGIN
  -- Allow authenticated users to insert analytics data
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to insert analytics'
  ) THEN
    CREATE POLICY "Allow authenticated users to insert analytics"
      ON public.analytics_visits
      FOR INSERT
      TO authenticated
      WITH CHECK (true);
  END IF;
END
$$;

DO $$
BEGIN
  -- Allow admins to update analytics (for blocking)
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow admins to update analytics'
  ) THEN
    CREATE POLICY "Allow admins to update analytics"
      ON public.analytics_visits
      FOR UPDATE
      TO authenticated
      USING (
        EXISTS (
          SELECT 1 FROM public.user_roles
          WHERE user_roles.user_id = auth.uid()
          AND user_roles.role IN ('admin', 'super_admin')
        )
      );
  END IF;
END
$$;

-- Enable RLS for blocked_ips and blocked_visitors
ALTER TABLE public.blocked_ips ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.blocked_visitors ENABLE ROW LEVEL SECURITY;

-- RLS policies for blocked_ips
DO $$
BEGIN
  -- Allow admins to manage blocked IPs
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow admins to manage blocked IPs'
  ) THEN
    CREATE POLICY "Allow admins to manage blocked IPs"
      ON public.blocked_ips
      FOR ALL
      TO authenticated
      USING (
        EXISTS (
          SELECT 1 FROM public.user_roles
          WHERE user_roles.user_id = auth.uid()
          AND user_roles.role IN ('admin', 'super_admin')
        )
      );
  END IF;
END
$$;

-- RLS policies for blocked_visitors
DO $$
BEGIN
  -- Allow admins to manage blocked visitors
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow admins to manage blocked visitors'
  ) THEN
    CREATE POLICY "Allow admins to manage blocked visitors"
      ON public.blocked_visitors
      FOR ALL
      TO authenticated
      USING (
        EXISTS (
          SELECT 1 FROM public.user_roles
          WHERE user_roles.user_id = auth.uid()
          AND user_roles.role IN ('admin', 'super_admin')
        )
      );
  END IF;
END
$$;

-- Add comments for documentation
COMMENT ON TABLE public.analytics_visits IS 'Stores analytics data for tracking unique user visits per day with location and device info';
COMMENT ON COLUMN public.analytics_visits.visitor_id IS 'Hashed identifier for unique visitors';
COMMENT ON COLUMN public.analytics_visits.page IS 'Page path that was visited';
COMMENT ON COLUMN public.analytics_visits.visit_date IS 'Date of the visit (for daily aggregation)';
COMMENT ON COLUMN public.analytics_visits.timestamp IS 'Exact timestamp of the visit';
COMMENT ON COLUMN public.analytics_visits.country IS 'User country based on IP geolocation';
COMMENT ON COLUMN public.analytics_visits.city IS 'User city based on IP geolocation';
COMMENT ON COLUMN public.analytics_visits.device_type IS 'Device type: desktop, mobile, tablet';
COMMENT ON COLUMN public.analytics_visits.browser IS 'Browser name and version';
COMMENT ON COLUMN public.analytics_visits.os IS 'Operating system name and version';
COMMENT ON COLUMN public.analytics_visits.is_blocked IS 'Whether this visitor is blocked from accessing the site';
COMMENT ON TABLE public.blocked_ips IS 'Stores IP addresses that are blocked from accessing the site';
COMMENT ON TABLE public.blocked_visitors IS 'Stores visitor IDs that are blocked from accessing the site';