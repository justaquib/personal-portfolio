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
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create unique index for visitor per day (this replaces the UNIQUE constraint)
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_visits_unique_daily
  ON public.analytics_visits(visitor_id, visit_date);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_analytics_visits_visitor_id
  ON public.analytics_visits(visitor_id);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_page
  ON public.analytics_visits(page);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_visit_date
  ON public.analytics_visits(visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_timestamp
  ON public.analytics_visits(timestamp DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE public.analytics_visits ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
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

-- Add comments for documentation
COMMENT ON TABLE public.analytics_visits IS 'Stores analytics data for tracking unique user visits per day';
COMMENT ON COLUMN public.analytics_visits.visitor_id IS 'Hashed identifier for unique visitors';
COMMENT ON COLUMN public.analytics_visits.page IS 'Page path that was visited';
COMMENT ON COLUMN public.analytics_visits.visit_date IS 'Date of the visit (for daily aggregation)';
COMMENT ON COLUMN public.analytics_visits.timestamp IS 'Exact timestamp of the visit';