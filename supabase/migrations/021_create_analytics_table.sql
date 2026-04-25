-- Supabase Database Migration
-- Create analytics table for tracking unique user visits

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Alter existing analytics_visits table to add new columns
ALTER TABLE public.analytics_visits
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS country VARCHAR(100),
ADD COLUMN IF NOT EXISTS city VARCHAR(100),
ADD COLUMN IF NOT EXISTS region VARCHAR(100),
ADD COLUMN IF NOT EXISTS latitude DECIMAL(10, 8),
ADD COLUMN IF NOT EXISTS longitude DECIMAL(11, 8),
ADD COLUMN IF NOT EXISTS device_type VARCHAR(50), -- 'desktop', 'mobile', 'tablet'
ADD COLUMN IF NOT EXISTS browser VARCHAR(100),
ADD COLUMN IF NOT EXISTS os VARCHAR(100),
ADD COLUMN IF NOT EXISTS session_start TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS time_on_page INTEGER, -- Time spent on this page in seconds
ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS blocked_reason TEXT,
ADD COLUMN IF NOT EXISTS blocked_by UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS blocked_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS visitor_type VARCHAR(20) DEFAULT 'anonymous', -- 'anonymous' or 'authenticated'
ADD COLUMN IF NOT EXISTS user_info JSONB; -- Store user info for authenticated visitors

-- Update existing records to have session_id (generate based on visitor_id and date)
UPDATE public.analytics_visits
SET session_id = CONCAT(visitor_id, '_', visit_date::text)
WHERE session_id IS NULL;

-- Make session_id NOT NULL after populating existing records
ALTER TABLE public.analytics_visits
ALTER COLUMN session_id SET NOT NULL;

-- Create blocked_ips table for IP-based blocking (if not exists)
CREATE TABLE IF NOT EXISTS public.blocked_ips (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  ip_address INET NOT NULL UNIQUE,
  reason TEXT,
  blocked_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create blocked_visitors table for visitor ID-based blocking (if not exists)
CREATE TABLE IF NOT EXISTS public.blocked_visitors (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  visitor_id VARCHAR(64) NOT NULL UNIQUE,
  reason TEXT,
  blocked_by UUID REFERENCES auth.users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create sessions table for aggregated session data (if not exists)
CREATE TABLE IF NOT EXISTS public.analytics_sessions (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL UNIQUE,
  visitor_id VARCHAR(64) NOT NULL,
  ip_address INET,
  user_agent TEXT,
  -- Location data (copied from first visit in session)
  country VARCHAR(100),
  city VARCHAR(100),
  region VARCHAR(100),
  device_type VARCHAR(50),
  browser VARCHAR(100),
  os VARCHAR(100),
  -- Session metrics
  start_time TIMESTAMPTZ NOT NULL,
  end_time TIMESTAMPTZ,
  duration_seconds INTEGER, -- Total session duration
  page_views INTEGER DEFAULT 1,
  pages_visited TEXT[], -- Array of pages visited
  referrer TEXT,
  -- Status
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- All index creation moved to the very end after all table modifications

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

-- Enable RLS for analytics_sessions
ALTER TABLE public.analytics_sessions ENABLE ROW LEVEL SECURITY;

-- RLS policies for analytics_sessions
DO $$
BEGIN
  -- Allow authenticated users to read session analytics
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to read sessions'
  ) THEN
    CREATE POLICY "Allow authenticated users to read sessions"
      ON public.analytics_sessions
      FOR SELECT
      TO authenticated
      USING (true);
  END IF;

  -- Allow system to insert/update sessions
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow system to manage sessions'
  ) THEN
    CREATE POLICY "Allow system to manage sessions"
      ON public.analytics_sessions
      FOR ALL
      TO authenticated
      USING (true);
  END IF;
END
$$;

-- Function to manage analytics sessions
CREATE OR REPLACE FUNCTION manage_analytics_session(
  p_visitor_id VARCHAR(64),
  p_session_id VARCHAR(64),
  p_page VARCHAR(255),
  p_ip_address INET,
  p_user_agent TEXT,
  p_country VARCHAR(100),
  p_city VARCHAR(100),
  p_region VARCHAR(100),
  p_device_type VARCHAR(50),
  p_browser VARCHAR(100),
  p_os VARCHAR(100),
  p_referrer TEXT
) RETURNS VOID AS $$
DECLARE
  v_session_exists BOOLEAN;
  v_last_visit_time TIMESTAMPTZ;
  v_session_duration INTEGER;
BEGIN
  -- Check if session already exists
  SELECT EXISTS(
    SELECT 1 FROM public.analytics_sessions
    WHERE session_id = p_session_id
  ) INTO v_session_exists;

  IF NOT v_session_exists THEN
    -- Create new session
    INSERT INTO public.analytics_sessions (
      session_id, visitor_id, ip_address, user_agent,
      country, city, region, device_type, browser, os,
      start_time, pages_visited, referrer
    ) VALUES (
      p_session_id, p_visitor_id, p_ip_address, p_user_agent,
      p_country, p_city, p_region, p_device_type, p_browser, p_os,
      NOW(), ARRAY[p_page], p_referrer
    );
  ELSE
    -- Update existing session
    -- Get the time of last visit in this session
    SELECT MAX(timestamp)
    INTO v_last_visit_time
    FROM public.analytics_visits
    WHERE session_id = p_session_id;

    -- Calculate session duration (in seconds)
    IF v_last_visit_time IS NOT NULL THEN
      v_session_duration := EXTRACT(EPOCH FROM (NOW() - v_last_visit_time))::INTEGER;
    ELSE
      v_session_duration := 0;
    END IF;

    -- Update session with new page and duration
    UPDATE public.analytics_sessions
    SET
      end_time = NOW(),
      duration_seconds = v_session_duration,
      page_views = page_views + 1,
      pages_visited = array_append(pages_visited, p_page),
      updated_at = NOW()
    WHERE session_id = p_session_id;
  END IF;
END;
$$ LANGUAGE plpgsql;

-- Create all indexes after all table modifications are complete
CREATE INDEX IF NOT EXISTS idx_analytics_visits_visitor_id
  ON public.analytics_visits(visitor_id);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_page
  ON public.analytics_visits(page);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_visit_date
  ON public.analytics_visits(visit_date DESC);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_ip_address
  ON public.analytics_visits(ip_address);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_country
  ON public.analytics_visits(country);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_device_type
  ON public.analytics_visits(device_type);

CREATE INDEX IF NOT EXISTS idx_analytics_visits_is_blocked
  ON public.analytics_visits(is_blocked);

-- Remove the old unique constraint and create new indexes
DROP INDEX IF EXISTS idx_analytics_visits_unique_daily;
CREATE INDEX IF NOT EXISTS idx_analytics_visits_session_id
  ON public.analytics_visits(session_id);
CREATE INDEX IF NOT EXISTS idx_analytics_visits_session_start
  ON public.analytics_visits(session_start);

-- Indexes for sessions table
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_visitor_id
  ON public.analytics_sessions(visitor_id);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_start_time
  ON public.analytics_sessions(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_sessions_is_active
  ON public.analytics_sessions(is_active);

-- Add comments for documentation
COMMENT ON TABLE public.analytics_visits IS 'Stores detailed analytics data for each page visit with location and device info';
COMMENT ON COLUMN public.analytics_visits.visitor_id IS 'Hashed identifier for unique visitors';
COMMENT ON COLUMN public.analytics_visits.session_id IS 'Unique identifier for each browsing session';
COMMENT ON COLUMN public.analytics_visits.page IS 'Page path that was visited';
COMMENT ON COLUMN public.analytics_visits.visit_date IS 'Date of the visit (for daily aggregation)';
COMMENT ON COLUMN public.analytics_visits.timestamp IS 'Exact timestamp of the page visit';
COMMENT ON COLUMN public.analytics_visits.session_start IS 'Timestamp when the session started';
COMMENT ON COLUMN public.analytics_visits.time_on_page IS 'Time spent on this page in seconds';
COMMENT ON COLUMN public.analytics_visits.country IS 'User country based on IP geolocation';
COMMENT ON COLUMN public.analytics_visits.city IS 'User city based on IP geolocation';
COMMENT ON COLUMN public.analytics_visits.device_type IS 'Device type: desktop, mobile, tablet';
COMMENT ON COLUMN public.analytics_visits.browser IS 'Browser name and version';
COMMENT ON COLUMN public.analytics_visits.os IS 'Operating system name and version';
COMMENT ON COLUMN public.analytics_visits.is_blocked IS 'Whether this visitor is blocked from accessing the site';
COMMENT ON TABLE public.analytics_sessions IS 'Aggregated session data with duration and page flow information';
COMMENT ON COLUMN public.analytics_sessions.session_id IS 'Unique session identifier';
COMMENT ON COLUMN public.analytics_sessions.duration_seconds IS 'Total session duration in seconds';
COMMENT ON COLUMN public.analytics_sessions.pages_visited IS 'Array of pages visited during the session';
COMMENT ON COLUMN public.analytics_sessions.page_views IS 'Total number of page views in the session';
COMMENT ON TABLE public.blocked_ips IS 'Stores IP addresses that are blocked from accessing the site';
COMMENT ON TABLE public.blocked_visitors IS 'Stores visitor IDs that are blocked from accessing the site';