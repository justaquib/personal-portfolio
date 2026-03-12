-- Add avatar_url column to user_roles table
ALTER TABLE user_roles ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- Create storage bucket for avatars
INSERT INTO storage.buckets (id, name, public, created_at, updated_at)
VALUES ('avatars', 'avatars', true, NOW(), NOW())
ON CONFLICT (id) DO NOTHING;

-- Set up storage policies for avatars
-- Note: Run these manually or drop existing policies first if they cause issues
