-- Remove avatar_url column from user_roles table
ALTER TABLE user_roles DROP COLUMN IF EXISTS avatar_url;
ALTER TABLE user_roles DROP COLUMN IF EXISTS name;
