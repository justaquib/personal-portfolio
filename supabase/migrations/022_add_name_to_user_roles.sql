-- Add name column to user_roles table
ALTER TABLE user_roles ADD COLUMN IF NOT EXISTS name VARCHAR(255);
