-- Create user_roles table for role-based access control
CREATE TABLE IF NOT EXISTS user_roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  email VARCHAR(255) NOT NULL,
  role VARCHAR(50) NOT NULL DEFAULT 'developer',
  created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_created_by ON user_roles(created_by);

-- Enable RLS
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;

-- Create policies for user_roles table

-- Anyone can read roles (needed for UI to show role info)
CREATE POLICY "Anyone can view user roles" 
  ON user_roles FOR SELECT 
  USING (true);

-- Only admins (created_by users) can insert new roles
CREATE POLICY "Users can insert roles for users they created" 
  ON user_roles FOR INSERT 
  WITH CHECK (created_by = auth.uid());

-- Only the user themselves or their creator can update
CREATE POLICY "Users can update their own role" 
  ON user_roles FOR UPDATE 
  USING (user_id = auth.uid() OR created_by = auth.uid());

-- Only the user themselves or their creator can delete
CREATE POLICY "Users can delete their own role" 
  ON user_roles FOR DELETE 
  USING (user_id = auth.uid() OR created_by = auth.uid());