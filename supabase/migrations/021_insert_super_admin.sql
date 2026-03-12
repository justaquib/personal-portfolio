-- Insert Super Admin role for the main user
-- Run this to set yourself as Super Admin

-- First, find your user_id from auth.users
-- SELECT id, email FROM auth.users WHERE email = 'techaquib@gmail.com';

-- Then insert your role (replace 'YOUR-USER-ID-HERE' with the actual user_id)
INSERT INTO user_roles (user_id, email, role, created_by)
SELECT 
  id,
  'techaquib@gmail.com',
  'super_admin',
  id
FROM auth.users 
WHERE email = 'techaquib@gmail.com'
ON CONFLICT (user_id) DO NOTHING;
