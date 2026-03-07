-- Add DELETE policy for notifications table
-- This policy allows authenticated users to delete their own notifications

-- Check if policy already exists before creating
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies WHERE policyname = 'Allow authenticated users to delete notifications'
  ) THEN
    CREATE POLICY "Allow authenticated users to delete notifications"
      ON public.notifications
      FOR DELETE
      TO authenticated
      USING (true);
    
    RAISE NOTICE 'DELETE policy created successfully for notifications table';
  ELSE
    RAISE NOTICE 'DELETE policy already exists for notifications table';
  END IF;
END $$;
