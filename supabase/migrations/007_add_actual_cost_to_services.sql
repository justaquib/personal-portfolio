-- Migration: Add actual_cost field to services table
-- This allows tracking the cost to calculate earnings

ALTER TABLE services 
ADD COLUMN IF NOT EXISTS actual_cost NUMERIC DEFAULT 0;
