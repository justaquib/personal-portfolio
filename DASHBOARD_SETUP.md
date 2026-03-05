# Payment Notification Dashboard - Setup Guide

This document explains how to set up the full-stack web application with email/password authentication and WhatsApp payment notifications.

## Features

- ✅ Email/password sign-in and Google OAuth using Supabase Auth
- ✅ Session persistence until explicit logout
- ✅ Protected dashboard route
- ✅ WhatsApp payment notifications via Twilio
- ✅ Contact management with name, phone, and active status
- ✅ Message templates for quick sending
- ✅ Notification history stored in Supabase PostgreSQL
- ✅ Error handling, loading states, and success feedback

## Setup Steps

### 1. Configure Supabase

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Create a new project or select an existing one
3. Navigate to **Settings** → **API**
4. Copy your **Project URL** and **anon public** key

### 2. Configure Environment Variables

Edit `.env.local` and replace the placeholder values:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Twilio Configuration (for WhatsApp)
TWILIO_ACCOUNT_SID=your-account-sid
TWILIO_AUTH_TOKEN=your-auth-token
TWILIO_PHONE_NUMBER=whatsapp:+14155238886
```

### 3. Enable Email Authentication in Supabase

1. Go to **Authentication** → **Providers** in Supabase
2. Ensure **Email** provider is enabled (it is by default)
3. Configure settings as needed

#### Disable Email Confirmation (For Testing)

If you want users to sign up without email verification:
1. Go to **Authentication** → **Settings** in Supabase
2. Find **Email** section
3. Disable **Confirm email**
4. Save changes

> ⚠️ **Important**: This is only recommended for development/testing. For production, keep email confirmation enabled.

### 4. Create Database Table

Run the SQL migrations in Supabase SQL Editor:

1. Go to **SQL Editor** in Supabase
2. Copy and execute the contents of `supabase/migrations/001_create_notifications_table.sql`
3. Then execute `supabase/migrations/002_add_contacts_and_templates.sql`

This creates:
- `notifications` table with columns: id, phone_number, message, timestamp, status
- `contacts` table with columns: id, user_id, name, phone_number, is_active
- `message_templates` table with columns: id, user_id, name, content
- Row Level Security (RLS) policies for user data isolation
- Indexes for better query performance

### 5. Configure Twilio for WhatsApp (Optional)

1. Create a [Twilio account](https://www.twilio.com/)
2. Get a Twilio phone number with WhatsApp capability
3. Verify your WhatsApp sender in Twilio
4. Update the `TWILIO_PHONE_NUMBER` environment variable

## Running the Application

```bash
cd web-app
npm run dev
```

Visit:
- Login page: http://localhost:3000/login
- Dashboard: http://localhost:3000/dashboard

## Application Flow

1. **Sign Up**: Create an account with email and password
2. **Login**: Sign in with your credentials
3. **Dashboard**: Protected route only accessible to authenticated users
4. **Send Notification**: Enter phone number and message, click send
5. **History**: All notifications are stored and displayed in the dashboard

## Authentication Options

### Current: Email/Password Sign-in
The application currently uses email and password for authentication:
- Sign up with any valid email address
- Password must be at least 6 characters
- Sessions persist until explicit logout

### Future: Google OAuth (Available for Re-enabling)
The Google OAuth code is preserved in the codebase. To re-enable it:
1. Enable Google provider in Supabase Authentication settings
2. Uncomment the Google button code in `src/app/login/page.tsx`
3. Configure Google OAuth credentials in Supabase

## API Routes

- `POST /api/notifications/send` - Send WhatsApp notification

## Project Structure

```
web-app/
├── src/
│   ├── app/
│   │   ├── dashboard/         # Protected dashboard page
│   │   ├── login/            # Login page with email/password
│   │   └── api/
│   │       └── notifications/ # WhatsApp notification API
│   ├── context/
│   │   └── AuthContext.tsx   # Authentication context
│   ├── lib/
│   │   └── supabase/         # Supabase client utilities
│   └── middleware.ts          # Auth middleware
├── supabase/
│   └── migrations/           # Database migrations
└── .env.local               # Environment variables
```

## Demo Mode

When Twilio is not configured, the application runs in demo mode:
- Notifications appear successful in the UI
- Records are still saved to the database
- Check server logs for "Would send WhatsApp to:" messages
