import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'
import Twilio from 'twilio'

const accountSid = process.env.TWILIO_ACCOUNT_SID
const authToken = process.env.TWILIO_AUTH_TOKEN
const twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER
const ADMIN_PHONE = process.env.ADMIN_PHONE || 'whatsapp:+1234567890' // Replace with actual admin phone

export async function GET(request: NextRequest) {
  try {
    console.log('Cron job keep-alive started at:', new Date().toISOString())

    const supabase = await createClient()

    // Perform database access to keep it active
    console.log('Checking database connection...')

    // Fetch some data from database to ensure connection
    const { data: contacts, error: contactsError } = await supabase
      .from('contacts')
      .select('id, name, email')
      .limit(1)

    if (contactsError) {
      console.error('Database error:', contactsError)
      return NextResponse.json(
        { error: 'Database connection failed', details: contactsError.message },
        { status: 500 }
      )
    }

    console.log(`Database active - found ${contacts?.length || 0} contacts`)

    // Send notification (WhatsApp message)
    console.log('Sending activity notification...')

    const isTwilioConfigured = accountSid?.startsWith('AC') && authToken && twilioPhoneNumber

    if (isTwilioConfigured) {
      const twilioClient = Twilio(accountSid, authToken)

      const message = `🤖 Web-App Keep-Alive Cron Job\n⏰ Time: ${new Date().toLocaleString()}\n📊 Database Status: Active\n👥 Contacts Count: ${contacts?.length || 0}\n✅ Activity logged successfully`

      try {
        const twilioMessage = await twilioClient.messages.create({
          body: message,
          from: twilioPhoneNumber,
          to: ADMIN_PHONE,
        })

        console.log('WhatsApp notification sent:', twilioMessage.sid)
      } catch (twilioError: any) {
        console.error('Failed to send WhatsApp notification:', twilioError.message)
        // Don't fail the cron job if notification fails
      }
    } else {
      console.log('Twilio not configured - skipping WhatsApp notification')
      console.log('Activity notification would be sent to:', ADMIN_PHONE)
    }

    // Log activity to database (optional - create a logs table if needed)
    // For now, just log to console

    console.log('Cron job keep-alive completed successfully')

    return NextResponse.json({
      success: true,
      message: 'Database keep-alive cron job completed',
      timestamp: new Date().toISOString(),
      databaseStatus: 'active',
      contactsCount: contacts?.length || 0,
      notificationSent: isTwilioConfigured ? 'yes' : 'simulated'
    })

  } catch (error: any) {
    console.error('Cron job keep-alive failed:', error)

    return NextResponse.json(
      {
        error: 'Cron job failed',
        details: error.message,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    )
  }
}

// Also support POST for flexibility
export async function POST(request: NextRequest) {
  return GET(request)
}