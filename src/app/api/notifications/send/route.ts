import { NextRequest, NextResponse } from 'next/server'
import Twilio from 'twilio'

const accountSid = process.env.TWILIO_ACCOUNT_SID
const authToken = process.env.TWILIO_AUTH_TOKEN
const twilioPhoneNumber = process.env.TWILIO_PHONE_NUMBER

export async function POST(request: NextRequest) {
  try {
    const { phoneNumber, message } = await request.json()

    // Validate required fields
    if (!phoneNumber || !message) {
      return NextResponse.json(
        { error: 'Phone number and message are required' },
        { status: 400 }
      )
    }

    // Format phone number for WhatsApp (must start with whatsapp:)
    const formattedPhone = phoneNumber.startsWith('whatsapp:')
      ? phoneNumber
      : `whatsapp:${phoneNumber.replace(/[^\d+]/g, '')}`

    // Check if Twilio is configured
    const isTwilioConfigured = accountSid?.startsWith('AC') && authToken && twilioPhoneNumber
    
    if (!isTwilioConfigured) {
      // For development/demo purposes, simulate a successful response
      // In production, this should return an error
      console.log('Twilio not configured. Simulating success.')
      console.log(`Would send WhatsApp to: ${formattedPhone}`)
      console.log(`Message: ${message}`)
      
      // Return success for demo (in production, remove this)
      return NextResponse.json({
        success: true,
        message: 'Notification sent (simulated - Twilio not configured)',
        sid: `DEMO_${Date.now()}`
      })
    }

    // Initialize Twilio client
    const twilioClient = Twilio(accountSid, authToken)

    // Send WhatsApp message via Twilio
    const twilioMessage = await twilioClient.messages.create({
      body: message,
      from: twilioPhoneNumber,
      to: formattedPhone,
    })

    return NextResponse.json({
      success: true,
      message: 'WhatsApp notification sent successfully',
      sid: twilioMessage.sid
    })

  } catch (error: any) {
    console.error('Error sending WhatsApp notification:', error)
    
    return NextResponse.json(
      { 
        error: error.message || 'Failed to send notification',
        details: error.code || 'UNKNOWN_ERROR'
      },
      { status: 500 }
    )
  }
}
