import { NextRequest, NextResponse } from 'next/server';
import { timingSafeEqual } from 'node:crypto';
import { createDemoSessionToken, DEMO_SESSION_COOKIE, DEMO_SESSION_MAX_AGE } from '@/lib/demo-session';

function matchesConfiguredSecret(input: string, expected: string | undefined) {
  if (!expected) return false;
  const inputBuffer = Buffer.from(input);
  const expectedBuffer = Buffer.from(expected);
  return inputBuffer.length === expectedBuffer.length && timingSafeEqual(inputBuffer, expectedBuffer);
}

export async function POST(request: NextRequest) {
  const username = process.env.DASHBOARD_DEMO_USERNAME;
  const password = process.env.DASHBOARD_DEMO_PASSWORD;
  if (!username || !password || !process.env.DASHBOARD_DEMO_SESSION_SECRET) {
    return NextResponse.json({ error: 'Demo sign-in is not configured on this server.' }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
  }

  try {
    const body = await request.json();
    const suppliedUsername = typeof body.username === 'string' ? body.username.trim() : '';
    const suppliedPassword = typeof body.password === 'string' ? body.password : '';
    const validUsername = matchesConfiguredSecret(suppliedUsername.toLowerCase(), username.toLowerCase());
    const validPassword = matchesConfiguredSecret(suppliedPassword, password);

    if (!validUsername || !validPassword) {
      return NextResponse.json({ error: 'Invalid demo username or password.' }, { status: 401, headers: { 'Cache-Control': 'no-store' } });
    }

    const response = NextResponse.json({ ok: true }, { headers: { 'Cache-Control': 'no-store' } });
    response.cookies.set(DEMO_SESSION_COOKIE, createDemoSessionToken(), {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/dashboard/demo',
      maxAge: DEMO_SESSION_MAX_AGE,
    });
    return response;
  } catch {
    return NextResponse.json({ error: 'Could not sign in to the demo workspace.' }, { status: 400, headers: { 'Cache-Control': 'no-store' } });
  }
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true }, { headers: { 'Cache-Control': 'no-store' } });
  response.cookies.set(DEMO_SESSION_COOKIE, '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/dashboard/demo',
    maxAge: 0,
  });
  return response;
}
