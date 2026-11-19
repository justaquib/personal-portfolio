import { createHmac, timingSafeEqual } from 'node:crypto';

export const DEMO_SESSION_COOKIE = 'demo_dashboard_session';
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 4;

function getSessionSecret() {
  const secret = process.env.DASHBOARD_DEMO_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error('DASHBOARD_DEMO_SESSION_SECRET must be set to a random value of at least 32 characters.');
  }
  return secret;
}

export function createDemoSessionToken() {
  const now = Math.floor(Date.now() / 1000);
  const payload = Buffer.from(JSON.stringify({ sub: 'dashboard-demo', iat: now, exp: now + SESSION_MAX_AGE_SECONDS })).toString('base64url');
  const signature = createHmac('sha256', getSessionSecret()).update(payload).digest('base64url');
  return `${payload}.${signature}`;
}

export function verifyDemoSessionToken(token: string | undefined | null) {
  if (!token) return false;
  try {
    const [payload, signature, extra] = token.split('.');
    if (!payload || !signature || extra) return false;
    const expected = createHmac('sha256', getSessionSecret()).update(payload).digest();
    const received = Buffer.from(signature, 'base64url');
    if (received.length !== expected.length || !timingSafeEqual(received, expected)) return false;
    const session = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8')) as { sub?: string; exp?: number };
    return session.sub === 'dashboard-demo' && typeof session.exp === 'number' && session.exp > Date.now() / 1000;
  } catch {
    return false;
  }
}

export const DEMO_SESSION_MAX_AGE = SESSION_MAX_AGE_SECONDS;
