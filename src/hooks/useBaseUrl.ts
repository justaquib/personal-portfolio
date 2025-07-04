export const useBaseUrl = () => {
  if (typeof window !== "undefined") {
    // Client-side
    return window.location.origin;
  }

  // Server-side (Next.js API routes, SSR, etc.)
  return process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : "http://localhost:3000";
};
