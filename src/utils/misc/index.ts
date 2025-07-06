export const getBaseUrl = (): string => {
  if (typeof window !== "undefined") {
    return window.location.origin;
  }

  return process.env.VERCEL_URL
    ? `https://${process.env.VERCEL_URL}`
    : "http://localhost:3000";
};

export function makeGetHref<T extends { href: string }>(basePath: string) {
  return (item: T) => `/${basePath}/${item.href}`;
}
