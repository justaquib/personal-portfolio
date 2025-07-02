import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/(.*)",
        has: [
          {
            type: "header",
            key: "x-forwarded-proto",
            value: "http",
          },
        ],
        destination: "http://www.justaquib.com/:path*",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
