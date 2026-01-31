import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  allowedDevOrigins: [
    '20.0.0.12',
    'http://20.0.0.12',
    'http://20.0.0.12:3000'
  ]
};

export default nextConfig;