import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Add the allowedDevOrigins property here
  allowedDevOrigins: [
    'http://20.0.0.12:3000'
  ],
  /* other config options here */
};

export default nextConfig;