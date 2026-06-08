import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lint is relaxed for the demo skeleton so a config quirk never blocks a
  // build; run `npm run lint` separately if you want to tighten it.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
