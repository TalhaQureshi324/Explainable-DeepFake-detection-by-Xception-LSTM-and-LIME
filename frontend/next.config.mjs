/** @type {import('next').NextConfig} */

// The frontend never runs TensorFlow. It talks to the FastAPI inference
// backend through a same-origin proxy (`/api/inference/:path*` -> backend),
// which removes CORS from the picture entirely.
//
// NOTE: Next.js bakes rewrites into the build manifest, so INFERENCE_API_URL
// must be set at BUILD time (`INFERENCE_API_URL=https://... npm run build`).
// On Vercel, set it as a project environment variable — same mechanism.
const INFERENCE_API_URL = process.env.INFERENCE_API_URL || "http://127.0.0.1:8000";

const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/inference/:path*",
        destination: `${INFERENCE_API_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
