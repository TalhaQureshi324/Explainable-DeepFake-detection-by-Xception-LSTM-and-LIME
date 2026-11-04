/** @type {import('next').NextConfig} */

// The frontend never runs TensorFlow. Inference traffic goes through the
// runtime route handler at /api/inference/[...path] (app/api/inference),
// which resolves INFERENCE_API_URL per request — compatible with Vercel
// service bindings and project env variables, which are runtime-only.
// Local development: set INFERENCE_API_URL in .env.local or use the default
// http://127.0.0.1:8000.
const nextConfig = {};

export default nextConfig;
