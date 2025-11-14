import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "https://api.phimnhalam.website",
        changeOrigin: true,
        secure: true,
      },
      "/movies": {
        target: "https://api.phimnhalam.website",
        changeOrigin: true,
        secure: true,
      },
      "/ai": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/ai/, ""),
      },
    },
  },
});
