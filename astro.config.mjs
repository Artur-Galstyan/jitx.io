import tailwind from "@astrojs/tailwind";
import vercel from "@astrojs/vercel/serverless";
import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkToc from "remark-toc";

// https://astro.build/config
export default defineConfig({
  integrations: [tailwind({ applyBaseStyles: false })],
  markdown: {
    shikiConfig: { theme: "catppuccin-macchiato", wrap: false },
    remarkPlugins: [remarkMath, remarkToc],
    rehypePlugins: [rehypeKatex],
  },
  output: "server",
  adapter: vercel({
    webAnalytics: {
      enabled: true,
    },
  }),
});
