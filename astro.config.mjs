import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import remarkToc from "remark-toc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
export default defineConfig({
  integrations: [
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  markdown: {
    shikiConfig: {
      theme: "catppuccin-macchiato",
      wrap: false,
    },
    remarkPlugins: [remarkMath, remarkToc],
    rehypePlugins: [rehypeKatex],
  },
});
