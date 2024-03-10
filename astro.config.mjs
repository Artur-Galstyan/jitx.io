import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import tailwind from "@astrojs/tailwind";
import remarkToc from "remark-toc";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  integrations: [mdx(), tailwind({ applyBaseStyles: false })],
  markdown: {
    shikiConfig: {
      theme: "catppuccin-macchiato",
      wrap: true,
    },
    remarkPlugins: [remarkMath, remarkToc],
    rehypePlugins: [rehypeKatex],
  },
});
