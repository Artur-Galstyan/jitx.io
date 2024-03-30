/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            "blockquote p:first-of-type::before": false,
            "blockquote p:first-of-type::after": false,
          },
        },
      },
      fontFamily: {
        sans: ['"Fira Code"', "ui-sans-serif", "system-ui"],
      },
    },
  },
  plugins: [require("@tailwindcss/typography"), require("daisyui")],
  daisyui: {
    themes: ["lofi", "dark", "cupcake"],
  },
};
