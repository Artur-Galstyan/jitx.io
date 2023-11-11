/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{html,js,svelte,ts}"],
  theme: {
    extend: {},
  },
  plugins: [
    require("@tailwindcss/typography"),
    require("daisyui"),
    require("tailwind-scrollbar"),
  ],
  daisyui: {
    themes: [
      {
        emerald: {
          ...require("daisyui/src/theming/themes")["[data-theme=emerald]"],
          pre: {
            padding: 2,
          },
        },
      },
      {
        dracula: {
          ...require("daisyui/src/theming/themes")["[data-theme=dracula]"],
          pre: {
            padding: 2,
          },
        },
      },
    ],
  },
};
