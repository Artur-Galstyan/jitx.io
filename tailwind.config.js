/** @type {import('tailwindcss').Config} */
export default {
    content: ["./src/**/*.{html,js,svelte,ts}"],
    theme: {
        extend: {}
    },
    plugins: [require("@tailwindcss/typography"), require("daisyui")],
    daisyui: {
        themes: [
            {
                emerald: {
                    ...require("daisyui/src/theming/themes")[
                        "[data-theme=emerald]"
                    ],
                    pre: {
                        padding: 2
                    }
                }
            }
        ]
    }
};
