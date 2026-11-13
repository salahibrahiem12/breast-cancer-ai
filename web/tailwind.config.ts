import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#D63384",
          dark: "#B02368",
          light: "#E05299",
        },
        pink: {
          primary: "#D63384",
          soft: "#FCE7F3",
          rose: "#F9A8D4",
          bg: "#FFF7FB",
          accent: "#BE185D",
          deep: "#9D174D",
        },
        slate: {
          text: "#334155",
        },
        ink: "#334155",
        muted: "#64748B",
        line: "#FBCFE8",
      },
      boxShadow: {
        card: "0 10px 30px -5px rgba(214, 51, 132, 0.08), 0 4px 6px -2px rgba(214, 51, 132, 0.04)",
        glow: "0 0 20px rgba(214, 51, 132, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;
