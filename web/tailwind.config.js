/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        bg: '#0a0a0a',
        surface: '#111111',
        surface2: '#181818',
        border: '#2a2a2a',
        'border-bright': '#3a3a3a',
        text: '#e8e8e8',
        'text-dim': '#666666',
        'text-muted': '#444444',
        'accent-green': '#4ADE80',
        'accent-green-dim': '#1a3d27',
        'accent-amber': '#FBBF24',
        'accent-amber-dim': '#3d2e0a',
        'accent-red': '#F87171',
        'accent-red-dim': '#3d1a1a',
        'accent-blue': '#60A5FA',
      },
      borderRadius: {
        none: '0px',
        DEFAULT: '0px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'blink': 'blink 1s step-end infinite',
        'count-up': 'count-up 0.5s ease-out forwards',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
    },
  },
  plugins: [],
};
