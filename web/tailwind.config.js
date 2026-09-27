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
        sans: ['var(--font-sans)', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'JetBrains Mono', 'Consolas', 'monospace'],
        serif: ['var(--font-serif)', 'Georgia', 'serif'],
      },
      colors: {
        ink: {
          950: '#07080A',
          900: '#0B0C0F',
          850: '#101216',
          800: '#15171C',
          700: '#1C1F25',
          600: '#262A32',
        },
        line: 'rgba(255,255,255,0.07)',
        'line-2': 'rgba(255,255,255,0.12)',
        fg: '#EDEDEF',
        'fg-2': '#A6A6B0',
        'fg-3': '#767680',
        'fg-4': '#52525B',
        mint: '#4ADE80',
        sky: '#60A5FA',
        amber: '#FBBF24',
        violet: '#A78BFA',
        rose: '#FB7185',
        cyan: '#22D3EE',
      },
      borderRadius: {
        xl: '14px',
        '2xl': '18px',
        '3xl': '24px',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        blink: 'blink 1s step-end infinite',
      },
    },
  },
  plugins: [],
};
