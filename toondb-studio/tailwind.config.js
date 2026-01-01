/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Semantic backgrounds
                background: {
                    app: 'var(--background-app)',
                    default: 'var(--background-default)',
                    card: 'var(--background-card)',
                    muted: 'var(--background-muted)',
                    medium: 'var(--background-medium)',
                    strong: 'var(--background-strong)',
                    inverse: 'var(--background-inverse)',
                    accent: 'var(--background-accent)',
                },
                // Semantic borders
                border: {
                    default: 'var(--border-default)',
                    input: 'var(--border-input)',
                    strong: 'var(--border-strong)',
                    accent: 'var(--border-accent)',
                },
                // Semantic text
                text: {
                    default: 'var(--text-default)',
                    muted: 'var(--text-muted)',
                    inverse: 'var(--text-inverse)',
                    accent: 'var(--text-accent)',
                    'on-accent': 'var(--text-on-accent)',
                },
                // Brand accents
                teal: {
                    DEFAULT: 'var(--color-block-teal)',
                },
                orange: {
                    DEFAULT: 'var(--color-block-orange)',
                }
            }
        },
    },
    plugins: [],
}
