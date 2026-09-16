import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { transformWithOxc } from 'vite';
import { defineConfig } from 'vitest/config';

const jsxInJs = {
  name: 'admitai-jsx-in-js',
  enforce: 'pre',
  async transform(code, id) {
    if (!/\/src\/.*\.js$/.test(id)) return null;
    return transformWithOxc(code, id, {
      lang: 'jsx',
      jsx: { runtime: 'automatic' }
    });
  }
};

export default defineConfig({
  plugins: [jsxInJs, react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.js']
  }
});
