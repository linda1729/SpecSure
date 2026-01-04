import path from 'path';
import fs from 'fs';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3001,
        host: '0.0.0.0',
        fs: {
          // 允许服务父目录中的 mockfrontend 文件
          allow: ['..'],
        },
      },
      plugins: [
        react(),
        // 自定义插件：处理 /app 路由到 mockfrontend
        {
          name: 'serve-mockfrontend',
          configureServer(server) {
            server.middlewares.use('/app', (req, res, next) => {
              const url = req.url || '/';
              const filePath = path.resolve(__dirname, '../mockfrontend', url === '/' ? 'index.html' : url.slice(1));
              
              if (fs.existsSync(filePath)) {
                const ext = path.extname(filePath);
                const contentTypes: Record<string, string> = {
                  '.html': 'text/html; charset=utf-8',
                  '.css': 'text/css; charset=utf-8',
                  '.js': 'application/javascript; charset=utf-8',
                };
                res.setHeader('Content-Type', contentTypes[ext] || 'text/plain');
                res.end(fs.readFileSync(filePath));
              } else {
                next();
              }
            });
          }
        }
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
    };
});
