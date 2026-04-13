import { chromium } from 'playwright';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

(async () => {
  const server = spawn('npm', ['run', 'dev'], { cwd: __dirname, shell: true });
  server.stdout.on('data', (data) => console.log(`vite: ${data}`));
  server.stderr.on('data', (data) => console.error(`vite err: ${data}`));
  
  // Wait for server to start
  await new Promise(r => setTimeout(r, 4000));

  console.log('Launching browser...');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.type(), msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  
  await page.goto('http://localhost:5173');
  
  console.log('Clicking BuildSight Brain tab...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('.nav-link'));
    const brainBtn = btns.find(b => b.textContent.includes('BuildSight Brain'));
    if (brainBtn) brainBtn.click();
  });

  await new Promise(r => setTimeout(r, 3000));
  
  await browser.close();
  server.kill();
  process.exit(0);
})();
