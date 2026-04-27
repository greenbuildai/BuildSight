import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('BROWSER CONSOLE:', msg.type(), msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));
  
  await page.goto('http://localhost:5173');
  await page.waitForTimeout(5000);
  
  // Click on the brain tab to open BuildSight Brain
  await page.evaluate(() => {
    const tabs = Array.from(document.querySelectorAll('button, a, div'));
    const brainTab = tabs.find(t => t.textContent && t.textContent.includes('BuildSight Brain'));
    if (brainTab) {
      (brainTab as HTMLElement).click();
    } else {
        const brainBtn = document.querySelector('.gm-topbar__nav-item:nth-child(2)');
        if (brainBtn) (brainBtn as HTMLElement).click();
    }
  });
  
  await page.waitForTimeout(5000);
  await browser.close();
})();