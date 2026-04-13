import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  
  await page.setContent(`
    <html><body>
      <iframe src="https://my.spline.design/9df9601f-9999-4c76-b44f-8328b500f7b8/" width="800" height="600"></iframe>
    </body></html>
  `);
  
  await page.waitForTimeout(3000);
  console.log("Iframe loaded successfully");
  
  await browser.close();
})();
