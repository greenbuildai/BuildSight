import puppeteer from 'puppeteer';
import path from 'path';

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  console.log('Navigating to BuildSight Dashboard...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle2' });

  console.log('Switching to GeoAI tab...');
  // Find the GeoAI button in the sidebar and click it
  // Based on common patterns in this dashboard, it likely has text "GEOAI" or an ID
  const geoAIButtonClick = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button, a'));
    const geoButton = buttons.find(b => b.textContent && b.textContent.includes('GEOAI'));
    if (geoButton) {
      geoButton.click();
      return true;
    }
    return false;
  });

  if (geoAIButtonClick) {
    console.log('GeoAI button clicked. Waiting for map to load...');
    await new Promise(r => setTimeout(r, 5000)); // Wait for transition and Leaflet
  } else {
    console.log('GeoAI button not found, searching sidebar links...');
    const geoLink = await page.$('a[href*="geoai"]');
    if (geoLink) {
        await geoLink.click();
        await new Promise(r => setTimeout(r, 5000));
    }
  }

  const screenshotPath = 'C:\\Users\\brigh\\.gemini\\antigravity\\brain\\b91a58d0-aa9e-4add-a034-a83e4fc79afb\\geoai_map_screenshot.png';
  console.log(`Taking screenshot: ${screenshotPath}`);
  await page.screenshot({ path: screenshotPath });

  console.log('Screenshot successfully captured.');
  await browser.close();
})();
