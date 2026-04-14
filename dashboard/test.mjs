import puppeteer from 'puppeteer';
import fs from 'fs';

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    page.on('console', msg => console.log('CONSOLE:', msg.text()));
    page.on('pageerror', err => console.log('PAGE_ERROR:', err.toString()));
    
    await page.goto('http://localhost:5173/');
    await new Promise(r => setTimeout(r, 2000));
    
    await page.evaluate(() => {
        const buttons = Array.from(document.querySelectorAll('.nav-link'));
        const geoAIButton = buttons.find(b => b.textContent && b.textContent.includes('GeoAI'));
        if (geoAIButton) geoAIButton.click();
    });
    
    await new Promise(r => setTimeout(r, 3000));
    
    const hasMapShell = await page.evaluate(() => !!document.querySelector('.geoai-map-shell'));
    const mapHtml = await page.evaluate(() => document.querySelector('.geoai-layout__map')?.innerHTML || 'NO MAP LAYOUT');
    console.log('HAS MAP SHELL:', hasMapShell);
    console.log('MAP HTML START');
    console.log(mapHtml.substring(0, 1000));
    console.log('MAP HTML END');
    
    const containerClasses = await page.evaluate(() => {
        const el = document.querySelector('.geoai-map-container');
        if (!el) return 'NOT FOUND';
        return JSON.stringify({
           className: el.className,
           clientWidth: el.clientWidth,
           clientHeight: el.clientHeight,
           offsetWidth: el.offsetWidth,
           offsetHeight: el.offsetHeight
        });
    });
    console.log('CONTAINER DIMENSIONS:', containerClasses);

    // Get any errors from Leaflet map initialization
    const leafletStatus = await page.evaluate(() => {
       return {
           hasWindowL: !!window.L,
           hasHeatmapOverlay: !!window.HeatmapOverlay
       }
    });
    console.log('LEAFLET STATUS:', leafletStatus);
    
    const html = await page.evaluate(() => document.body.innerHTML);
    fs.writeFileSync('body.html', html);
    await page.screenshot({ path: 'screenshot.png' });
    
    await browser.close();
})();
