import puppeteer from 'puppeteer';

(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    await page.goto('http://localhost:5173/?view=geoai');
    await new Promise(r => setTimeout(r, 3000));
    
    const status = await page.evaluate(() => {
        return {
            windowKeys: Object.keys(window).filter(k => k.toLowerCase().includes('heatmap')),
            HeatmapOverlay: typeof window.HeatmapOverlay,
            L_HeatmapOverlay: typeof (window.L ? window.L.HeatmapOverlay : undefined)
        }
    });
    console.log(status);
    
    await browser.close();
})();
