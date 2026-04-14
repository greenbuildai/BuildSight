import puppeteer from 'puppeteer';
import path from 'path';

(async () => {
    console.log("Starting Chrome...");
    const browser = await puppeteer.launch({ 
        args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security'] 
    });
    
    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 900 });
        
        // Use path resolution starting one level up (where render_geojson.html is)
        const filePath = `file:///${path.resolve('../render_geojson.html').replace(/\\/g, '/')}`;
        console.log(`Navigating to ${filePath}`);
        
        await page.goto(filePath, { waitUntil: 'networkidle0' });
        
        console.log("Waiting for rendering...");
        await new Promise(r => setTimeout(r, 4000));
        
        const savePath = 'C:\\Users\\brigh\\.gemini\\antigravity\\brain\\b91a58d0-aa9e-4add-a034-a83e4fc79afb\\artifacts\\geoai_direct_render_screenshot.png';
        const savePathFallback = 'C:\\Users\\brigh\\.gemini\\antigravity\\brain\\b91a58d0-aa9e-4add-a034-a83e4fc79afb\\geoai_map_screenshot_2.png';
        
        console.log(`Taking screenshot: ${savePathFallback}`);
        
        await page.screenshot({ path: savePathFallback, fullPage: true });
        console.log("Screenshot successfully captured.");
    } catch (e) {
        console.error("Error capturing screenshot:", e);
    } finally {
        await browser.close();
        console.log("Chrome closed.");
    }
})();
