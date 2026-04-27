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
        
        const filePath = `file:///${path.resolve('render_geojson.html').replace(/\\/g, '/')}`;
        console.log(`Navigating to ${filePath}`);
        
        await page.goto(filePath, { waitUntil: 'networkidle0' });
        
        // Let it render the tiles completely
        console.log("Waiting for rendering...");
        await new Promise(r => setTimeout(r, 4000));
        
        const savePath = 'C:\\Users\\brigh\\.gemini\\antigravity\\brain\\b91a58d0-aa9e-4add-a034-a83e4fc79afb\\artifacts\\geoai_direct_render_screenshot.png';
        console.log(`Taking screenshot: ${savePath}`);
        
        await page.screenshot({ path: savePath, fullPage: true });
        console.log("Screenshot successfully captured.");
    } catch (e) {
        console.error("Error capturing screenshot:", e);
    } finally {
        await browser.close();
        console.log("Chrome closed.");
    }
})();
