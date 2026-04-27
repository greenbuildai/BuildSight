import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Ensure we use an absolute URI
        await page.goto("file:///E:/Company/Green%20Build%20AI/Prototypes/BuildSight/zone_viewer.html")
        # wait a couple seconds for tiles to theoretically load
        await page.wait_for_timeout(3000)
        # Capture screenshot
        await page.screenshot(path="zone_viewer_tilt.png")
        await browser.close()

asyncio.run(run())
