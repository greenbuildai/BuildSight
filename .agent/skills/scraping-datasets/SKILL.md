---
name: scraping-datasets
description: Automates the collection and structure of training datasets from web sources. Use when the user requests data collection, web scraping, or dataset creation for model training.
---

# Dataset Web Scraper Skill

## When to use this skill
- Collecting images or text for ML model training.
- Need to download bulk files from a specific website structure.
- User asks to "scrape data", "get training images", or "build a dataset".

## Workflow

1.  **Target Analysis** (Manual Step using Browser Tools)
    -   Use `mcp_browsermcp_browser_navigate` to visit the target URL.
    -   Use `mcp_browsermcp_browser_snapshot` to identify the CSS selectors for the data items (e.g., image tags `img.gallery-item`, link tags `a.download-link`).
    -   Verify pagination logic (e.g., `?page=2` or "Next" button selector).

2.  **Configuration**
    -   Update the `scripts/scrape_dataset.py` template with:
        -   `BASE_URL`: The starting URL.
        -   `ITEM_SELECTOR`: CSS selector for the main data container.
        -   `DATA_SELECTOR`: CSS selector for the specific data (e.g., `img['src']`).
        -   `PAGINATION_Logic`: How to find the next page.

3.  **Dry Run**
    -   Run the script with a limit (e.g., `--limit 5`) to verify selectors work.
    -   Check the output directory for correct files.

4.  **Bulk Execution**
    -   Run the full scrape.
    -   Monitor for errors (403 Forbidden, 404 Not Found).
    -   *Tip:* If 403s occur, ensure User-Agent headers are set correctly in the script.

5.  **Validation & Cleaning**
    -   Remove corrupt files or empty data points.
    -   (Optional) Split into `train/`, `val/`, `test/` folders.

## Instructions

-   **Respect `robots.txt`**: Always check `target.com/robots.txt` first.
-   **Rate Limiting**: Ensure the script has `time.sleep()` calls to avoid DOSing the target.
-   **Error Handling**: The script handles network errors gracefully; do not stop on single failures.

## Resources
-   [Python Scraper Template](scripts/scrape_dataset.py)
