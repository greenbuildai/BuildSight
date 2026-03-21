/**
 * Backend health check utilities
 * Checks if the backend server is reachable and responsive
 */

/**
 * Check if backend is reachable
 * @param {string} apiBase - Base API URL
 * @param {number} timeout - Timeout in ms (default 3000)
 * @returns {Promise<{healthy: boolean, error: string|null, latency: number}>}
 */
export async function checkBackendHealth(apiBase, timeout = 3000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    const startTime = Date.now();

    try {
        const response = await fetch(`${apiBase}/health`, {
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const latency = Date.now() - startTime;

        if (!response.ok) {
            return {
                healthy: false,
                error: `Backend returned ${response.status}`,
                latency
            };
        }

        return { healthy: true, error: null, latency };
    } catch (err) {
        clearTimeout(timeoutId);

        // Categorize error types
        let errorMessage = 'Unknown error';

        if (err.name === 'AbortError') {
            errorMessage = 'Backend timeout - server not responding';
        } else if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            errorMessage = 'Cannot connect to backend - server may not be running';
        } else {
            errorMessage = err.message;
        }

        return {
            healthy: false,
            error: errorMessage,
            latency: Date.now() - startTime
        };
    }
}
