const envApiBase = import.meta.env.VITE_API_BASE_URL;
let resolvedBase = 'http://localhost:8000';

if (envApiBase) {
    resolvedBase = envApiBase;
    console.log('[API] Using environment API base:', resolvedBase);
} else if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location;
    if (!port || port === '8000') {
        resolvedBase = `${protocol}//${hostname}${port ? `:${port}` : ''}`;
    } else {
        resolvedBase = `${protocol}//${hostname}:8000`;
    }
    console.log('[API] Auto-resolved API base:', resolvedBase, { protocol, hostname, port });
} else {
    console.log('[API] Using default API base:', resolvedBase);
}

console.log('[API] Final API base URL:', resolvedBase);

export const apiBase = resolvedBase;
