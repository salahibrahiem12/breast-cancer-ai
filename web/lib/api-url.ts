/**
 * Utility functions for constructing and sanitizing API URLs.
 *
 * Ensures that accidental query strings (such as UTM parameters like
 * `?utm_source=chatgpt.com`), trailing slashes, fragments, or malformed
 * inputs in NEXT_PUBLIC_API_URL are cleanly stripped, leaving only the
 * valid origin and basePath.
 */

const DEFAULT_API_URL = "http://127.0.0.1:8000";

/**
 * Sanitizes an API base URL string into a clean origin + pathname without
 * trailing slashes, search params, or hashes.
 */
export function sanitizeApiBaseUrl(rawUrl?: string | null): string {
  const trimmed = (rawUrl ?? "").trim();
  if (!trimmed) {
    return DEFAULT_API_URL;
  }

  try {
    // If input is absolute or relative with a host, URL parser will handle it
    const parsed = new URL(trimmed.startsWith("http://") || trimmed.startsWith("https://")
      ? trimmed
      : `http://${trimmed}`
    );

    // If original string had no protocol, retain original format if needed, but in our app
    // NEXT_PUBLIC_API_URL is always an HTTP/HTTPS absolute URL or host
    const protocol = trimmed.startsWith("https://") ? "https:" : (trimmed.startsWith("http://") ? "http:" : parsed.protocol);
    const host = parsed.host;
    // Strip trailing slash from pathname, but preserve sub-paths if API is hosted at a subpath
    const cleanPath = parsed.pathname.replace(/\/+$/, "");

    return `${protocol}//${host}${cleanPath}`;
  } catch {
    // Fallback regex cleanup if URL constructor throws on an obscure string
    const withoutQuery = trimmed.split(/[?#]/)[0];
    const clean = withoutQuery.replace(/\/+$/, "");
    return clean || DEFAULT_API_URL;
  }
}

/**
 * Constructs a fully qualified, validated endpoint URL from a base URL and endpoint path.
 *
 * Examples:
 * buildApiUrl("https://breastcare-api.onrender.com/?utm_source=chatgpt.com", "/api/analyze")
 * => "https://breastcare-api.onrender.com/api/analyze"
 *
 * buildApiUrl("https://breastcare-api.onrender.com/", "api/analyze")
 * => "https://breastcare-api.onrender.com/api/analyze"
 */
export function buildApiUrl(endpointPath: string, rawBaseUrl?: string | null): string {
  const base = sanitizeApiBaseUrl(rawBaseUrl !== undefined ? rawBaseUrl : process.env.NEXT_PUBLIC_API_URL);
  const normalizedPath = endpointPath.startsWith("/") ? endpointPath : `/${endpointPath}`;
  return `${base}${normalizedPath}`;
}
