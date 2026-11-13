import test from "node:test";
import assert from "node:assert/strict";
import { sanitizeApiBaseUrl, buildApiUrl } from "../lib/api-url.ts";

test("sanitizeApiBaseUrl: handles empty, undefined, or null by returning default", () => {
  assert.equal(sanitizeApiBaseUrl(undefined), "http://127.0.0.1:8000");
  assert.equal(sanitizeApiBaseUrl(null), "http://127.0.0.1:8000");
  assert.equal(sanitizeApiBaseUrl(""), "http://127.0.0.1:8000");
  assert.equal(sanitizeApiBaseUrl("   "), "http://127.0.0.1:8000");
});

test("sanitizeApiBaseUrl: strips trailing slashes", () => {
  assert.equal(sanitizeApiBaseUrl("https://breastcare-api.onrender.com/"), "https://breastcare-api.onrender.com");
  assert.equal(sanitizeApiBaseUrl("https://breastcare-api.onrender.com///"), "https://breastcare-api.onrender.com");
  assert.equal(sanitizeApiBaseUrl("http://localhost:8000/"), "http://localhost:8000");
});

test("sanitizeApiBaseUrl: removes accidental query params (such as utm_source)", () => {
  assert.equal(
    sanitizeApiBaseUrl("https://breastcare-api.onrender.com/?utm_source=chatgpt.com"),
    "https://breastcare-api.onrender.com"
  );
  assert.equal(
    sanitizeApiBaseUrl("https://breastcare-api.onrender.com?utm_source=chatgpt.com&ref=123"),
    "https://breastcare-api.onrender.com"
  );
  assert.equal(
    sanitizeApiBaseUrl("http://localhost:8000/?foo=bar#section"),
    "http://localhost:8000"
  );
});

test("sanitizeApiBaseUrl: preserves subpaths if API is hosted behind a path prefix", () => {
  assert.equal(
    sanitizeApiBaseUrl("https://example.com/api-backend/?utm_source=chatgpt.com"),
    "https://example.com/api-backend"
  );
  assert.equal(
    sanitizeApiBaseUrl("https://example.com/api-backend/"),
    "https://example.com/api-backend"
  );
});

test("buildApiUrl: constructs clean endpoint paths regardless of base URL format", () => {
  const malformedBase = "https://breastcare-api.onrender.com/?utm_source=chatgpt.com";
  assert.equal(
    buildApiUrl("/api/analyze", malformedBase),
    "https://breastcare-api.onrender.com/api/analyze"
  );
  assert.equal(
    buildApiUrl("api/analyze", malformedBase),
    "https://breastcare-api.onrender.com/api/analyze"
  );
  assert.equal(
    buildApiUrl("/api/validate-image", malformedBase),
    "https://breastcare-api.onrender.com/api/validate-image"
  );
});

test("buildApiUrl: uses default when base is empty or omitted", () => {
  assert.equal(buildApiUrl("/api/analyze", ""), "http://127.0.0.1:8000/api/analyze");
  assert.equal(buildApiUrl("/api/validate-image", null), "http://127.0.0.1:8000/api/validate-image");
});
