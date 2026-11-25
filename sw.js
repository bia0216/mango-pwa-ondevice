// Service worker for the on-device app.
// App files: network first (always the newest), cache when offline.
// Model files and ONNX Runtime: cache first (they are big and only change with a new cache name), so the app
// works offline after the first scan.
const CACHE = "mango-ondevice-v3";
const ASSETS = [
  "./",
  "index.html",
  "style.css",
  "app.js",
  "ondevice.js",
  "manifest.webmanifest",
  "icons/icon.svg",
  "icons/icon-192.png",
  "icons/icon-512.png",
];
const ORT_PREFIX = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/";   // same version as ondevice.js

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

function cacheFirst(req) {
  return caches.match(req).then((hit) => hit || fetch(req).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
    return res;
  }));
}

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  if (url.href.startsWith(ORT_PREFIX)) { e.respondWith(cacheFirst(e.request)); return; }   // ONNX Runtime (CDN)
  if (url.origin !== self.location.origin) return;                                   // never touch model-server calls
  if (url.pathname.includes("/models/")) { e.respondWith(cacheFirst(e.request)); return; }   // .onnx + models.json
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request, { ignoreSearch: true }).then((r) => r || caches.match("index.html")))
  );
});
