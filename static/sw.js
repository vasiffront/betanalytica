const CACHE = 'betanalytica-v4';

// Статические ресурсы которые кэшируются при установке
const PRECACHE = [
  '/',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css'
];

// API-пути — всегда через сеть, никогда не кэшируются
const API_PATHS = ['/calculate', '/build_express', '/clear', '/restore', '/football_today', '/football_stats'];

// ── Установка: закэшировать статику ─────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

// ── Активация: удалить старые кэши ──────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// ── Запросы: кэш-стратегия ──────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // API-запросы — только сеть (данные должны быть свежими)
  if (API_PATHS.includes(url.pathname)) return;

  // POST-запросы — только сеть
  if (event.request.method !== 'GET') return;

  // Остальное: сначала кэш, если нет — сеть с сохранением в кэш
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (!response || response.status !== 200 || response.type === 'opaque') {
          return response;
        }
        const clone = response.clone();
        caches.open(CACHE).then(cache => cache.put(event.request, clone));
        return response;
      });
    })
  );
});
