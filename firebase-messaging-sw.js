/* Radar Cocodrilo Pro v10.9 — Firebase Cloud Messaging Service Worker */
importScripts('https://www.gstatic.com/firebasejs/10.14.1/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.14.1/firebase-messaging-compat.js');

firebase.initializeApp({
  apiKey: "AIzaSyCnbl-l8AvaG8CZfSjnisYLqSCXdsHQnVw",
  authDomain: "cocodrilopro-64220.firebaseapp.com",
  projectId: "cocodrilopro-64220",
  storageBucket: "cocodrilopro-64220.firebasestorage.app",
  messagingSenderId: "842936884826",
  appId: "1:842936884826:web:3e82485a7f3281b7074263",
  measurementId: "G-PWMZG95KPM"
});

const messaging = firebase.messaging();

messaging.onBackgroundMessage((payload) => {
  const data = payload?.data || {};
  const notif = payload?.notification || {};
  const title = notif.title || data.title || '🚨 ENTRAR YA — Cocodrilo Pro';
  const body = notif.body || data.body || 'Se ha detectado una oportunidad extrema en el Radar.';
  const ticker = (data.ticker || '').toUpperCase();
  const targetUrl = data.url || (ticker ? `./?ticker=${encodeURIComponent(ticker)}` : './?filter=ENTRADA_YA');

  const options = {
    body,
    tag: ticker ? `cocodrilo-entrar-ya-${ticker}` : 'cocodrilo-entrar-ya',
    renotify: true,
    requireInteraction: true,
    vibrate: [220, 100, 220, 100, 280],
    data: { url: targetUrl, ticker },
  };

  self.registration.showNotification(title, options);
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetRel = event.notification?.data?.url || './?filter=ENTRADA_YA';
  const targetFull = new URL(targetRel, self.registration.scope).href;

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url.startsWith(self.registration.scope) && 'focus' in client) {
          if ('navigate' in client) {
            return client.navigate(targetFull).then((c) => (c ? c.focus() : client.focus())).catch(() => client.focus());
          }
          return client.focus();
        }
      }
      if (clients.openWindow) return clients.openWindow(targetFull);
      return null;
    })
  );
});
