/**
 * auth.js — ASGARD AIOps RBAC Kimlik Doğrulama Modülü
 * Tüm HTML sayfalarına dahil edilir.
 */

// Sayfa içeriği yüklenmeden (flash efekti olmadan) yönlendirme
(function() {
    const TOKEN_KEY = 'asgard_token';
    const path = window.location.pathname;
    const currentPage = path.split('/').pop();
    
    // login.html'de veya kök dizinde (RedirectResponse zaten yolluyor) döngüye girmemesi için
    if (currentPage === 'login.html' || currentPage === '' || path === '/') return;

    if (!localStorage.getItem(TOKEN_KEY)) {
        window.location.replace('login.html');
    }
})();

const AUTH_API = 'http://127.0.0.1:5003';
const TOKEN_KEY = 'asgard_token';

/* ── Sayfa → İzin Anahtarı Haritası ──────────────────────────── */
const PAGE_MAP = {
    'dashboard.html': 'dashboard',
    'it.html': 'it',
    'ıot.html': 'iot',
    'finans.html': 'finance',
    'otonomLojistik.html': 'logistics',
    'erişim.html': 'access',
    'yapayZekaV2.html': 'ai',
    'ayarlar.html': 'settings',
    'profil.html': 'profile',
};

/* ── Token İşlemleri ──────────────────────────────────────────── */
function saveToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
}

function getToken() {
    return localStorage.getItem(TOKEN_KEY);
}

function clearToken() {
    localStorage.removeItem(TOKEN_KEY);
}

/* ── Logout ───────────────────────────────────────────────────── */
function logout() {
    clearToken();
    window.location.replace('login.html');
}

/* ── getUser: /api/me ile kullanıcı bilgisini çek ────────────── */
async function getUser() {
    const token = getToken();
    if (!token) return null;
    try {
        const res = await fetch(`${AUTH_API}/api/me`, {
            headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) return null;
        return await res.json();
    } catch {
        return null;
    }
}

/* ── checkAuth: Token geçersizse login sayfasına yönlendir ────── */
async function checkAuth() {
    const token = getToken();
    if (!token) {
        window.location.replace('login.html');
        return null;
    }
    const user = await getUser();
    if (!user) {
        clearToken();
        window.location.replace('login.html');
        return null;
    }
    return user;
}

/* ── guardPage: Bu sayfaya erişim yoksa dashboard'a yönlendir ── */
function guardPage(user) {
    const currentPage = window.location.pathname.split('/').pop() || 'dashboard.html';
    const pageKey = PAGE_MAP[currentPage];
    if (!pageKey) return; // Bilinmeyen sayfa, geç

    if (!user.pages.includes(pageKey)) {
        window.location.replace('dashboard.html');
    }
}

/* ── filterSidebar: İzin verilmeyen menü öğelerini gizle ─────── */
function filterSidebar(user) {
    // Sidebar menü öğeleri
    const roleToPage = {
        'dashboard': 'dashboard',
        'it': 'it',
        'iot': 'iot',
        'finance': 'finance',
        'logistics': 'logistics',
        'access': 'access',
        'ai': 'ai',
        'settings': 'settings',
        'profile': 'profile',
    };

    document.querySelectorAll('.menu-item[data-department]').forEach(item => {
        const dept = item.dataset.department;
        const pageKey = roleToPage[dept];
        if (pageKey && !user.pages.includes(pageKey)) {
            item.style.display = 'none';
        }
    });
}

/* ── updateSidebarUser: Alt kısımdaki kullanıcı adını güncelle ─ */
function updateSidebarUser(user) {
    const nameEl = document.querySelector('.sidebar .text-sm.font-medium');
    const emailEl = document.querySelector('.sidebar .text-xs.text-slate-400');
    const initialsEl = document.querySelector('.sidebar .text-xs.font-bold');
    const avatarContainer = document.querySelector('.sidebar .logo-icon') || document.querySelector('.sidebar .w-8.h-8.rounded-lg');
    
    if (nameEl) nameEl.textContent = user.name;
    if (emailEl) emailEl.textContent = user.email;
    
    const savedImg = localStorage.getItem('asgard_profile_img');
    if (savedImg && avatarContainer) {
        avatarContainer.innerHTML = `<img src="${savedImg}" class="w-full h-full object-cover rounded-lg" style="min-width: 100%; min-height: 100%;">`;
        if (initialsEl) initialsEl.style.display = 'none';
    } else if (initialsEl) {
        initialsEl.textContent = user.initials;
        initialsEl.style.display = '';
    }
}

/* ── bindNavigation: Global URL Router for Sidebar Menu ─────────── */
function bindNavigation() {
    document.querySelectorAll('.menu-item[data-department]').forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault(); // Prevent default link behavior if any
            const dept = this.dataset.department;

            // Re-map internal data-departments to the actual HTML files
            const routeMap = {
                'dashboard': 'dashboard.html',
                'it': 'it.html',
                'iot': 'ıot.html',
                'finance': 'finans.html',
                'logistics': 'otonomLojistik.html',
                'access': 'erişim.html',
                'ai': 'yapayZekaV2.html',
                'settings': 'ayarlar.html',
                'profile': 'profil.html',
            };

            const targetPage = routeMap[dept];
            if (targetPage) {
                window.location.href = targetPage;
            }
        });
    });
}

/* ── Tüm sayfalarda çalışacak ortak başlatıcı ────────────────── */
async function initAuth() {
    const user = await checkAuth();
    if (!user) return null;
    guardPage(user);
    filterSidebar(user);
    updateSidebarUser(user);
    bindNavigation();
    
    // Start global stream seamlessly after login
    startGlobalBackgroundStream();
    
    return user;
}

/* ── Global Background Data Stream ──────────────────────────────── *
 * Bu modül tüm departmanların anlık verilerini arkada dinler ve   *
 * geçiş yapıldığında verilerin "sıfırdan" başlamasını engeller.   */

function startGlobalBackgroundStream() {
    if (window.asgardStreamRunning) return;
    window.asgardStreamRunning = true;

    const BUFFER_KEY = 'asgard_global_stream_buffer';
    const MAX_POINTS = 30;

    function seedGlobalBuffer() {
        let buffer = JSON.parse(sessionStorage.getItem(BUFFER_KEY) || '[]');
        if (buffer.length < MAX_POINTS) {
            const needed = MAX_POINTS - buffer.length;
            const padding = [];
            for (let i = 0; i < needed; i++) {
                padding.push({
                    anomaly_score: 0.1 + Math.random() * 0.1,
                    fin_cost: 1.8 + Math.random() * 0.1,
                    fin_fraud: 0,
                    fin_risk: 0.15 + Math.random() * 0.05,
                    fin_inv: 0.8 + Math.random() * 0.2,
                    net_lat: 0.4 + Math.random() * 0.1,
                    net_thr: 1.0 + Math.random() * 0.2,
                    net_p_loss: 0.1 * Math.random(),
                    net_sec: 0.9 + Math.random() * 0.1,
                    iot_temp: 0.45 + Math.random() * 0.05,
                    iot_vib: 0.2 + Math.random() * 0.05,
                    iot_cycle: 0.8 + Math.random() * 0.1,
                    iot_torq: 0.6 + Math.random() * 0.1,
                    log_path: 0.5 + Math.random() * 0.1,
                    log_soc: 0.8 + Math.random() * 0.1,
                    log_task: 0.7 + Math.random() * 0.1,
                    it_cpu: 0.7 + Math.random() * 0.1,
                    iot_pdr: 0.8 + Math.random() * 0.1,
                    anomaly: 0
                });
            }
            buffer = [...padding, ...buffer];
            sessionStorage.setItem(BUFFER_KEY, JSON.stringify(buffer));
        }
    }

    seedGlobalBuffer();
    
    let currentInterval = (parseInt(localStorage.getItem('asgard_refresh_rate')) || 1) * 1000;
    let timerId = null;

    async function fetchIteration() {
        try {
            const res = await fetch('http://127.0.0.1:5003/api/stream');
            if (res.ok) {
                const d = await res.json();
                let buffer = JSON.parse(sessionStorage.getItem(BUFFER_KEY) || '[]');
                buffer.push(d);
                if (buffer.length > MAX_POINTS) buffer.shift();
                sessionStorage.setItem(BUFFER_KEY, JSON.stringify(buffer));
                window.dispatchEvent(new CustomEvent('asgard_stream_data', { detail: d }));
            }
        } catch(e) {}
        
        // Schedule next with potentially updated interval
        timerId = setTimeout(fetchIteration, currentInterval);
    }

    // Start
    fetchIteration();

    // Listen for setting changes
    window.addEventListener('storage', (e) => {
        if (e.key === 'asgard_refresh_rate') {
            const newRate = parseInt(e.newValue) || 1;
            currentInterval = newRate * 1000;
            console.log(`[ASGARD] Refresh rate updated to ${newRate}s`);
            // Optional: reset timer to apply immediately
            clearTimeout(timerId);
            timerId = setTimeout(fetchIteration, currentInterval);
        }
    });
}
