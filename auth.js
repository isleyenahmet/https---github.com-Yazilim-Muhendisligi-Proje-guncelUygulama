/**
 * auth.js — NEXUS AIOps RBAC Kimlik Doğrulama Modülü
 * Tüm HTML sayfalarına dahil edilir.
 */

// Sayfa içeriği yüklenmeden (flash efekti olmadan) yönlendirme
(function() {
    const TOKEN_KEY = 'nexus_token';
    const path = window.location.pathname;
    const currentPage = path.split('/').pop();
    
    // login.html'de veya kök dizinde (RedirectResponse zaten yolluyor) döngüye girmemesi için
    if (currentPage === 'login.html' || currentPage === '' || path === '/') return;

    if (!localStorage.getItem(TOKEN_KEY)) {
        window.location.href = 'login.html';
    }
})();

const AUTH_API = 'http://127.0.0.1:5003';
const TOKEN_KEY = 'nexus_token';

/* ── Sayfa → İzin Anahtarı Haritası ──────────────────────────── */
const PAGE_MAP = {
    'dashboard.html': 'dashboard',
    'it.html': 'it',
    'ıot.html': 'iot',
    'finans.html': 'finance',
    'inasnKaynaklari.html': 'hr',
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
    window.location.href = 'login.html';
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
        window.location.href = 'login.html';
        return null;
    }
    const user = await getUser();
    if (!user) {
        clearToken();
        window.location.href = 'login.html';
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
        window.location.href = 'dashboard.html';
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
        'hr': 'hr',
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
    if (nameEl) nameEl.textContent = user.name;
    if (emailEl) emailEl.textContent = user.email;
    if (initialsEl) initialsEl.textContent = user.initials;
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
                'hr': 'inasnKaynaklari.html',
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
    return user;
}
