/**
 * ASGARD Presentation Mode Trigger
 * Adds a hidden/convenient way to start/stop the 10-minute demo.
 */

async function startPresentationMode() {
    if (confirm("10 dakikalık Sunum Modu (IT & IoT Krizi) başlatılacaktır. Emin misiniz?")) {
        try {
            const token = localStorage.getItem('asgard_token');
            const response = await fetch('http://127.0.0.1:5003/api/demo/presentation-start', {
                method: 'POST',
                headers: { 
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            if (response.ok) {
                if (window.showAIToast) {
                    showAIToast("Sunum Modu Aktif", "10 dakikalık anomali senaryosu başlatıldı.", "Tamam", "critical");
                } else if (window.showToast) {
                    showToast("Sunum modu aktif edildi.", "success");
                } else {
                    alert(data.message);
                }
                
                // Refresh UI if needed
                if (window.checkDemoStatus) window.checkDemoStatus();
            } else {
                alert("Hata: " + data.detail);
            }
        } catch (error) {
            console.error("Presentation trigger error:", error);
            alert("Sunum modu başlatılamadı.");
        }
    }
}

async function stopPresentationMode() {
    try {
        const token = localStorage.getItem('asgard_token');
        const response = await fetch('http://127.0.0.1:5003/api/demo/presentation-stop', {
            method: 'POST',
            headers: { 
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        if (response.ok) {
            if (window.showAIToast) {
                showAIToast("Sunum Modu Kapalı", "Sistem gerçek zamanlı verilere dönüyor.", "Tamam", "stable");
            } else if (window.showToast) {
                showToast("Sunum modu durduruldu.", "info");
            } else {
                alert(data.message);
            }
            
            // Refresh UI if needed
            if (window.checkDemoStatus) window.checkDemoStatus();
        } else {
            alert("Hata: " + data.detail);
        }
    } catch (error) {
        console.error("Presentation stop error:", error);
        alert("Sunum modu durdurulamadı.");
    }
}

// Add global functions to be called from UI
window.asgardStartDemo = startPresentationMode;
window.asgardStopDemo = stopPresentationMode;
