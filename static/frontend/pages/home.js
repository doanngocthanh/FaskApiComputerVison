/**
 * Home Page JavaScript
 */
export function init() {
    console.log('🏠 Home page initialized');
    loadSystemStats();
}

async function loadSystemStats() {
    try {
        const statsContainer = document.getElementById('system-stats');
        if (!statsContainer) return;

        // Load card configuration stats
        const configData = await window.componentManager.apiRequest('/card/config');
        
        // Create stats HTML
        const statsHtml = `
            <div class="grid grid-cols-2">
                <div class="text-center">
                    <div style="font-size: 2rem; color: var(--primary-color); font-weight: bold;">
                        ${configData.card_categories_count}
                    </div>
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">
                        Loại Thẻ Hỗ Trợ
                    </div>
                </div>
                <div class="text-center">
                    <div style="font-size: 2rem; color: var(--success-color); font-weight: bold;">
                        ${configData.card_types_count}
                    </div>
                    <div style="font-size: 0.875rem; color: var(--text-secondary);">
                        Kiểu Thẻ
                    </div>
                </div>
            </div>
            <div class="mt-4">
                <h5 style="margin-bottom: 0.5rem;">📋 Loại Thẻ Được Hỗ Trợ:</h5>
                <div class="grid grid-cols-1 gap-1">
                    ${configData.card_categories.map(cat => `
                        <div class="flex items-center gap-2">
                            <span style="color: var(--success-color);">✓</span>
                            <span>${cat.name} (${cat.nameEn})</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        statsContainer.innerHTML = statsHtml;
        
    } catch (error) {
        console.error('❌ Failed to load system stats:', error);
        const statsContainer = document.getElementById('system-stats');
        if (statsContainer) {
            statsContainer.innerHTML = `
                <div class="alert alert-warning">
                    <p>⚠️ Không thể tải thống kê hệ thống</p>
                </div>
            `;
        }
    }
}
