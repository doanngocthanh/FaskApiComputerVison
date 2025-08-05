/**
 * Card Detection Page JavaScript
 */
export function init() {
    console.log('🆔 Card Detection page initialized');
    
    setupFileUpload();
    loadSupportedCards();
    loadDetectionHistory();
}

function setupFileUpload() {
    const fileUpload = document.getElementById('file-upload');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');
    const detectBtn = document.getElementById('detect-btn');
    const clearBtn = document.getElementById('clear-btn');
    const uploadProgress = document.getElementById('upload-progress');

    // Drag and drop events
    fileUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUpload.classList.add('dragover');
    });

    fileUpload.addEventListener('dragleave', () => {
        fileUpload.classList.remove('dragover');
    });

    fileUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUpload.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // Detect button
    detectBtn.addEventListener('click', () => {
        const file = fileInput.files[0];
        if (file) {
            detectCard(file);
        }
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        clearPreview();
    });
}

function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        window.componentManager.showNotification('❌ Vui lòng chọn file ảnh', 'error');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        window.componentManager.showNotification('❌ File quá lớn (tối đa 10MB)', 'error');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const previewImage = document.getElementById('preview-image');
        const imagePreview = document.getElementById('image-preview');
        
        previewImage.src = e.target.result;
        imagePreview.classList.remove('hidden');
        
        // Hide upload area
        document.querySelector('.file-upload .text-center').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function clearPreview() {
    const imagePreview = document.getElementById('image-preview');
    const fileInput = document.getElementById('file-input');
    
    imagePreview.classList.add('hidden');
    fileInput.value = '';
    
    // Show upload area
    document.querySelector('.file-upload .text-center').style.display = 'block';
    
    // Clear results
    showDetectionResults(null);
}

async function detectCard(file) {
    try {
        const uploadProgress = document.getElementById('upload-progress');
        const detectBtn = document.getElementById('detect-btn');
        
        // Show progress
        uploadProgress.classList.remove('hidden');
        detectBtn.disabled = true;
        detectBtn.innerHTML = '<div class="loading" style="width: 16px; height: 16px;"></div> Đang xử lý...';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Make API request
        const response = await fetch('/api/v1/card/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        // Show results
        showDetectionResults(result);
        
        // Save to history
        saveToHistory(result, file.name);
        
        window.componentManager.showNotification('✅ Nhận diện thẻ thành công!', 'success');
        
    } catch (error) {
        console.error('❌ Card detection failed:', error);
        window.componentManager.showNotification(`❌ Lỗi nhận diện: ${error.message}`, 'error');
        
        showDetectionResults({ error: error.message });
        
    } finally {
        // Hide progress
        const uploadProgress = document.getElementById('upload-progress');
        const detectBtn = document.getElementById('detect-btn');
        
        uploadProgress.classList.add('hidden');
        detectBtn.disabled = false;
        detectBtn.innerHTML = '🔍 Nhận Diện Thẻ';
    }
}

function showDetectionResults(result) {
    const resultsContainer = document.getElementById('detection-results');
    
    if (!result || result.error) {
        resultsContainer.innerHTML = `
            <div class="alert alert-error">
                <h4>❌ Lỗi Nhận Diện</h4>
                <p>${result?.error || 'Có lỗi xảy ra trong quá trình nhận diện'}</p>
            </div>
        `;
        return;
    }
    
    if (!result.detections || result.detections.length === 0) {
        resultsContainer.innerHTML = `
            <div class="alert alert-warning">
                <h4>⚠️ Không Phát Hiện Thẻ</h4>
                <p>${result.message || 'Không tìm thấy thẻ nào trong ảnh'}</p>
                <p style="font-size: 0.875rem; margin-top: 0.5rem;">
                    Thử với ảnh khác hoặc đảm bảo thẻ được chụp rõ nét
                </p>
            </div>
        `;
        return;
    }
    
    const detection = result.detections[0];
    const confidence = (detection.confidence * 100).toFixed(1);
    
    resultsContainer.innerHTML = `
        <div class="alert alert-success">
            <h4>✅ Nhận Diện Thành Công</h4>
        </div>
        
        <div class="mb-4">
            <h5 style="margin-bottom: 0.5rem;">🎯 Kết Quả:</h5>
            <div class="grid grid-cols-1 gap-2">
                <div class="flex justify-between">
                    <span>Loại thẻ:</span>
                    <strong>${detection.card_category.name}</strong>
                </div>
                <div class="flex justify-between">
                    <span>Kiểu thẻ:</span>
                    <strong>${detection.card_type.name}</strong>
                </div>
                <div class="flex justify-between">
                    <span>Độ tin cậy:</span>
                    <strong style="color: ${confidence > 80 ? 'var(--success-color)' : confidence > 60 ? 'var(--warning-color)' : 'var(--error-color)'}">
                        ${confidence}%
                    </strong>
                </div>
                <div class="flex justify-between">
                    <span>Label:</span>
                    <code>${detection.detected_label}</code>
                </div>
            </div>
        </div>
        
        <div class="mb-4">
            <h5 style="margin-bottom: 0.5rem;">🔍 Đặc Điểm OCR:</h5>
            <div class="grid grid-cols-2 gap-2">
                <div class="flex items-center gap-2">
                    <span style="color: ${detection.ocr_features.has_portrait ? 'var(--success-color)' : 'var(--error-color)'}">
                        ${detection.ocr_features.has_portrait ? '✓' : '✗'}
                    </span>
                    <span>Ảnh chân dung</span>
                </div>
                <div class="flex items-center gap-2">
                    <span style="color: ${detection.ocr_features.has_qr_code ? 'var(--success-color)' : 'var(--error-color)'}">
                        ${detection.ocr_features.has_qr_code ? '✓' : '✗'}
                    </span>
                    <span>QR Code</span>
                </div>
                <div class="flex items-center gap-2">
                    <span style="color: ${detection.ocr_features.has_basic_info ? 'var(--success-color)' : 'var(--error-color)'}">
                        ${detection.ocr_features.has_basic_info ? '✓' : '✗'}
                    </span>
                    <span>Thông tin cơ bản</span>
                </div>
                <div class="flex items-center gap-2">
                    <span style="color: ${detection.ocr_features.has_address_info ? 'var(--success-color)' : 'var(--error-color)'}">
                        ${detection.ocr_features.has_address_info ? '✓' : '✗'}
                    </span>
                    <span>Thông tin địa chỉ</span>
                </div>
            </div>
        </div>
        
        ${detection.ocr_features.detected_info_types.length > 0 ? `
            <div>
                <h5 style="margin-bottom: 0.5rem;">📋 Thông Tin Phát Hiện:</h5>
                <div class="flex flex-wrap gap-1">
                    ${detection.ocr_features.detected_info_types.map(type => `
                        <span class="badge" style="background: var(--primary-color); color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">
                            ${type}
                        </span>
                    `).join('')}
                </div>
            </div>
        ` : ''}
    `;
}

async function loadSupportedCards() {
    try {
        const container = document.getElementById('supported-cards');
        const configData = await window.componentManager.apiRequest('/card/config');
        
        const cardsHtml = `
            <div class="grid grid-cols-2">
                <div>
                    <h5 style="margin-bottom: 1rem;">📋 Loại Thẻ Hỗ Trợ:</h5>
                    <div class="grid grid-cols-1 gap-2">
                        ${configData.card_categories.map(cat => `
                            <div class="flex items-center gap-2">
                                <span style="color: var(--success-color);">✓</span>
                                <span>${cat.name}</span>
                                <span style="color: var(--text-secondary); font-size: 0.875rem;">(${cat.nameEn})</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div>
                    <h5 style="margin-bottom: 1rem;">🔄 Kiểu Thẻ:</h5>
                    <div class="grid grid-cols-1 gap-2">
                        ${configData.card_types.map(type => `
                            <div class="flex items-center gap-2">
                                <span style="color: var(--success-color);">✓</span>
                                <span>${type.name}</span>
                                <span style="color: var(--text-secondary); font-size: 0.875rem;">(${type.nameEn})</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = cardsHtml;
        
    } catch (error) {
        console.error('❌ Failed to load supported cards:', error);
        document.getElementById('supported-cards').innerHTML = `
            <div class="alert alert-warning">
                <p>⚠️ Không thể tải danh sách thẻ hỗ trợ</p>
            </div>
        `;
    }
}

function saveToHistory(result, filename) {
    try {
        const history = JSON.parse(localStorage.getItem('detection_history') || '[]');
        
        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            filename,
            result,
            success: result.detections && result.detections.length > 0
        };
        
        history.unshift(entry);
        
        // Keep only last 10 entries
        if (history.length > 10) {
            history.splice(10);
        }
        
        localStorage.setItem('detection_history', JSON.stringify(history));
        loadDetectionHistory();
        
    } catch (error) {
        console.error('❌ Failed to save to history:', error);
    }
}

function loadDetectionHistory() {
    try {
        const history = JSON.parse(localStorage.getItem('detection_history') || '[]');
        const container = document.getElementById('detection-history');
        
        if (history.length === 0) {
            container.innerHTML = `
                <p style="color: var(--text-secondary); text-align: center;">
                    Chưa có lịch sử nhận diện
                </p>
            `;
            return;
        }
        
        const historyHtml = history.map(entry => {
            const date = new Date(entry.timestamp).toLocaleString('vi-VN');
            const status = entry.success ? 
                '<span style="color: var(--success-color);">✅ Thành công</span>' :
                '<span style="color: var(--error-color);">❌ Thất bại</span>';
            
            const cardInfo = entry.success && entry.result.detections?.[0] ?
                `${entry.result.detections[0].card_category.name} - ${entry.result.detections[0].card_type.name}` :
                'Không nhận diện được';
            
            return `
                <div class="flex justify-between items-center p-2 border border-gray-200 rounded">
                    <div>
                        <div style="font-weight: 500;">${entry.filename}</div>
                        <div style="font-size: 0.875rem; color: var(--text-secondary);">${cardInfo}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">${date}</div>
                    </div>
                    <div>${status}</div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = historyHtml;
        
        // Setup clear history button
        document.getElementById('clear-history-btn').addEventListener('click', () => {
            localStorage.removeItem('detection_history');
            loadDetectionHistory();
            window.componentManager.showNotification('🗑️ Đã xóa lịch sử', 'info');
        });
        
    } catch (error) {
        console.error('❌ Failed to load detection history:', error);
    }
}
