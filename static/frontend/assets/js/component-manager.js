/**
 * Component Manager - Quản lý tất cả components
 */
class ComponentManager {
    constructor() {
        this.components = new Map();
        this.config = {
            apiBaseUrl: '/api/v1',
            componentsPath: '/static/frontend/components'
        };
        this.init();
    }

    /**
     * Khởi tạo component manager
     */
    async init() {
        console.log('🚀 Initializing Component Manager...');
        
        // Load menu config trước
        await this.loadMenuConfig();
        
        // Load các components cơ bản
        await this.loadComponent('header');
        await this.loadComponent('navigation');
        
        // Setup router
        this.setupRouter();
        
        // Setup mobile menu
        this.setupMobileMenu();
        
        console.log('✅ Component Manager initialized');
    }

    /**
     * Load menu configuration từ API hoặc local config
     */
    async loadMenuConfig() {
        try {
            // Menu configuration - dễ dàng thay đổi tại đây
            this.menuConfig = {
                logo: {
                    text: 'OCR System',
                    icon: 'AI',
                    href: '/' // Link về root URL
                },
                items: [
                    {
                        id: 'home',
                        label: 'Trang Chủ',
                        icon: '🏠',
                        href: '/static/frontend/pages/home.html',
                        active: true
                    },
                    {
                        id: 'card-detection',
                        label: 'Nhận Diện Thẻ',
                        icon: '🆔',
                        href: '/static/frontend/pages/card-detection.html'
                    },
                    {
                        id: 'mrz-extraction',
                        label: 'Trích Xuất MRZ',
                        icon: '📄',
                        href: '/static/frontend/pages/mrz-extraction.html'
                    },
                    {
                        id: 'mrz-parser',
                        label: 'Phân Tích MRZ',
                        icon: '🔍',
                        href: '/static/frontend/pages/mrz-parser.html'
                    },
                    {
                        id: 'config',
                        label: 'Cấu Hình',
                        icon: '⚙️',
                        href: '/static/frontend/pages/config.html'
                    },
                    {
                        id: 'api-docs',
                        label: 'API Docs',
                        icon: '📚',
                        href: '/docs',
                        external: true
                    }
                ]
            };
            
            console.log('✅ Menu configuration loaded');
        } catch (error) {
            console.error('❌ Failed to load menu config:', error);
        }
    }

    /**
     * Load component HTML và JavaScript
     */
    async loadComponent(componentName) {
        try {
            // Load HTML
            const htmlResponse = await fetch(`${this.componentsPath}/${componentName}.html`);
            if (!htmlResponse.ok) {
                throw new Error(`Failed to load ${componentName}.html`);
            }
            const html = await htmlResponse.text();

            // Load JavaScript nếu có
            let jsModule = null;
            try {
                jsModule = await import(`${this.componentsPath}/${componentName}.js`);
            } catch (jsError) {
                console.log(`📝 No JS module for ${componentName}`);
            }

            // Store component
            this.components.set(componentName, {
                html,
                module: jsModule,
                loaded: true
            });

            console.log(`✅ Component '${componentName}' loaded`);
            return true;
        } catch (error) {
            console.error(`❌ Failed to load component '${componentName}':`, error);
            return false;
        }
    }

    /**
     * Render component với data
     */
    renderComponent(componentName, data = {}) {
        const component = this.components.get(componentName);
        if (!component) {
            console.error(`❌ Component '${componentName}' not found`);
            return '';
        }

        let html = component.html;

        // Replace placeholders với data
        Object.keys(data).forEach(key => {
            const placeholder = new RegExp(`{{${key}}}`, 'g');
            html = html.replace(placeholder, data[key]);
        });

        return html;
    }

    /**
     * Mount component vào DOM element
     */
    async mountComponent(componentName, targetSelector, data = {}) {
        const target = document.querySelector(targetSelector);
        if (!target) {
            console.error(`❌ Target element '${targetSelector}' not found`);
            return false;
        }

        // Load component nếu chưa load
        if (!this.components.has(componentName)) {
            await this.loadComponent(componentName);
        }

        // Render và mount
        const html = this.renderComponent(componentName, data);
        target.innerHTML = html;

        // Execute component's JavaScript nếu có
        const component = this.components.get(componentName);
        if (component.module && component.module.init) {
            component.module.init(target, data);
        }

        console.log(`✅ Component '${componentName}' mounted to '${targetSelector}'`);
        return true;
    }

    /**
     * Setup client-side routing
     */
    setupRouter() {
        // Handle navigation clicks including logo
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href^="/static/frontend"]');
            const logo = e.target.closest('a.logo');
            
            if (link && !link.hasAttribute('data-external')) {
                e.preventDefault();
                this.navigateTo(link.href);
            } else if (logo && (logo.href === '/' || logo.href === window.location.origin + '/')) {
                e.preventDefault();
                this.navigateTo('/static/frontend/pages/home.html');
            }
        });

        // Handle browser back/forward
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.page) {
                this.loadPage(e.state.page, false);
            }
        });

        // Load initial page
        this.navigateToCurrentUrl();
    }

    /**
     * Navigate to URL
     */
    navigateTo(url) {
        const path = new URL(url, window.location.origin).pathname;
        history.pushState({ page: path }, '', url);
        this.loadPage(path);
    }

    /**
     * Navigate to current URL
     */
    navigateToCurrentUrl() {
        const path = window.location.pathname;
        
        // If we're at root (/), home page, or app routes, load home page
        if (path === '/' || path === '/app' || path === '/frontend' || path === '/static/frontend/index.html') {
            this.navigateTo('/static/frontend/pages/home.html');
        } else if (path.startsWith('/static/frontend/pages/')) {
            this.loadPage(path, false);
        } else {
            // Default to home for unknown paths
            this.navigateTo('/static/frontend/pages/home.html');
        }
    }

    /**
     * Load page content
     */
    async loadPage(path, updateHistory = true) {
        try {
            const response = await fetch(path);
            if (!response.ok) {
                throw new Error(`Failed to load page: ${path}`);
            }
            
            const html = await response.text();
            const mainContent = document.querySelector('#main-content');
            if (mainContent) {
                mainContent.innerHTML = html;
                
                // Ensure CSS is still loaded for page content
                this.ensurePageStyles();
                
                // Update active navigation
                this.updateActiveNavigation(path);
                
                // Execute page-specific JavaScript
                this.executePageScript(path);
                
                console.log(`✅ Page loaded: ${path}`);
            }
        } catch (error) {
            console.error(`❌ Failed to load page: ${path}`, error);
            this.showErrorPage(error.message);
        }
    }

    /**
     * Ensure page styles are applied
     */
    ensurePageStyles() {
        // Check if main CSS is loaded
        const cssLink = document.querySelector('link[href*="main.css"]');
        if (!cssLink) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = '/static/frontend/assets/css/main.css?v=2.0';
            document.head.appendChild(link);
            console.log('📄 CSS reloaded for page content');
        }
        
        // Add any missing utility classes that might be needed
        if (!document.querySelector('style[data-component-styles]')) {
            const style = document.createElement('style');
            style.setAttribute('data-component-styles', 'true');
            style.textContent = `
                /* Ensure grid classes work */
                .grid { display: grid; gap: 1.5rem; }
                .grid-cols-1 { grid-template-columns: repeat(1, 1fr); }
                .grid-cols-2 { grid-template-columns: repeat(2, 1fr); }
                .grid-cols-3 { grid-template-columns: repeat(3, 1fr); }
                .grid-cols-4 { grid-template-columns: repeat(4, 1fr); }
                
                /* Ensure spacing works */
                .mb-4 { margin-bottom: 1rem; }
                .mt-4 { margin-top: 1rem; }
                .gap-2 { gap: 0.5rem; }
                
                /* Responsive behavior */
                @media (max-width: 768px) {
                    .grid-cols-2, .grid-cols-3, .grid-cols-4 {
                        grid-template-columns: 1fr;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    /**
     * Update active navigation item
     */
    updateActiveNavigation(currentPath) {
        // Remove active class from all nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // Add active class to current nav link
        const currentLink = document.querySelector(`a[href="${currentPath}"]`);
        if (currentLink) {
            currentLink.classList.add('active');
        }
    }

    /**
     * Execute page-specific JavaScript
     */
    async executePageScript(path) {
        try {
            const pageName = path.split('/').pop().replace('.html', '');
            const scriptPath = `/static/frontend/pages/${pageName}.js`;
            
            // Dynamically import page script
            const module = await import(scriptPath);
            if (module.init) {
                module.init();
            }
        } catch (error) {
            // Page script is optional
            console.log(`📝 No script for page: ${path}`);
        }
    }

    /**
     * Setup mobile menu toggle
     */
    setupMobileMenu() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('.mobile-menu-toggle')) {
                const navMenu = document.querySelector('.nav-menu');
                if (navMenu) {
                    navMenu.classList.toggle('active');
                }
            }
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.header')) {
                const navMenu = document.querySelector('.nav-menu');
                if (navMenu && navMenu.classList.contains('active')) {
                    navMenu.classList.remove('active');
                }
            }
        });
    }

    /**
     * Show error page
     */
    showErrorPage(message) {
        const mainContent = document.querySelector('#main-content');
        if (mainContent) {
            mainContent.innerHTML = `
                <div class="text-center">
                    <div class="alert alert-error">
                        <h3>❌ Lỗi</h3>
                        <p>${message}</p>
                    </div>
                    <button class="btn btn-primary" onclick="window.location.reload()">
                        🔄 Tải Lại Trang
                    </button>
                </div>
            `;
        }
    }

    /**
     * Utility: Make API request
     */
    async apiRequest(endpoint, options = {}) {
        try {
            const url = `${this.config.apiBaseUrl}${endpoint}`;
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('❌ API Request failed:', error);
            throw error;
        }
    }

    /**
     * Utility: Show notification
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            animation: slideIn 0.3s ease;
        `;
        notification.innerHTML = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, duration);
    }

    /**
     * Get menu configuration
     */
    getMenuConfig() {
        return this.menuConfig;
    }
}

// CSS cho animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Export global instance
window.ComponentManager = ComponentManager;
