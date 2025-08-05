/**
 * Header Component JavaScript
 */
export function init(element, data) {
    console.log('🎯 Header component initialized');
    
    // Generate navigation items
    generateNavigationItems();
}

function generateNavigationItems() {
    const navMenu = document.querySelector('#nav-menu');
    if (!navMenu || !window.componentManager) return;
    
    const menuConfig = window.componentManager.getMenuConfig();
    if (!menuConfig || !menuConfig.items) return;
    
    // Generate navigation HTML
    let navigationHtml = '';
    menuConfig.items.forEach(item => {
        const activeClass = item.active ? ' active' : '';
        const externalAttr = item.external ? ' data-external="true" target="_blank"' : '';
        
        navigationHtml += `
            <li class="nav-item">
                <a href="${item.href}" class="nav-link${activeClass}"${externalAttr}>
                    <span class="nav-icon">${item.icon}</span>
                    ${item.label}
                </a>
            </li>
        `;
    });
    
    navMenu.innerHTML = navigationHtml;
    
    // Update logo
    const logo = document.querySelector('.logo');
    if (logo && menuConfig.logo) {
        const logoIcon = logo.querySelector('.logo-icon');
        const logoTextNode = Array.from(logo.childNodes).find(node => 
            node.nodeType === Node.TEXT_NODE && node.textContent.trim()
        );
        
        if (logoIcon) logoIcon.textContent = menuConfig.logo.icon;
        if (logoTextNode) logoTextNode.textContent = menuConfig.logo.text;
        
        // Update logo href if provided
        if (menuConfig.logo.href) {
            logo.setAttribute('href', menuConfig.logo.href);
        }
    }
}
