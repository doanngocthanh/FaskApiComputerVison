// Navigation Component JavaScript
class NavigationComponent {
    constructor() {
        this.nav = document.querySelector('nav');
        this.mobileMenuBtn = document.querySelector('.mobile-menu-btn');
        this.navLinks = document.querySelector('.nav-links');
        this.dropdownToggles = document.querySelectorAll('.dropdown-toggle');
        this.dropdowns = document.querySelectorAll('.dropdown');
        
        this.init();
    }

    init() {
        this.setupScrollEffect();
        this.setupMobileMenu();
        this.setupDropdowns();
        this.setupKeyboardNavigation();
        this.setupClickOutside();
        this.setupHoverEffects();
    }

    // Navbar scroll effect
    setupScrollEffect() {
        if (!this.nav) return;

        window.addEventListener('scroll', () => {
            if (window.scrollY > 80) {
                this.nav.classList.add('scrolled');
            } else {
                this.nav.classList.remove('scrolled');
            }
        });
    }

    // Mobile menu functionality
    setupMobileMenu() {
        if (!this.mobileMenuBtn || !this.navLinks) return;

        this.mobileMenuBtn.addEventListener('click', () => {
            this.toggleMobileMenu();
        });

        // Close mobile menu when clicking on links
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 768) {
                    this.closeMobileMenu();
                }
            });
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth > 768) {
                this.closeMobileMenu();
                this.resetDropdowns();
            }
        });
    }

    toggleMobileMenu() {
        this.mobileMenuBtn.classList.toggle('active');
        this.navLinks.classList.toggle('active');
        
        // Prevent body scroll when menu is open
        document.body.style.overflow = this.navLinks.classList.contains('active') ? 'hidden' : '';
    }

    closeMobileMenu() {
        this.mobileMenuBtn.classList.remove('active');
        this.navLinks.classList.remove('active');
        document.body.style.overflow = '';
    }

    // Dropdown functionality
    setupDropdowns() {
        this.dropdownToggles.forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                
                if (window.innerWidth <= 768) {
                    this.toggleMobileDropdown(toggle);
                }
            });
        });
    }

    toggleMobileDropdown(toggle) {
        const dropdown = toggle.nextElementSibling;
        const isVisible = dropdown.style.display === 'block';
        
        // Close all other dropdowns
        this.dropdowns.forEach(d => {
            if (d !== dropdown) {
                d.style.display = 'none';
            }
        });
        
        // Toggle current dropdown
        dropdown.style.display = isVisible ? 'none' : 'block';
        
        // Animate dropdown if opening
        if (!isVisible && typeof gsap !== 'undefined') {
            gsap.fromTo(dropdown, 
                { opacity: 0, y: -10 },
                { opacity: 1, y: 0, duration: 0.3, ease: "power2.out" }
            );
        }
    }

    resetDropdowns() {
        this.dropdowns.forEach(dropdown => {
            dropdown.style.display = '';
        });
    }

    // Keyboard navigation
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeMobileMenu();
                if (window.innerWidth <= 768) {
                    this.resetDropdowns();
                }
            }
        });
    }

    // Click outside to close
    setupClickOutside() {
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-item') && !e.target.closest('.mobile-menu-btn')) {
                this.closeMobileMenu();
                
                if (window.innerWidth <= 768) {
                    this.resetDropdowns();
                }
            }
        });
    }

    // Hover effects and animations
    setupHoverEffects() {
        // Smooth hover animations for dropdown items
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        
        dropdownItems.forEach(item => {
            item.addEventListener('mouseenter', () => {
                if (window.innerWidth > 768 && typeof gsap !== 'undefined') {
                    gsap.to(item, {
                        x: 5,
                        duration: 0.2,
                        ease: "power2.out"
                    });
                }
            });

            item.addEventListener('mouseleave', () => {
                if (window.innerWidth > 768 && typeof gsap !== 'undefined') {
                    gsap.to(item, {
                        x: 0,
                        duration: 0.2,
                        ease: "power2.out"
                    });
                }
            });
        });

        // Logo hover effect
        const logo = document.querySelector('.logo');
        if (logo && typeof gsap !== 'undefined') {
            logo.addEventListener('mouseenter', () => {
                gsap.to(logo, {
                    scale: 1.05,
                    duration: 0.2,
                    ease: "power2.out"
                });
            });

            logo.addEventListener('mouseleave', () => {
                gsap.to(logo, {
                    scale: 1,
                    duration: 0.2,
                    ease: "power2.out"
                });
            });
        }
    }

    // Method to highlight active menu item
    setActiveMenuItem(path) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            
            const link = item.querySelector('a');
            if (link && link.getAttribute('href') === path) {
                item.classList.add('active');
            }
        });
    }

    // Method to add custom menu item
    addMenuItem(item, position = 'end') {
        const navLinks = this.navLinks;
        if (!navLinks) return;

        const li = document.createElement('li');
        li.className = 'nav-item';
        li.innerHTML = item;

        if (position === 'end') {
            navLinks.appendChild(li);
        } else if (position === 'start') {
            navLinks.insertBefore(li, navLinks.firstChild);
        } else if (typeof position === 'number') {
            const children = navLinks.children;
            if (position < children.length) {
                navLinks.insertBefore(li, children[position]);
            } else {
                navLinks.appendChild(li);
            }
        }
    }
}

// Initialize navigation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.navigationComponent = new NavigationComponent();
    
    // Set active menu item based on current path
    const currentPath = window.location.pathname;
    window.navigationComponent.setActiveMenuItem(currentPath);
});

// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('a[href^="#"], a[href^="/#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            const targetId = href.includes('#') ? href.split('#')[1] : null;
            
            if (targetId) {
                const target = document.getElementById(targetId);
                if (target) {
                    if (typeof gsap !== 'undefined') {
                        gsap.to(window, {
                            duration: 1,
                            scrollTo: target,
                            ease: "power2.inOut"
                        });
                    } else {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            }
        });
    });
});
