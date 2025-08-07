/**
 * Authentication Utility
 * Shared authentication functions for admin panel
 */

class AuthManager {
    constructor() {
        this.tokenKey = 'admin_token';
        this.userKey = 'admin_user';
        this.baseUrl = '/api/admin/auth';
        this.loginUrl = '/admin/login';
        this.dashboardUrl = '/admin';
        // originalFetch will be set externally to avoid timing issues
        this.originalFetch = null;
    }

    /**
     * Get authentication token from cookie
     */
    getToken() {
        const cookies = document.cookie.split('; ');
        const tokenCookie = cookies.find(row => row.startsWith(`${this.tokenKey}=`));
        return tokenCookie ? tokenCookie.split('=')[1] : null;
    }

    /**
     * Set authentication token in cookie
     */
    setToken(token, days = 7) {
        const expires = new Date();
        expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
        document.cookie = `${this.tokenKey}=${token}; expires=${expires.toUTCString()}; path=/; SameSite=Strict`;
    }

    /**
     * Remove authentication token
     */
    removeToken() {
        document.cookie = `${this.tokenKey}=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT; SameSite=Strict`;
        localStorage.removeItem(this.userKey);
    }

    /**
     * Get user info from localStorage
     */
    getUser() {
        const userStr = localStorage.getItem(this.userKey);
        return userStr ? JSON.parse(userStr) : null;
    }

    /**
     * Set user info in localStorage
     */
    setUser(user) {
        localStorage.setItem(this.userKey, JSON.stringify(user));
    }

    /**
     * Enhanced fetch wrapper with automatic auth headers
     */
    async fetch(url, options = {}) {
        const token = this.getToken();
        
        // Default options
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        // Add auth header if token exists and not login endpoint
        if (token && !url.includes('/auth/login')) {
            defaultOptions.headers['Authorization'] = `Bearer ${token}`;
        }

        // Merge options
        const finalOptions = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers
            }
        };

        try {
            // Use fetch directly since no global override exists
            const response = await fetch(url, finalOptions);

            // Handle 401 Unauthorized
            if (response.status === 401) {
                console.warn('Authentication failed - redirecting to login');
                this.redirectToLogin();
                return null;
            }

            return response;
        } catch (error) {
            console.error('Auth fetch error:', error);
            throw error;
        }
    }

    /**
     * Check if user is authenticated
     */
    async checkAuth() {
        const token = this.getToken();
        if (!token) {
            return false;
        }

        try {
            const response = await this.fetch(`${this.baseUrl}/check`);
            
            if (response && response.ok) {
                const data = await response.json();
                this.setUser(data.user);
                return true;
            } else {
                this.removeToken();
                return false;
            }
        } catch (error) {
            console.error('Auth check failed:', error);
            this.removeToken();
            return false;
        }
    }

    /**
     * Login user
     */
    async login(username, password) {
        try {
            // Use fetch directly for login to avoid any wrapper issues
            const response = await fetch(`${this.baseUrl}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.setToken(data.access_token);
                this.setUser(data.user);
                return { success: true, user: data.user };
            } else {
                return { success: false, error: data.detail || 'Login failed' };
            }
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, error: 'Network error occurred' };
        }
    }

    /**
     * Logout user
     */
    async logout() {
        try {
            const token = this.getToken();
            if (token) {
                await this.fetch(`${this.baseUrl}/logout`, {
                    method: 'POST'
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.removeToken();
            this.redirectToLogin();
        }
    }

    /**
     * Change password
     */
    async changePassword(oldPassword, newPassword) {
        try {
            const response = await this.fetch(`${this.baseUrl}/change-password`, {
                method: 'POST',
                body: JSON.stringify({
                    old_password: oldPassword,
                    new_password: newPassword
                })
            });

            if (response && response.ok) {
                return { success: true };
            } else {
                const data = await response.json();
                return { success: false, error: data.detail || 'Password change failed' };
            }
        } catch (error) {
            console.error('Password change error:', error);
            return { success: false, error: 'Network error occurred' };
        }
    }

    /**
     * Get user profile
     */
    async getProfile() {
        try {
            const response = await this.fetch(`${this.baseUrl}/profile`);
            
            if (response && response.ok) {
                const profile = await response.json();
                return { success: true, profile };
            } else {
                return { success: false, error: 'Failed to load profile' };
            }
        } catch (error) {
            console.error('Profile error:', error);
            return { success: false, error: 'Network error occurred' };
        }
    }

    /**
     * Redirect to login page
     */
    redirectToLogin() {
        if (window.location.pathname !== this.loginUrl) {
            window.location.href = this.loginUrl;
        }
    }

    /**
     * Redirect to dashboard
     */
    redirectToDashboard() {
        window.location.href = this.dashboardUrl;
    }

    /**
     * Initialize auth on page load
     */
    async init() {
        const isLoginPage = window.location.pathname === this.loginUrl;
        
        if (isLoginPage) {
            // On login page, don't auto-redirect - let the page handle it
            console.log('Auth manager initialized on login page');
            return;
        } else {
            // On protected pages, ensure user is authenticated
            const isAuthenticated = await this.checkAuth();
            if (!isAuthenticated) {
                this.redirectToLogin();
            }
        }
    }

    /**
     * Show authentication status in UI
     */
    updateUI() {
        const user = this.getUser();
        const userElements = document.querySelectorAll('[data-auth-user]');
        const authElements = document.querySelectorAll('[data-auth-required]');

        if (user) {
            // Update user display elements
            userElements.forEach(element => {
                const field = element.getAttribute('data-auth-user');
                if (field && user[field]) {
                    element.textContent = user[field];
                } else if (!field) {
                    element.textContent = user.username || 'User';
                }
            });

            // Show authenticated elements
            authElements.forEach(element => {
                element.style.display = '';
            });
        } else {
            // Hide authenticated elements
            authElements.forEach(element => {
                element.style.display = 'none';
            });
        }
    }
}

// Store original fetch before any modifications
const originalFetch = window.fetch;

// Create global auth manager instance
window.auth = new AuthManager();
// Set the original fetch reference
window.auth.originalFetch = originalFetch;

// Auto-initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', function() {
    window.auth.init().then(() => {
        window.auth.updateUI();
    });
});

// No global fetch override - templates should use auth.fetch() directly

// Utility functions for backward compatibility
function getToken() {
    return window.auth.getToken();
}

async function checkAuth() {
    return await window.auth.checkAuth();
}

async function logout() {
    await window.auth.logout();
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AuthManager;
}
