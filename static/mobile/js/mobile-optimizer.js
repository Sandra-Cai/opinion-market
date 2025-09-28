/**
 * Mobile Optimizer JavaScript
 * Provides mobile-specific optimizations and features
 */

class MobileOptimizer {
    constructor() {
        this.deviceInfo = null;
        this.connectionType = 'unknown';
        this.isOnline = navigator.onLine;
        this.cache = new Map();
        this.performanceMetrics = {};
        
        this.init();
    }
    
    async init() {
        await this.detectDevice();
        await this.detectConnection();
        this.setupEventListeners();
        this.optimizeForDevice();
        this.startPerformanceMonitoring();
    }
    
    async detectDevice() {
        const userAgent = navigator.userAgent;
        const screen = {
            width: window.screen.width,
            height: window.screen.height,
            pixelRatio: window.devicePixelRatio || 1
        };
        
        // Detect device type
        const isMobile = /Mobile|Android|iPhone|iPod|BlackBerry|Windows Phone|Opera Mini|IEMobile/i.test(userAgent);
        const isTablet = /iPad|Android.*Tablet|Kindle|Silk|PlayBook/i.test(userAgent);
        
        this.deviceInfo = {
            type: isMobile ? 'mobile' : isTablet ? 'tablet' : 'desktop',
            screen: screen,
            isTouchDevice: 'ontouchstart' in window,
            supportsWebP: this.supportsWebP(),
            supportsAvif: this.supportsAvif(),
            userAgent: userAgent
        };
        
        // Send device info to server
        try {
            const response = await fetch('/api/v1/mobile/device/detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    additional_info: {
                        screen_width: screen.width,
                        screen_height: screen.height,
                        pixel_ratio: screen.pixelRatio
                    }
                })
            });
            
            const data = await response.json();
            if (data.success) {
                this.deviceInfo = { ...this.deviceInfo, ...data.data.device_info };
                this.performanceMetrics = data.data.performance_metrics;
            }
        } catch (error) {
            console.warn('Failed to send device info to server:', error);
        }
    }
    
    async detectConnection() {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            this.connectionType = connection.effectiveType || 'unknown';
            
            // Listen for connection changes
            connection.addEventListener('change', () => {
                this.connectionType = connection.effectiveType || 'unknown';
                this.optimizeForConnection();
            });
        }
    }
    
    supportsWebP() {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
    }
    
    supportsAvif() {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        return canvas.toDataURL('image/avif').indexOf('data:image/avif') === 0;
    }
    
    setupEventListeners() {
        // Online/offline events
        window.addEventListener('online', () => {
            this.isOnline = true;
            this.onConnectionChange(true);
        });
        
        window.addEventListener('offline', () => {
            this.isOnline = false;
            this.onConnectionChange(false);
        });
        
        // Visibility change (app backgrounding)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.onAppBackground();
            } else {
                this.onAppForeground();
            }
        });
        
        // Page unload
        window.addEventListener('beforeunload', () => {
            this.onPageUnload();
        });
    }
    
    optimizeForDevice() {
        const deviceType = this.deviceInfo.type;
        
        // Add device-specific CSS classes
        document.body.classList.add(`device-${deviceType}`);
        
        if (this.deviceInfo.isTouchDevice) {
            document.body.classList.add('touch-device');
        }
        
        // Optimize images
        this.optimizeImages();
        
        // Optimize fonts
        this.optimizeFonts();
        
        // Setup lazy loading
        this.setupLazyLoading();
        
        // Optimize animations
        this.optimizeAnimations();
    }
    
    optimizeForConnection() {
        const connectionType = this.connectionType;
        
        // Adjust image quality based on connection
        if (connectionType === 'slow-2g' || connectionType === '2g') {
            this.setImageQuality('low');
        } else if (connectionType === '3g') {
            this.setImageQuality('medium');
        } else {
            this.setImageQuality('high');
        }
        
        // Adjust preloading based on connection
        if (connectionType === 'slow-2g' || connectionType === '2g') {
            this.disablePreloading();
        } else {
            this.enablePreloading();
        }
    }
    
    optimizeImages() {
        const images = document.querySelectorAll('img');
        
        images.forEach(img => {
            // Add lazy loading
            if (!img.hasAttribute('loading')) {
                img.setAttribute('loading', 'lazy');
            }
            
            // Optimize image format
            this.optimizeImageFormat(img);
            
            // Add responsive images
            this.addResponsiveImages(img);
        });
    }
    
    optimizeImageFormat(img) {
        const src = img.src;
        if (!src) return;
        
        // Convert to WebP or AVIF if supported
        if (this.deviceInfo.supportsAvif && !src.includes('.avif')) {
            img.src = src.replace(/\.(jpg|jpeg|png)$/i, '.avif');
        } else if (this.deviceInfo.supportsWebP && !src.includes('.webp')) {
            img.src = src.replace(/\.(jpg|jpeg|png)$/i, '.webp');
        }
    }
    
    addResponsiveImages(img) {
        const src = img.src;
        if (!src) return;
        
        // Create srcset for different screen densities
        const baseSrc = src.replace(/\.(jpg|jpeg|png|webp|avif)$/i, '');
        const extension = src.match(/\.(jpg|jpeg|png|webp|avif)$/i)?.[0] || '.jpg';
        
        const srcset = [
            `${baseSrc}${extension} 1x`,
            `${baseSrc}@2x${extension} 2x`,
            `${baseSrc}@3x${extension} 3x`
        ].join(', ');
        
        img.setAttribute('srcset', srcset);
    }
    
    setImageQuality(quality) {
        const qualityMap = {
            'low': 0.6,
            'medium': 0.8,
            'high': 1.0
        };
        
        document.documentElement.style.setProperty('--image-quality', qualityMap[quality]);
    }
    
    optimizeFonts() {
        // Preload critical fonts
        const criticalFonts = [
            'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
        ];
        
        criticalFonts.forEach(fontUrl => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'style';
            link.href = fontUrl;
            document.head.appendChild(link);
        });
    }
    
    setupLazyLoading() {
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        observer.unobserve(img);
                    }
                });
            });
            
            document.querySelectorAll('img[data-src]').forEach(img => {
                imageObserver.observe(img);
            });
        }
    }
    
    optimizeAnimations() {
        // Reduce animations on low-end devices
        if (this.deviceInfo.type === 'mobile' && this.connectionType === 'slow-2g') {
            document.body.classList.add('reduce-motion');
        }
        
        // Use CSS transforms for better performance
        document.body.classList.add('hardware-accelerated');
    }
    
    enablePreloading() {
        // Preload critical resources
        const criticalResources = [
            '/api/v1/markets',
            '/api/v1/user/profile'
        ];
        
        criticalResources.forEach(resource => {
            const link = document.createElement('link');
            link.rel = 'prefetch';
            link.href = resource;
            document.head.appendChild(link);
        });
    }
    
    disablePreloading() {
        // Remove preload links
        document.querySelectorAll('link[rel="prefetch"]').forEach(link => {
            link.remove();
        });
    }
    
    startPerformanceMonitoring() {
        // Monitor performance metrics
        if ('performance' in window) {
            window.addEventListener('load', () => {
                setTimeout(() => {
                    this.collectPerformanceMetrics();
                }, 1000);
            });
        }
    }
    
    collectPerformanceMetrics() {
        const navigation = performance.getEntriesByType('navigation')[0];
        const paint = performance.getEntriesByType('paint');
        
        const metrics = {
            loadTime: navigation.loadEventEnd - navigation.loadEventStart,
            domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
            firstPaint: paint.find(entry => entry.name === 'first-paint')?.startTime || 0,
            firstContentfulPaint: paint.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0,
            deviceType: this.deviceInfo.type,
            connectionType: this.connectionType
        };
        
        // Send metrics to server
        this.sendPerformanceMetrics(metrics);
    }
    
    async sendPerformanceMetrics(metrics) {
        try {
            await fetch('/api/v1/mobile/analytics/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    event_type: 'performance_metrics',
                    event_data: metrics
                })
            });
        } catch (error) {
            console.warn('Failed to send performance metrics:', error);
        }
    }
    
    onConnectionChange(isOnline) {
        if (isOnline) {
            // Sync offline data
            this.syncOfflineData();
        } else {
            // Enable offline mode
            this.enableOfflineMode();
        }
    }
    
    async syncOfflineData() {
        try {
            const offlineActions = this.getOfflineActions();
            if (offlineActions.length === 0) return;
            
            const response = await fetch('/api/v1/mobile/offline/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    actions: offlineActions,
                    last_sync_time: this.getLastSyncTime()
                })
            });
            
            const data = await response.json();
            if (data.success) {
                this.clearOfflineActions();
                this.handleSyncUpdates(data.data.updates);
            }
        } catch (error) {
            console.warn('Failed to sync offline data:', error);
        }
    }
    
    enableOfflineMode() {
        document.body.classList.add('offline-mode');
        this.showOfflineIndicator();
    }
    
    showOfflineIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'offline-indicator';
        indicator.textContent = 'You are offline. Changes will be synced when you reconnect.';
        document.body.appendChild(indicator);
    }
    
    onAppBackground() {
        // Pause non-essential operations
        this.pauseAnimations();
        this.reduceUpdateFrequency();
    }
    
    onAppForeground() {
        // Resume operations
        this.resumeAnimations();
        this.normalizeUpdateFrequency();
        
        // Check for updates
        this.checkForUpdates();
    }
    
    onPageUnload() {
        // Save important state
        this.saveState();
        
        // Send analytics
        this.sendPageUnloadAnalytics();
    }
    
    pauseAnimations() {
        document.body.classList.add('animations-paused');
    }
    
    resumeAnimations() {
        document.body.classList.remove('animations-paused');
    }
    
    reduceUpdateFrequency() {
        // Reduce WebSocket heartbeat frequency
        if (window.websocketManager) {
            window.websocketManager.setHeartbeatInterval(60000); // 1 minute
        }
    }
    
    normalizeUpdateFrequency() {
        // Restore normal update frequency
        if (window.websocketManager) {
            window.websocketManager.setHeartbeatInterval(30000); // 30 seconds
        }
    }
    
    async checkForUpdates() {
        try {
            const response = await fetch('/api/v1/mobile/offline/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    actions: [],
                    last_sync_time: this.getLastSyncTime()
                })
            });
            
            const data = await response.json();
            if (data.success && data.data.updates) {
                this.handleSyncUpdates(data.data.updates);
            }
        } catch (error) {
            console.warn('Failed to check for updates:', error);
        }
    }
    
    handleSyncUpdates(updates) {
        // Handle market updates
        if (updates.markets && updates.markets.length > 0) {
            this.updateMarkets(updates.markets);
        }
        
        // Handle trade updates
        if (updates.trades && updates.trades.length > 0) {
            this.updateTrades(updates.trades);
        }
        
        // Handle notifications
        if (updates.notifications && updates.notifications.length > 0) {
            this.showNotifications(updates.notifications);
        }
    }
    
    updateMarkets(markets) {
        // Update market data in the UI
        markets.forEach(market => {
            const marketElement = document.querySelector(`[data-market-id="${market.id}"]`);
            if (marketElement) {
                this.updateMarketElement(marketElement, market);
            }
        });
    }
    
    updateTrades(trades) {
        // Update trade data in the UI
        trades.forEach(trade => {
            const tradeElement = document.querySelector(`[data-trade-id="${trade.id}"]`);
            if (tradeElement) {
                this.updateTradeElement(tradeElement, trade);
            }
        });
    }
    
    showNotifications(notifications) {
        notifications.forEach(notification => {
            this.showNotification(notification);
        });
    }
    
    showNotification(notification) {
        const notificationElement = document.createElement('div');
        notificationElement.className = 'notification';
        notificationElement.textContent = notification.message;
        
        document.body.appendChild(notificationElement);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notificationElement.remove();
        }, 5000);
    }
    
    getOfflineActions() {
        return JSON.parse(localStorage.getItem('offline_actions') || '[]');
    }
    
    clearOfflineActions() {
        localStorage.removeItem('offline_actions');
    }
    
    getLastSyncTime() {
        return parseInt(localStorage.getItem('last_sync_time') || '0');
    }
    
    saveState() {
        // Save important application state
        const state = {
            currentPage: window.location.pathname,
            timestamp: Date.now()
        };
        
        localStorage.setItem('app_state', JSON.stringify(state));
    }
    
    sendPageUnloadAnalytics() {
        // Send page unload analytics
        this.sendAnalytics('page_unload', {
            page: window.location.pathname,
            timeOnPage: Date.now() - this.pageLoadTime
        });
    }
    
    async sendAnalytics(eventType, eventData) {
        try {
            await fetch('/api/v1/mobile/analytics/track', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.getAuthToken()}`
                },
                body: JSON.stringify({
                    event_type: eventType,
                    event_data: eventData
                })
            });
        } catch (error) {
            console.warn('Failed to send analytics:', error);
        }
    }
    
    getAuthToken() {
        return localStorage.getItem('auth_token') || '';
    }
    
    // Public API methods
    getDeviceInfo() {
        return this.deviceInfo;
    }
    
    getConnectionType() {
        return this.connectionType;
    }
    
    isOnline() {
        return this.isOnline;
    }
    
    optimizeContent(content) {
        // Optimize content based on device capabilities
        if (this.deviceInfo.type === 'mobile') {
            return this.optimizeForMobile(content);
        }
        return content;
    }
    
    optimizeForMobile(content) {
        // Reduce content size for mobile
        if (content.length > 1000) {
            return content.substring(0, 1000) + '...';
        }
        return content;
    }
}

// Initialize mobile optimizer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.mobileOptimizer = new MobileOptimizer();
    window.mobileOptimizer.pageLoadTime = Date.now();
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MobileOptimizer;
}
