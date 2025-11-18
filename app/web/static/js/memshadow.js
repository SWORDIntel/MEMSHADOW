/**
 * MEMSHADOW Web Interface
 * Shared JavaScript utilities and API wrappers
 */

// API Base URL
const API_BASE = '/api';

// ============================================================================
// API Wrapper Functions
// ============================================================================

class MEMSHADOW_API {
    /**
     * Generic API request
     */
    static async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const finalOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, finalOptions);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || data.detail || 'Request failed');
            }

            return data;

        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    /**
     * GET request
     */
    static async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }

    /**
     * POST request
     */
    static async post(endpoint, data = null) {
        return this.request(endpoint, {
            method: 'POST',
            body: data ? JSON.stringify(data) : null
        });
    }

    /**
     * PUT request
     */
    static async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    static async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // ========================================================================
    // System Endpoints
    // ========================================================================

    static async getStatus() {
        return this.get('/status');
    }

    static async getStats() {
        return this.get('/stats');
    }

    static async getConfig() {
        return this.get('/config');
    }

    static async getConfigSection(section) {
        return this.get(`/config/${section}`);
    }

    static async updateConfig(section, updates) {
        return this.put(`/config/${section}`, { section, updates });
    }

    static async saveConfig() {
        return this.post('/config/save');
    }

    // ========================================================================
    // Federated Learning Endpoints
    // ========================================================================

    static async startFederated() {
        return this.post('/federated/start');
    }

    static async stopFederated() {
        return this.post('/federated/stop');
    }

    static async joinFederation(peers) {
        return this.post('/federated/join', peers);
    }

    static async shareFederatedUpdate(updateData, privacyBudget = 0.1) {
        return this.post('/federated/update', {
            update_data: updateData,
            privacy_budget: privacyBudget
        });
    }

    static async getFederatedStats() {
        return this.get('/federated/stats');
    }

    static async getFederatedPeers() {
        return this.get('/federated/peers');
    }

    // ========================================================================
    // Meta-Learning Endpoints
    // ========================================================================

    static async adaptToTask(taskData) {
        return this.post('/meta-learning/adapt', taskData);
    }

    static async getPerformanceMetrics() {
        return this.get('/meta-learning/performance');
    }

    static async getImprovementProposals() {
        return this.get('/meta-learning/proposals');
    }

    // ========================================================================
    // Consciousness Endpoints
    // ========================================================================

    static async startConsciousness() {
        return this.post('/consciousness/start');
    }

    static async stopConsciousness() {
        return this.post('/consciousness/stop');
    }

    static async processConsciously(inputItems, goalContext, mode = 'hybrid') {
        return this.post('/consciousness/process', {
            input_items: inputItems,
            goal_context: goalContext,
            processing_mode: mode
        });
    }

    static async getConsciousnessState() {
        return this.get('/consciousness/state');
    }

    static async getWorkspaceItems() {
        return this.get('/consciousness/workspace');
    }

    // ========================================================================
    // Self-Modifying Endpoints
    // ========================================================================

    static async startSelfModifying() {
        return this.post('/self-modifying/start');
    }

    static async stopSelfModifying() {
        return this.post('/self-modifying/stop');
    }

    static async improveFunction(functionName, sourceCode, categories, autoApply = false) {
        return this.post('/self-modifying/improve', {
            function_name: functionName,
            source_code: sourceCode,
            categories: categories,
            auto_apply: autoApply
        });
    }

    static async getSelfModifyingStatus() {
        return this.get('/self-modifying/status');
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';

    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));

    return (bytes / Math.pow(1024, i)).toFixed(2) + ' ' + sizes[i];
}

/**
 * Format duration (seconds) to human-readable string
 */
function formatDuration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(0)}s`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

/**
 * Format percentage
 */
function formatPercent(value, decimals = 1) {
    return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        return false;
    }
}

/**
 * Download data as JSON file
 */
function downloadJSON(data, filename) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

/**
 * Parse JSON safely
 */
function safeJSONParse(text, fallback = null) {
    try {
        return JSON.parse(text);
    } catch (error) {
        console.error('JSON parse error:', error);
        return fallback;
    }
}

/**
 * Debounce function calls
 */
function debounce(func, wait) {
    let timeout;

    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };

        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function calls
 */
function throttle(func, limit) {
    let inThrottle;

    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// ============================================================================
// Notification System
// ============================================================================

class NotificationManager {
    static show(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `tempest-alert tempest-alert-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            min-width: 300px;
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, duration);
    }

    static success(message, duration) {
        this.show(message, 'success', duration);
    }

    static error(message, duration) {
        this.show(message, 'danger', duration);
    }

    static warning(message, duration) {
        this.show(message, 'warning', duration);
    }

    static info(message, duration) {
        this.show(message, 'info', duration);
    }
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================================================
// Chart/Graph Utilities (for future visualizations)
// ============================================================================

class SimpleLineChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = [];
        this.maxDataPoints = 100;
    }

    addDataPoint(value) {
        this.data.push(value);

        if (this.data.length > this.maxDataPoints) {
            this.data.shift();
        }

        this.render();
    }

    render() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const ctx = this.ctx;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (this.data.length < 2) return;

        // Find min/max
        const min = Math.min(...this.data);
        const max = Math.max(...this.data);
        const range = max - min || 1;

        // Draw line
        ctx.strokeStyle = '#00ff41';
        ctx.lineWidth = 2;
        ctx.beginPath();

        this.data.forEach((value, index) => {
            const x = (index / (this.maxDataPoints - 1)) * width;
            const y = height - ((value - min) / range) * height;

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00ff41';
        ctx.stroke();
    }
}

// ============================================================================
// Export API
// ============================================================================

// Make API available globally
window.MEMSHADOW_API = MEMSHADOW_API;
window.NotificationManager = NotificationManager;
window.SimpleLineChart = SimpleLineChart;
