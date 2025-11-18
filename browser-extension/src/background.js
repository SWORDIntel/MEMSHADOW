/**
 * MEMSHADOW Background Service Worker
 *
 * Handles API communication, state management, and cross-tab synchronization
 */

// Configuration
const API_BASE = 'http://localhost:8000/api/v1';
const SYNC_INTERVAL = 300000; // 5 minutes

// State
let authToken = null;
let currentUser = null;
let syncTimer = null;

/**
 * Initialize background service
 */
async function initialize() {
  console.log('[MEMSHADOW] Background service initializing');

  // Load stored auth
  const stored = await chrome.storage.local.get(['authToken', 'currentUser']);
  if (stored.authToken) {
    authToken = stored.authToken;
    currentUser = stored.currentUser;
  }

  // Start sync timer
  startSyncTimer();

  console.log('[MEMSHADOW] Background service initialized');
}

/**
 * Start periodic sync timer
 */
function startSyncTimer() {
  if (syncTimer) {
    clearInterval(syncTimer);
  }

  syncTimer = setInterval(async () => {
    await syncWithBackend();
  }, SYNC_INTERVAL);

  // Initial sync
  syncWithBackend();
}

/**
 * Sync with MEMSHADOW backend
 */
async function syncWithBackend() {
  if (!authToken) return;

  try {
    const response = await fetch(`${API_BASE}/sync/status`, {
      headers: {
        'Authorization': `Bearer ${authToken}`
      }
    });

    if (response.ok) {
      const data = await response.json();
      console.log('[MEMSHADOW] Sync status:', data);

      // Update badge if there are pending items
      if (data.pending_items > 0) {
        chrome.action.setBadgeText({ text: String(data.pending_items) });
        chrome.action.setBadgeBackgroundColor({ color: '#667eea' });
      } else {
        chrome.action.setBadgeText({ text: '' });
      }
    }
  } catch (error) {
    console.error('[MEMSHADOW] Sync failed', error);
  }
}

/**
 * API request wrapper with auth
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const headers = {
    'Content-Type': 'application/json',
    ...options.headers
  };

  if (authToken) {
    headers['Authorization'] = `Bearer ${authToken}`;
  }

  const response = await fetch(url, {
    ...options,
    headers
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(error.error || `API error: ${response.status}`);
  }

  return response.json();
}

/**
 * Handle messages from content scripts and popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse);
  return true; // Keep channel open for async response
});

/**
 * Handle message routing
 */
async function handleMessage(message, sender) {
  console.log('[MEMSHADOW] Message received:', message.action);

  try {
    switch (message.action) {
      case 'login':
        return await handleLogin(message.credentials);

      case 'logout':
        return await handleLogout();

      case 'getProjects':
        return await handleGetProjects();

      case 'createProject':
        return await handleCreateProject(message.data);

      case 'searchMemory':
        return await handleSearchMemory(message.query);

      case 'getContext':
        return await handleGetContext(message.data);

      case 'createCheckpoint':
        return await handleCreateCheckpoint(message.data);

      case 'openPopup':
        return await handleOpenPopup(message.tab);

      default:
        return { success: false, error: 'Unknown action' };
    }
  } catch (error) {
    console.error(`[MEMSHADOW] Error handling ${message.action}:`, error);
    return { success: false, error: error.message };
  }
}

/**
 * Handle login
 */
async function handleLogin(credentials) {
  const data = await apiRequest('/auth/login', {
    method: 'POST',
    body: JSON.stringify(credentials)
  });

  authToken = data.token;
  currentUser = data.user;

  await chrome.storage.local.set({
    authToken,
    currentUser
  });

  startSyncTimer();

  return { success: true, user: currentUser };
}

/**
 * Handle logout
 */
async function handleLogout() {
  authToken = null;
  currentUser = null;

  await chrome.storage.local.remove(['authToken', 'currentUser']);

  if (syncTimer) {
    clearInterval(syncTimer);
  }

  return { success: true };
}

/**
 * Handle get projects
 */
async function handleGetProjects() {
  const data = await apiRequest('/claude/projects');
  return { success: true, projects: data.projects };
}

/**
 * Handle create project
 */
async function handleCreateProject(projectData) {
  const data = await apiRequest('/claude/projects', {
    method: 'POST',
    body: JSON.stringify(projectData)
  });

  return { success: true, project: data };
}

/**
 * Handle search memory
 */
async function handleSearchMemory(query) {
  const data = await apiRequest('/memory/search', {
    method: 'POST',
    body: JSON.stringify({ query, limit: 20 })
  });

  return { success: true, results: data.results };
}

/**
 * Handle get context
 */
async function handleGetContext(contextData) {
  const data = await apiRequest('/claude/context', {
    method: 'POST',
    body: JSON.stringify(contextData)
  });

  return { success: true, context: data.context };
}

/**
 * Handle create checkpoint
 */
async function handleCreateCheckpoint(checkpointData) {
  const data = await apiRequest('/claude/checkpoint', {
    method: 'POST',
    body: JSON.stringify(checkpointData)
  });

  return { success: true, checkpoint: data };
}

/**
 * Handle open popup
 */
async function handleOpenPopup(tab) {
  // Open popup programmatically
  chrome.action.openPopup();
  return { success: true };
}

/**
 * Handle extension installation
 */
chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    console.log('[MEMSHADOW] Extension installed');

    // Set default settings
    await chrome.storage.local.set({
      autoCapture: true,
      autoInject: false,
      theme: 'tempest-c' // TEMPEST Level C compliant theme
    });

    // Open welcome page
    chrome.tabs.create({
      url: 'public/welcome.html'
    });
  } else if (details.reason === 'update') {
    console.log('[MEMSHADOW] Extension updated');
  }
});

/**
 * Handle extension startup
 */
chrome.runtime.onStartup.addListener(() => {
  console.log('[MEMSHADOW] Extension started');
  initialize();
});

// Initialize on load
initialize();
