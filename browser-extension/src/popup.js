/**
 * MEMSHADOW Popup UI Controller
 * TEMPEST Level C Compliant Interface
 */

// State
let currentProject = null;
let projects = [];
let searchTimeout = null;

/**
 * Initialize popup
 */
async function initialize() {
  console.log('[MEMSHADOW] Popup initializing');

  // Load state
  const stored = await chrome.storage.local.get(['currentProject', 'autoCapture', 'autoInject']);
  currentProject = stored.currentProject;

  // Set toggles
  document.getElementById('auto-capture').checked = stored.autoCapture !== false;
  document.getElementById('auto-inject').checked = stored.autoInject === true;

  // Load data
  await loadProjects();
  await loadStats();

  // Setup event listeners
  setupEventListeners();

  // Update status
  updateStatus('Connected');

  console.log('[MEMSHADOW] Popup initialized');
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Tab switching
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Dashboard actions
  document.getElementById('create-checkpoint').addEventListener('click', createCheckpoint);
  document.getElementById('inject-context').addEventListener('click', injectContext);
  document.getElementById('current-project').addEventListener('change', handleProjectChange);

  // Search
  document.getElementById('search-input').addEventListener('input', handleSearch);

  // Projects
  document.getElementById('new-project').addEventListener('click', createProject);

  // Settings
  document.getElementById('auto-capture').addEventListener('change', handleSettingChange);
  document.getElementById('auto-inject').addEventListener('change', handleSettingChange);
  document.getElementById('logout').addEventListener('click', handleLogout);
}

/**
 * Switch tab
 */
function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });

  // Update tab content
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.toggle('active', content.id === tabName);
  });

  // Load tab data
  if (tabName === 'projects') {
    loadProjects();
  } else if (tabName === 'dashboard') {
    loadStats();
  }
}

/**
 * Update status
 */
function updateStatus(status) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = status;
  statusEl.classList.add('connected');
}

/**
 * Load projects
 */
async function loadProjects() {
  try {
    const response = await chrome.runtime.sendMessage({ action: 'getProjects' });

    if (response.success) {
      projects = response.projects || [];
      renderProjects();
      updateProjectSelect();
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to load projects', error);
  }
}

/**
 * Render projects list
 */
function renderProjects() {
  const container = document.getElementById('projects-list');

  if (projects.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📁</div>
        <div>No projects yet</div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <ul class="list">
      ${projects.map(project => `
        <li class="list-item" data-id="${project.project_id}">
          <div class="list-item-title">${escapeHtml(project.name)}</div>
          <div class="list-item-meta">
            ${project.milestones_count || 0} milestones •
            ${project.memories_count || 0} memories
          </div>
        </li>
      `).join('')}
    </ul>
  `;

  // Add click listeners
  container.querySelectorAll('.list-item').forEach(item => {
    item.addEventListener('click', () => {
      const projectId = item.dataset.id;
      selectProject(projectId);
    });
  });
}

/**
 * Update project select dropdown
 */
function updateProjectSelect() {
  const select = document.getElementById('current-project');
  select.innerHTML = '<option value="">Select project...</option>';

  projects.forEach(project => {
    const option = document.createElement('option');
    option.value = project.project_id;
    option.textContent = project.name;

    if (project.project_id === currentProject) {
      option.selected = true;
    }

    select.appendChild(option);
  });
}

/**
 * Select project
 */
async function selectProject(projectId) {
  currentProject = projectId;
  await chrome.storage.local.set({ currentProject });

  updateProjectSelect();
  updateStatus(`Project: ${projects.find(p => p.project_id === projectId)?.name || 'Unknown'}`);

  // Switch to dashboard
  switchTab('dashboard');
}

/**
 * Create new project
 */
async function createProject() {
  const name = prompt('Project name:');
  if (!name) return;

  const description = prompt('Project description (optional):') || '';

  try {
    const response = await chrome.runtime.sendMessage({
      action: 'createProject',
      data: { name, description, objectives: [] }
    });

    if (response.success) {
      await loadProjects();
      selectProject(response.project.project_id);
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to create project', error);
    alert('Failed to create project');
  }
}

/**
 * Handle project change
 */
async function handleProjectChange(e) {
  const projectId = e.target.value;
  if (projectId) {
    selectProject(projectId);
  }
}

/**
 * Create checkpoint
 */
async function createCheckpoint() {
  if (!currentProject) {
    alert('Please select a project first');
    return;
  }

  const summary = prompt('Session summary:');
  if (!summary) return;

  const objective = prompt('Current objective:');

  try {
    const response = await chrome.runtime.sendMessage({
      action: 'createCheckpoint',
      data: {
        project_id: currentProject,
        summary,
        current_objective: objective || 'Continue work',
        next_steps: []
      }
    });

    if (response.success) {
      alert('✓ Checkpoint created');
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to create checkpoint', error);
    alert('Failed to create checkpoint');
  }
}

/**
 * Inject context
 */
async function injectContext() {
  if (!currentProject) {
    alert('Please select a project first');
    return;
  }

  const query = prompt('What context do you need?');
  if (!query) return;

  try {
    // Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab.url.includes('claude.ai')) {
      alert('Please open Claude.ai first');
      return;
    }

    // Send message to content script
    const response = await chrome.tabs.sendMessage(tab.id, {
      action: 'injectContext',
      query
    });

    if (response.success) {
      window.close();
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to inject context', error);
    alert('Failed to inject context');
  }
}

/**
 * Handle search
 */
function handleSearch(e) {
  const query = e.target.value.trim();

  // Debounce search
  clearTimeout(searchTimeout);

  if (query.length < 2) {
    document.getElementById('search-results').innerHTML = '';
    return;
  }

  searchTimeout = setTimeout(async () => {
    await performSearch(query);
  }, 300);
}

/**
 * Perform search
 */
async function performSearch(query) {
  const container = document.getElementById('search-results');
  container.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';

  try {
    const response = await chrome.runtime.sendMessage({
      action: 'searchMemory',
      query
    });

    if (response.success) {
      renderSearchResults(response.results);
    }
  } catch (error) {
    console.error('[MEMSHADOW] Search failed', error);
    container.innerHTML = '<div class="empty-state">Search failed</div>';
  }
}

/**
 * Render search results
 */
function renderSearchResults(results) {
  const container = document.getElementById('search-results');

  if (results.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🔍</div>
        <div>No results found</div>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <ul class="list">
      ${results.map(result => `
        <li class="list-item">
          <div class="list-item-title">${escapeHtml(truncate(result.content, 100))}</div>
          <div class="list-item-meta">
            Score: ${result.score.toFixed(2)} •
            ${new Date(result.created_at).toLocaleDateString()}
          </div>
        </li>
      `).join('')}
    </ul>
  `;
}

/**
 * Load stats
 */
async function loadStats() {
  // Mock stats - would fetch from API in production
  document.getElementById('stat-memories').textContent = '127';
  document.getElementById('stat-sessions').textContent = '8';
}

/**
 * Handle setting change
 */
async function handleSettingChange(e) {
  const setting = e.target.id;
  const value = e.target.checked;

  await chrome.storage.local.set({
    [toCamelCase(setting)]: value
  });

  console.log('[MEMSHADOW] Setting updated:', setting, value);
}

/**
 * Handle logout
 */
async function handleLogout() {
  if (confirm('Are you sure you want to logout?')) {
    await chrome.runtime.sendMessage({ action: 'logout' });
    window.close();
  }
}

/**
 * Utility: Escape HTML
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Utility: Truncate text
 */
function truncate(text, maxLength) {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

/**
 * Utility: Convert kebab-case to camelCase
 */
function toCamelCase(str) {
  return str.replace(/-([a-z])/g, g => g[1].toUpperCase());
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initialize);
