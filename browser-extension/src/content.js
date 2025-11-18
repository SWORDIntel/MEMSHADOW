/**
 * MEMSHADOW Content Script for Claude.ai
 *
 * Captures Claude conversations and injects relevant context
 */

// Configuration
const CONFIG = {
  apiEndpoint: 'http://localhost:8000/api/v1',
  captureInterval: 2000, // Check for new messages every 2s
  autoCapture: true,
  autoInject: true
};

// State
let currentProject = null;
let currentSession = null;
let lastCapturedMessageCount = 0;
let isInitialized = false;

/**
 * Initialize the extension on Claude.ai
 */
async function initialize() {
  if (isInitialized) return;

  console.log('[MEMSHADOW] Initializing on Claude.ai');

  // Get or create session
  currentSession = await getOrCreateSession();

  // Load user preferences
  const prefs = await chrome.storage.local.get(['currentProject', 'autoCapture', 'autoInject']);
  currentProject = prefs.currentProject;
  CONFIG.autoCapture = prefs.autoCapture !== false;
  CONFIG.autoInject = prefs.autoInject !== false;

  // Start monitoring
  if (CONFIG.autoCapture) {
    startConversationMonitoring();
  }

  // Add UI elements
  injectUI();

  isInitialized = true;
  console.log('[MEMSHADOW] Initialized', { session: currentSession, project: currentProject });
}

/**
 * Start monitoring Claude conversation
 */
function startConversationMonitoring() {
  setInterval(async () => {
    const messages = extractMessages();

    if (messages.length > lastCapturedMessageCount) {
      const newMessages = messages.slice(lastCapturedMessageCount);
      await captureMessages(newMessages);
      lastCapturedMessageCount = messages.length;
    }
  }, CONFIG.captureInterval);

  console.log('[MEMSHADOW] Started conversation monitoring');
}

/**
 * Extract messages from Claude UI
 */
function extractMessages() {
  const messages = [];

  // Claude.ai uses specific DOM structure for messages
  // This is a simplified selector - would need to be updated based on actual Claude.ai structure
  const messageElements = document.querySelectorAll('[data-test-render-count]');

  messageElements.forEach((el, index) => {
    const isUser = el.classList.contains('user-message') ||
                   el.querySelector('[data-is-user="true"]');

    const content = el.textContent.trim();

    // Extract code blocks
    const codeBlocks = [];
    el.querySelectorAll('pre code').forEach(code => {
      const language = code.className.match(/language-(\w+)/)?.[1] || 'plaintext';
      codeBlocks.push({
        language,
        content: code.textContent
      });
    });

    messages.push({
      index,
      role: isUser ? 'user' : 'assistant',
      content,
      codeBlocks,
      timestamp: new Date().toISOString()
    });
  });

  return messages;
}

/**
 * Capture messages to MEMSHADOW backend
 */
async function captureMessages(messages) {
  for (const message of messages) {
    try {
      if (message.role === 'user') {
        await captureUserMessage(message);
      } else {
        await captureAssistantResponse(message);
      }
    } catch (error) {
      console.error('[MEMSHADOW] Failed to capture message', error);
    }
  }
}

/**
 * Capture user message
 */
async function captureUserMessage(message) {
  const response = await fetch(`${CONFIG.apiEndpoint}/claude/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: message.content,
      project_id: currentProject,
      conversation_id: currentSession,
      metadata: {
        timestamp: message.timestamp
      }
    })
  });

  if (response.ok) {
    console.log('[MEMSHADOW] User message captured');
  }
}

/**
 * Capture assistant response
 */
async function captureAssistantResponse(message) {
  const response = await fetch(`${CONFIG.apiEndpoint}/claude/response`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: message.content,
      conversation_id: currentSession,
      code_blocks: message.codeBlocks,
      metadata: {
        timestamp: message.timestamp
      }
    })
  });

  if (response.ok) {
    const data = await response.json();
    console.log('[MEMSHADOW] Assistant response captured', {
      artifacts: data.artifacts?.length || 0
    });
  }
}

/**
 * Inject relevant context into Claude
 */
async function injectContext(query) {
  try {
    const response = await fetch(`${CONFIG.apiEndpoint}/claude/context`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        project_id: currentProject,
        session_id: currentSession,
        max_tokens: 5000
      })
    });

    if (response.ok) {
      const { context } = await response.json();
      return context;
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to get context', error);
  }

  return null;
}

/**
 * Inject MEMSHADOW UI elements
 */
function injectUI() {
  // Create floating action button
  const fab = document.createElement('div');
  fab.id = 'memshadow-fab';
  fab.innerHTML = `
    <button id="memshadow-fab-btn" title="MEMSHADOW Actions">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </button>
  `;
  fab.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 10000;
  `;

  const fabBtn = fab.querySelector('#memshadow-fab-btn');
  fabBtn.style.cssText = `
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.2s;
  `;

  fabBtn.addEventListener('mouseenter', () => {
    fabBtn.style.transform = 'scale(1.1)';
  });

  fabBtn.addEventListener('mouseleave', () => {
    fabBtn.style.transform = 'scale(1.0)';
  });

  fabBtn.addEventListener('click', () => {
    showQuickActions();
  });

  document.body.appendChild(fab);

  console.log('[MEMSHADOW] UI injected');
}

/**
 * Show quick actions menu
 */
function showQuickActions() {
  // Create modal
  const modal = document.createElement('div');
  modal.id = 'memshadow-quick-actions';
  modal.innerHTML = `
    <div class="memshadow-modal-overlay">
      <div class="memshadow-modal-content">
        <h2>MEMSHADOW Quick Actions</h2>
        <div class="memshadow-actions">
          <button id="memshadow-create-checkpoint">
            📍 Create Checkpoint
          </button>
          <button id="memshadow-search-memory">
            🔍 Search Memory
          </button>
          <button id="memshadow-inject-context">
            💉 Inject Context
          </button>
          <button id="memshadow-view-project">
            📁 View Project
          </button>
        </div>
        <button id="memshadow-close">Close</button>
      </div>
    </div>
  `;

  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    .memshadow-modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10001;
    }
    .memshadow-modal-content {
      background: white;
      padding: 24px;
      border-radius: 12px;
      max-width: 400px;
      width: 90%;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .memshadow-modal-content h2 {
      margin-top: 0;
      color: #333;
    }
    .memshadow-actions {
      display: grid;
      gap: 12px;
      margin: 20px 0;
    }
    .memshadow-actions button {
      padding: 12px 16px;
      border: none;
      border-radius: 6px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      cursor: pointer;
      font-size: 14px;
      text-align: left;
      transition: transform 0.2s;
    }
    .memshadow-actions button:hover {
      transform: translateY(-2px);
    }
    #memshadow-close {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 6px;
      background: white;
      cursor: pointer;
    }
  `;
  modal.appendChild(style);

  document.body.appendChild(modal);

  // Add event listeners
  modal.querySelector('#memshadow-close').addEventListener('click', () => {
    modal.remove();
  });

  modal.querySelector('.memshadow-modal-overlay').addEventListener('click', (e) => {
    if (e.target.classList.contains('memshadow-modal-overlay')) {
      modal.remove();
    }
  });

  modal.querySelector('#memshadow-create-checkpoint').addEventListener('click', async () => {
    await createCheckpoint();
    modal.remove();
  });

  modal.querySelector('#memshadow-search-memory').addEventListener('click', () => {
    // Open popup for search
    chrome.runtime.sendMessage({ action: 'openPopup', tab: 'search' });
    modal.remove();
  });

  modal.querySelector('#memshadow-inject-context').addEventListener('click', async () => {
    const query = prompt('What context do you need?');
    if (query) {
      const context = await injectContext(query);
      if (context) {
        // Insert context into Claude's input
        insertContextIntoInput(context);
      }
    }
    modal.remove();
  });
}

/**
 * Create session checkpoint
 */
async function createCheckpoint() {
  const summary = prompt('Session summary:');
  if (!summary) return;

  const objective = prompt('Current objective:');

  try {
    const response = await fetch(`${CONFIG.apiEndpoint}/claude/checkpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSession,
        project_id: currentProject,
        summary,
        current_objective: objective || 'Continue work',
        next_steps: []
      })
    });

    if (response.ok) {
      alert('✓ Checkpoint created successfully!');
    }
  } catch (error) {
    console.error('[MEMSHADOW] Failed to create checkpoint', error);
    alert('Failed to create checkpoint');
  }
}

/**
 * Insert context into Claude input
 */
function insertContextIntoInput(context) {
  // Find Claude's text input
  const input = document.querySelector('textarea[placeholder*="Talk to Claude"]') ||
                document.querySelector('[contenteditable="true"]');

  if (input) {
    const formattedContext = `\n\n<relevant_context>\n${context}\n</relevant_context>\n\n`;

    if (input.tagName === 'TEXTAREA') {
      input.value += formattedContext;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      input.textContent += formattedContext;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    console.log('[MEMSHADOW] Context injected into input');
  }
}

/**
 * Get or create session
 */
async function getOrCreateSession() {
  const stored = await chrome.storage.local.get(['currentSession']);

  if (stored.currentSession) {
    return stored.currentSession;
  }

  // Create new session
  const sessionId = `session_${Date.now()}`;
  await chrome.storage.local.set({ currentSession: sessionId });

  return sessionId;
}

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'captureNow') {
    const messages = extractMessages();
    captureMessages(messages);
    sendResponse({ success: true, count: messages.length });
  } else if (message.action === 'injectContext') {
    injectContext(message.query).then(context => {
      if (context) {
        insertContextIntoInput(context);
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false });
      }
    });
    return true; // Keep channel open for async response
  }
});

// Initialize when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
