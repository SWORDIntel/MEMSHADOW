# MEMSHADOW Browser Extension

**Phase 6: Browser Extension & UI**

TEMPEST Level C compliant browser extension for Claude.ai integration.

## Features

### Auto-Capture
- Automatically captures Claude conversations
- Extracts code blocks and artifacts
- Tracks turn-by-turn interactions
- Stores conversation history

### Context Injection
- Intelligent context retrieval
- Relevant memory injection
- Project-aware suggestions
- Token-optimized context

### Project Management
- Multi-project organization
- Milestone tracking
- Session checkpointing
- Project-wide search

### TEMPEST Level C Compliance
- Low-contrast display (3:1 ratio)
- Subdued gray color scheme (#3a3a3a background)
- No bright whites or pure blacks
- Minimal electromagnetic emanation
- Secure visual design

## Installation

### Development Installation

1. **Build Extension**
   ```bash
   cd browser-extension
   # No build step required - pure JavaScript
   ```

2. **Load in Chrome/Edge**
   - Open `chrome://extensions`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select `browser-extension` directory

3. **Load in Firefox**
   - Open `about:debugging`
   - Click "This Firefox"
   - Click "Load Temporary Add-on"
   - Select `browser-extension/manifest.json`

### Production Build

```bash
# Create production zip
cd browser-extension
zip -r memshadow-extension.zip . -x "*.git*" -x "*node_modules*"
```

## Usage

### Initial Setup

1. **Install Extension**
   - Follow installation instructions above

2. **Configure Backend**
   - Ensure MEMSHADOW backend is running on `localhost:8000`
   - Or update `CONFIG.apiEndpoint` in `src/content.js`

3. **Create Project**
   - Click extension icon
   - Navigate to "Projects" tab
   - Click "New Project"
   - Enter project name and description

### Capturing Conversations

1. **Navigate to Claude.ai**
   - Extension auto-activates on Claude.ai pages
   - Floating action button appears in bottom-right

2. **Auto-Capture** (Default: Enabled)
   - Conversations are automatically captured
   - Check popup for capture status

3. **Manual Checkpoint**
   - Click floating action button
   - Select "Create Checkpoint"
   - Enter session summary and objectives

### Injecting Context

1. **Via Popup**
   - Click extension icon
   - Click "Inject Context"
   - Enter query for relevant context
   - Context is inserted into Claude input

2. **Via Floating Button**
   - Click floating action button on Claude.ai
   - Select "Inject Context"
   - Enter your query

### Searching Memories

1. **Open Popup**
   - Click extension icon
   - Navigate to "Search" tab

2. **Enter Query**
   - Type search query (minimum 2 characters)
   - Results appear automatically (debounced 300ms)

3. **View Results**
   - Relevance score displayed
   - Click result to view details

## Architecture

### Content Script (`src/content.js`)
- Runs on Claude.ai pages
- Monitors conversation
- Extracts messages and code blocks
- Injects UI elements
- Handles context injection

### Background Worker (`src/background.js`)
- Handles API communication
- Manages authentication
- Synchronizes with backend
- Coordinates cross-tab state

### Popup UI (`public/popup.html`, `src/popup.js`)
- TEMPEST Level C compliant design
- Dashboard with stats
- Search interface
- Project management
- Settings panel

## Configuration

### API Endpoint

Update in `src/content.js`:

```javascript
const CONFIG = {
  apiEndpoint: 'http://localhost:8000/api/v1',
  captureInterval: 2000,
  autoCapture: true,
  autoInject: true
};
```

### TEMPEST Theme

The extension uses a TEMPEST Level C compliant color scheme:

- **Background**: `#3a3a3a` (subdued dark gray)
- **Text**: `#a8a8a8` (medium gray)
- **Borders**: `#505050` (subtle borders)
- **Accents**: `#909090` (low contrast)

All colors maintain <3:1 contrast ratio for minimal electromagnetic emanation.

## Security Features

### TEMPEST Level C
- Low-contrast visual design
- Subdued color palette
- Minimal screen emanations
- Secure by design

### Data Protection
- Local-first operation
- Encrypted communication with backend
- No cloud dependencies in extension
- User-controlled data capture

### Permissions

The extension requires:

- `storage`: Store user preferences
- `activeTab`: Access current Claude.ai tab
- `scripting`: Inject context into page
- `host_permissions`: Claude.ai and localhost API

## API Integration

The extension communicates with MEMSHADOW backend via REST API:

### Endpoints Used

```javascript
POST /api/v1/claude/message      // Capture user message
POST /api/v1/claude/response     // Capture assistant response
POST /api/v1/claude/context      // Get relevant context
POST /api/v1/claude/checkpoint   // Create session checkpoint
GET  /api/v1/claude/projects     // List projects
POST /api/v1/claude/projects     // Create project
POST /api/v1/memory/search       // Search memories
GET  /api/v1/sync/status         // Get sync status
```

## Development

### File Structure

```
browser-extension/
├── manifest.json           # Extension manifest (MV3)
├── src/
│   ├── content.js          # Content script for Claude.ai
│   ├── background.js       # Background service worker
│   └── popup.js            # Popup UI controller
├── public/
│   ├── popup.html          # Popup UI (TEMPEST-C)
│   └── icons/              # Extension icons
└── README.md               # This file
```

### Adding Features

1. **Content Script Features**
   - Edit `src/content.js`
   - Access Claude.ai DOM
   - Use `chrome.runtime.sendMessage()` for background communication

2. **Background Features**
   - Edit `src/background.js`
   - Handle API calls
   - Manage persistent state

3. **UI Features**
   - Edit `public/popup.html` and `src/popup.js`
   - Maintain TEMPEST-C color scheme
   - Keep low contrast (<3:1)

## Troubleshooting

### Extension Not Loading
- Check Chrome/Firefox developer mode enabled
- Verify manifest.json is valid
- Check browser console for errors

### Capture Not Working
- Verify on Claude.ai page
- Check content script loaded (inspect page)
- Verify backend is running
- Check CORS settings on backend

### Context Injection Fails
- Ensure project is selected
- Verify backend connection
- Check Claude.ai DOM structure (may change)

## Browser Compatibility

- ✅ Chrome/Edge 88+ (Manifest V3)
- ✅ Firefox 109+ (Manifest V3)
- ⚠️ Safari (requires conversion to WebExtension)

## License

Part of MEMSHADOW project - See main repository for license

## Version History

### 1.0.0 (Current)
- Initial release
- Auto-capture conversations
- Context injection
- Project management
- TEMPEST Level C UI
- Chrome/Firefox support
