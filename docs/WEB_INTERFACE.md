# MEMSHADOW Web Interface

**TEMPEST CLASS C** Military-Grade Command & Control Interface

## Overview

The MEMSHADOW Web Interface provides comprehensive access to all Phase 8 advanced memory systems through a high-security, high-contrast TEMPEST CLASS C themed interface.

## Features

### Core Interface
- **Real-time dashboard** with system status monitoring
- **Component management** for all Phase 8 systems
- **Configuration panel** with live updates
- **System logs** with terminal-style output
- **TEMPEST CLASS C theming** - Military-grade visual security standard

### Supported Components

1. **Federated Learning** (Phase 8.1)
   - Start/stop federated coordinator
   - Peer management and federation joining
   - Differential privacy configuration
   - Gossip protocol settings
   - Update broadcasting with privacy budgets

2. **Meta-Learning** (Phase 8.2)
   - MAML task adaptation
   - Performance metrics tracking
   - Improvement proposals
   - Continual learning management

3. **Consciousness Architecture** (Phase 8.3)
   - Global workspace monitoring
   - Attention director control
   - Metacognitive state visualization
   - Processing mode selection

4. **Self-Modification Engine** (Phase 8.4)
   - Safety level configuration
   - Code introspection
   - Improvement proposals
   - Test generation and validation

## Installation

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
cd /path/to/MEMSHADOW
pip install -r requirements-web.txt
```

### Dependencies Include:
- FastAPI - Modern web framework
- Uvicorn - ASGI server
- Jinja2 - Template engine
- PyJWT - Authentication
- Structlog - Structured logging

## Running the Interface

### Quick Start

```bash
python run_web.py
```

The interface will be available at: `http://localhost:8000/`

### Custom Configuration

```bash
# Custom host/port
python run_web.py --host 0.0.0.0 --port 8080

# Enable auto-reload (development)
python run_web.py --reload

# Multiple workers (production)
python run_web.py --workers 4
```

### Production Deployment

For production use with Gunicorn:

```bash
gunicorn app.web.api:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - \
    --error-logfile -
```

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Configuration

Configuration is stored in `config/memshadow.json` and includes:

### System Configuration
```json
{
  "system": {
    "name": "MEMSHADOW",
    "version": "1.0.0",
    "environment": "production"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  }
}
```

### Security Configuration
```json
{
  "security": {
    "secret_key": "CHANGE_ME_IN_PRODUCTION",
    "token_expiry_hours": 24,
    "require_authentication": true,
    "tempest_class": "C",
    "audit_logging": true
  }
}
```

### Component Configuration

Each Phase 8 component has dedicated configuration:

```json
{
  "federated": {
    "enabled": false,
    "node_id": "memshadow_001",
    "privacy_budget": 1.0
  },
  "consciousness": {
    "enabled": false,
    "workspace_capacity": 7,
    "attention_heads": 8
  },
  "self_modifying": {
    "enabled": false,
    "safety_level": "read_only",
    "enable_auto_apply": false
  }
}
```

## TEMPEST CLASS C Theme

The interface uses a military-grade TEMPEST CLASS C design standard:

### Visual Features
- **Pure black background** (#000000) - Minimal EM emissions
- **Phosphor green text** (#00ff41) - High contrast CRT aesthetic
- **Monospace fonts** - Terminal-like appearance
- **CRT effects** - Scanlines and subtle flicker
- **Status indicators** - Color-coded operational states
- **Security classification** - Visible CLASS C designation

### Color Coding
- **Green** (#00ff41) - Operational/Success
- **Amber** (#ffaa00) - Warning/Degraded
- **Red** (#ff0000) - Alert/Offline
- **Cyan** (#00ffff) - Information
- **Magenta** (#ff00ff) - Critical

### Design Principles
1. **High Contrast** - All elements have >7:1 contrast ratio
2. **Monospace Typography** - Courier New/Consolas for technical precision
3. **Glow Effects** - Phosphor-style text glow
4. **Terminal Aesthetics** - Command-line inspired interface
5. **Electromagnetic Security** - Dark backgrounds minimize emissions

## API Endpoints

### System
- `GET /` - Main dashboard
- `GET /api/status` - System status
- `GET /api/stats` - Component statistics
- `GET /api/config` - Get configuration
- `PUT /api/config/{section}` - Update configuration

### Federated Learning
- `POST /api/federated/start` - Start coordinator
- `POST /api/federated/stop` - Stop coordinator
- `POST /api/federated/join` - Join federation
- `POST /api/federated/update` - Broadcast update
- `GET /api/federated/stats` - Get statistics
- `GET /api/federated/peers` - Get connected peers

### Meta-Learning
- `POST /api/meta-learning/adapt` - Adapt to task
- `GET /api/meta-learning/performance` - Get metrics
- `GET /api/meta-learning/proposals` - Get improvement proposals

### Consciousness
- `POST /api/consciousness/start` - Start integrator
- `POST /api/consciousness/stop` - Stop integrator
- `POST /api/consciousness/process` - Process consciously
- `GET /api/consciousness/state` - Get state
- `GET /api/consciousness/workspace` - Get workspace items

### Self-Modifying
- `POST /api/self-modifying/start` - Start engine
- `POST /api/self-modifying/stop` - Stop engine
- `POST /api/self-modifying/improve` - Request improvement
- `GET /api/self-modifying/status` - Get status

## Security

### Authentication

Currently implements JWT-based authentication:

```python
# Login to get token
POST /api/auth/login
{
  "username": "admin",
  "password": "admin"  # CHANGE IN PRODUCTION!
}

# Use token in requests
Authorization: Bearer <token>
```

**⚠️ IMPORTANT**: Change default credentials in production!

### Security Best Practices

1. **Change default credentials** immediately
2. **Use HTTPS** in production
3. **Set strong SECRET_KEY** in configuration
4. **Enable audit logging** for compliance
5. **Restrict network access** via firewall
6. **Regular security updates** for dependencies

### TEMPEST Compliance

The TEMPEST CLASS C theme provides:
- **Minimal electromagnetic emissions** (dark backgrounds)
- **High contrast** for secure viewing environments
- **No reflective elements** that could compromise security
- **Visual security classification** prominently displayed

## Usage Examples

### Starting Federated Learning

1. Navigate to `/federated`
2. Click "START COORDINATOR"
3. Add peer addresses (one per line):
   ```
   node_002:8000
   node_003:8000
   192.168.1.100:8000
   ```
4. Click "JOIN FEDERATION"
5. Configure privacy settings (ε=1.0 recommended)
6. Broadcast updates with privacy budgets

### Conscious Processing

1. Navigate to `/consciousness`
2. Click "START INTEGRATOR"
3. Send items for conscious processing
4. Monitor workspace utilization (7±2 items)
5. View attention allocation and confidence estimates

### Self-Modification (⚠️ Use with Caution)

1. Navigate to `/self-modifying`
2. Review **SAFETY WARNING**
3. Select safety level (default: READ_ONLY)
4. **Do not enable auto-apply** without thorough testing
5. Review all proposals before implementation

## Monitoring

### Dashboard Metrics

The main dashboard displays:
- System uptime
- Active component count
- Component status (Operational/Degraded/Offline)
- Real-time logs

### System Logs

Terminal-style log viewer shows:
- Component start/stop events
- Configuration changes
- Errors and warnings
- API requests

## Troubleshooting

### Port Already in Use

```bash
# Use different port
python run_web.py --port 8080
```

### Import Errors

```bash
# Ensure all dependencies installed
pip install -r requirements-web.txt

# Verify Python path
export PYTHONPATH=/path/to/MEMSHADOW:$PYTHONPATH
```

### Components Won't Start

Check configuration in `config/memshadow.json`:
- Verify paths are correct
- Check permissions
- Review logs for errors

### CSS/JS Not Loading

Ensure static file directories exist:
```bash
mkdir -p app/web/static/css
mkdir -p app/web/static/js
mkdir -p app/web/templates
```

## Development

### File Structure

```
app/web/
├── __init__.py
├── api.py              # FastAPI application
├── auth.py             # Authentication
├── config.py           # Configuration management
├── static/
│   ├── css/
│   │   └── tempest-class-c.css  # TEMPEST theme
│   └── js/
│       └── memshadow.js  # Client-side API
└── templates/
    ├── index.html      # Main dashboard
    ├── federated.html  # Federated learning
    ├── meta_learning.html
    ├── consciousness.html
    └── self_modifying.html
```

### Adding New Endpoints

1. Define endpoint in `api.py`:
```python
@app.get("/api/my-endpoint")
async def my_endpoint():
    return {"message": "Hello"}
```

2. Add client-side wrapper in `memshadow.js`:
```javascript
class MEMSHADOW_API {
    static async myEndpoint() {
        return this.get('/my-endpoint');
    }
}
```

3. Call from HTML:
```javascript
const data = await MEMSHADOW_API.myEndpoint();
```

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Interactive visualizations (charts, graphs)
- [ ] User management and role-based access
- [ ] Advanced configuration UI
- [ ] Export/import configuration
- [ ] Detailed component pages with full controls
- [ ] Performance monitoring dashboards
- [ ] Alert/notification system
- [ ] Multi-language support
- [ ] Dark/light theme toggle (while maintaining TEMPEST compliance)

## Support

For issues or questions:
- GitHub Issues: https://github.com/SWORDIntel/MEMSHADOW/issues
- Documentation: See `docs/` directory
- API Docs: http://localhost:8000/api/docs

## License

Same license as MEMSHADOW project.

---

**TEMPEST CLASS C** - Designed for secure, high-contrast operation environments.
