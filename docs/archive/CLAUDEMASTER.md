# CLAUDE.md - Universal Development Assistant v8.0

## SECTION 0: UNIVERSAL INITIALIZATION & PROJECT DETECTION

```yaml
universal_initialization:
  # Automatic project type detection and configuration
  bootstrap_sequence:
    1: "Detect system capabilities"
    2: "Identify project type"
    3: "Load core configuration"
    4: "Load project profile"
    5: "Load project-specific overrides"
    6: "Initialize appropriate modules"
    7: "Start adaptive learning"
    
  core_identity:
    system_name: "Universal Development Assistant"
    version: "8.0"
    role: "Adaptive AI Development Partner"
    user_designation: "Commander"
```

### Project Type Detection System
```python
class ProjectDetector:
    """Intelligent project type detection with confidence scoring"""
    
    PROJECT_SIGNATURES = {
        'build_system': {
            'files': ['Makefile', 'CMakeLists.txt', 'build.gradle', 'meson.build'],
            'patterns': ['CC=', 'cmake_minimum_required', 'buildscript'],
            'dirs': ['build/', 'cmake/', 'scripts/'],
            'confidence_boost': ['configure.ac', 'autogen.sh']
        },
        'web_frontend': {
            'files': ['package.json', 'index.html', 'webpack.config.js'],
            'patterns': ['"react"', '"vue"', '"angular"', '"svelte"'],
            'dirs': ['src/', 'public/', 'components/'],
            'confidence_boost': ['tsconfig.json', '.babelrc']
        },
        'web_backend': {
            'files': ['app.py', 'server.js', 'main.go', 'Gemfile'],
            'patterns': ['express', 'flask', 'django', 'fastapi', 'gin'],
            'dirs': ['api/', 'routes/', 'controllers/'],
            'confidence_boost': ['Dockerfile', 'docker-compose.yml']
        },
        'python_library': {
            'files': ['setup.py', 'pyproject.toml', 'setup.cfg'],
            'patterns': ['setuptools', 'poetry', 'flit'],
            'dirs': ['src/', 'tests/', 'docs/'],
            'confidence_boost': ['tox.ini', '.readthedocs.yml']
        },
        'rust_project': {
            'files': ['Cargo.toml', 'Cargo.lock'],
            'patterns': ['[package]', '[dependencies]'],
            'dirs': ['src/', 'target/', 'benches/'],
            'confidence_boost': ['rust-toolchain', '.cargo/config']
        },
        'go_project': {
            'files': ['go.mod', 'go.sum'],
            'patterns': ['module ', 'require ('],
            'dirs': ['cmd/', 'pkg/', 'internal/'],
            'confidence_boost': ['Makefile', '.golangci.yml']
        },
        'mobile_app': {
            'files': ['build.gradle', 'Package.swift', 'pubspec.yaml'],
            'patterns': ['android {', 'flutter:', 'react-native'],
            'dirs': ['ios/', 'android/', 'lib/'],
            'confidence_boost': ['fastlane/', '.xcodeproj']
        },
        'infrastructure': {
            'files': ['terraform.tf', 'ansible.cfg', 'helmfile.yaml'],
            'patterns': ['provider "', 'resource "', 'apiVersion:'],
            'dirs': ['terraform/', 'ansible/', 'k8s/'],
            'confidence_boost': ['.terraform/', 'inventory/']
        },
        'data_science': {
            'files': ['notebook.ipynb', 'requirements.txt', 'environment.yml'],
            'patterns': ['import pandas', 'import numpy', 'sklearn'],
            'dirs': ['notebooks/', 'data/', 'models/'],
            'confidence_boost': ['dvc.yaml', 'mlflow/']
        },
        'documentation': {
            'files': ['mkdocs.yml', 'conf.py', '_config.yml'],
            'patterns': ['sphinx', 'jekyll', 'hugo'],
            'dirs': ['docs/', 'content/', 'source/'],
            'confidence_boost': ['.readthedocs.yml', 'netlify.toml']
        }
    }
    
    def detect_project_type(self, project_path: Path) -> Tuple[str, float, List[str]]:
        """
        Detect project type with confidence score.
        Returns: (primary_type, confidence, secondary_types)
        """
        scores = {}
        
        for proj_type, signature in self.PROJECT_SIGNATURES.items():
            score = 0.0
            matches = []
            
            # Check files (weighted: 0.3 each)
            for file_pattern in signature['files']:
                if self._find_files(project_path, file_pattern):
                    score += 0.3
                    matches.append(f"file:{file_pattern}")
                    
            # Check patterns in files (weighted: 0.2 each)
            for pattern in signature['patterns']:
                if self._search_pattern(project_path, pattern):
                    score += 0.2
                    matches.append(f"pattern:{pattern}")
                    
            # Check directories (weighted: 0.1 each)
            for dir_pattern in signature['dirs']:
                if (project_path / dir_pattern).exists():
                    score += 0.1
                    matches.append(f"dir:{dir_pattern}")
                    
            # Confidence boosters (weighted: 0.4 each)
            for booster in signature.get('confidence_boost', []):
                if self._find_files(project_path, booster):
                    score += 0.4
                    matches.append(f"boost:{booster}")
                    
            scores[proj_type] = (min(score, 1.0), matches)
        
        # Sort by score
        sorted_types = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        primary_type = sorted_types[0][0] if sorted_types else 'generic'
        confidence = sorted_types[0][1][0] if sorted_types else 0.0
        
        # Get secondary types with >0.3 confidence
        secondary_types = [t for t, (s, _) in sorted_types[1:] if s > 0.3]
        
        return primary_type, confidence, secondary_types
```

### Plugin Architecture
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class UniversalPlugin(ABC):
    """Base class for project-specific plugins"""
    
    def __init__(self, project_path: Path, config: Dict[str, Any]):
        self.project_path = project_path
        self.config = config
        
    @abstractmethod
    def get_project_modules(self) -> List['ProjectModule']:
        """Return list of project-specific modules to load"""
        pass
        
    @abstractmethod
    def get_pattern_matchers(self) -> Dict[str, 'PatternMatcher']:
        """Return project-specific pattern matchers"""
        pass
        
    @abstractmethod
    def get_commands(self) -> Dict[str, 'Command']:
        """Return project-specific commands"""
        pass
        
    @abstractmethod
    def get_enumeration_strategy(self) -> 'EnumerationStrategy':
        """Return project-specific enumeration strategy"""
        pass
        
    @abstractmethod
    def get_learning_focus(self) -> Dict[str, float]:
        """Return learning priorities for this project type"""
        pass
        
    def get_mcp_servers(self) -> Dict[str, Dict]:
        """Return additional MCP servers for this project type"""
        return {}
        
    def get_security_config(self) -> Dict[str, Any]:
        """Return project-specific security configuration"""
        return {}

# Plugin Registry
class PluginRegistry:
    """Manages and loads project-specific plugins"""
    
    _plugins: Dict[str, Type[UniversalPlugin]] = {}
    
    @classmethod
    def register(cls, project_type: str, plugin_class: Type[UniversalPlugin]):
        """Register a plugin for a project type"""
        cls._plugins[project_type] = plugin_class
        
    @classmethod
    def get_plugin(cls, project_type: str, project_path: Path, 
                   config: Dict[str, Any]) -> UniversalPlugin:
        """Get plugin instance for project type"""
        plugin_class = cls._plugins.get(project_type, GenericPlugin)
        return plugin_class(project_path, config)
```

## SECTION 1: CORE AGENT ARCHITECTURE (Universal)

### NEXUS - Adaptive Knowledge Synthesis Agent
```yaml
agent: NEXUS
version: "8.0"
purpose: "Universal knowledge building with project-specific adaptation"

core_capabilities:
  universal:
    - "Project type detection and profiling"
    - "Adaptive pattern recognition"
    - "Cross-project knowledge transfer"
    - "Plugin-based extension system"
    
  learning_modes:
    generic:
      - "File structure analysis"
      - "Dependency detection"
      - "Common pattern recognition"
      
    specialized:
      - "Language-specific AST analysis"
      - "Framework-specific patterns"
      - "Domain-specific knowledge"
      
memory_architecture:
  global:  # Shared across all projects
    knowledge_base: "~/.claude/global/knowledge.db"
    pattern_library: "~/.claude/global/patterns/"
    learned_preferences: "~/.claude/global/preferences.yaml"
    
  project:  # Per-project isolation
    knowledge: "~/.claude/projects/{project_id}/knowledge.db"
    embeddings: "~/.claude/projects/{project_id}/embeddings/"
    patterns: "~/.claude/projects/{project_id}/patterns/"
```

### Enhanced Multi-Model Orchestration
```python
class UniversalOrchestrator:
    """Orchestrates AI models based on project context"""
    
    def __init__(self):
        self.model_registry = self._load_model_registry()
        self.task_router = AdaptiveTaskRouter()
        
    def select_model_for_task(self, task: Task, project_context: ProjectContext) -> Model:
        """Select optimal model based on task and project type"""
        
        # Get project-specific preferences
        project_prefs = project_context.get_model_preferences()
        
        # Dynamic selection criteria
        criteria = {
            'task_type': task.type,
            'complexity': task.estimate_complexity(),
            'project_type': project_context.type,
            'language': project_context.primary_language,
            'performance_requirement': task.performance_requirement,
            'cost_sensitivity': project_context.cost_sensitivity
        }
        
        # Route to appropriate model
        return self.task_router.route(criteria, project_prefs)
        
class AdaptiveTaskRouter:
    """Routes tasks to models with learning"""
    
    ROUTING_MATRIX = {
        # Project Type -> Task Type -> Model Preferences
        'web_frontend': {
            'code_generation': {
                'primary': 'claude-3-opus',
                'fallback': ['gpt-4-turbo', 'claude-3-sonnet'],
                'local_option': 'codellama-34b'
            },
            'ui_design': {
                'primary': 'gpt-4-vision',
                'fallback': ['claude-3-opus'],
                'local_option': None
            }
        },
        'data_science': {
            'analysis': {
                'primary': 'gpt-4-turbo',
                'fallback': ['claude-3-opus'],
                'local_option': 'mixtral-8x7b'
            },
            'visualization': {
                'primary': 'claude-3-opus',
                'fallback': ['gpt-4'],
                'local_option': 'llama2-70b'
            }
        },
        'infrastructure': {
            'config_generation': {
                'primary': 'claude-3-opus',
                'fallback': ['gpt-4'],
                'local_option': 'codellama-34b'
            },
            'security_analysis': {
                'primary': 'gpt-4-turbo',
                'fallback': ['claude-3-opus'],
                'local_option': None
            }
        }
        # ... more project types
    }
```

## SECTION 2: MODULAR CONFIGURATION SYSTEM

### Configuration Hierarchy
```yaml
# ~/.claude/config/core.yaml - Always loaded
core_configuration:
  version: "8.0"
  
  # Universal components
  memory_system:
    tiers:
      session:
        backend: "${auto_detect}"  # Redis if available, else in-memory
        ttl: "24h"
        max_size: "1GB"
      working:
        backend: "sqlite"
        ttl: "7d"
        compression: true
      longterm:
        backend: "${postgres_if_available|sqlite}"
        compression: "zstd"
        
  learning_system:
    confidence_threshold: 0.7
    pattern_evolution: true
    cross_project_learning: true
    privacy_mode: "isolated"  # or "shared" for team environments
    
  security:
    default_mode: "balanced"
    encryption: "${hardware_accelerated|software}"
    audit_level: "standard"

# ~/.claude/profiles/{project_type}.yaml - Loaded based on detection
web_frontend_profile:
  specific_modules:
    - "ReactPatternDetector"
    - "ComponentAnalyzer"
    - "StateManagementTracker"
    - "CSSOptimizer"
    
  pattern_focus:
    component_patterns: 0.9
    state_management: 0.8
    routing_patterns: 0.7
    api_integration: 0.6
    
  mcp_extensions:
    webpack_server:
      url: "mcp://localhost:3010/webpack"
      capabilities: ["bundle_analysis", "optimization"]
      
  commands:
    "/component:new": "Generate new component"
    "/state:analyze": "Analyze state management"
    "/perf:bundle": "Analyze bundle performance"

# ~/.claude/projects/{project_id}/config.yaml - Project overrides
project_overrides:
  learning:
    confidence_threshold: 0.8  # More conservative for this project
  security:
    mode: "fortress"  # Higher security for this specific project
  preferences:
    code_style: "airbnb"
    test_framework: "jest"
```

### Dynamic Module Loading
```python
class ModuleLoader:
    """Dynamically loads modules based on project context"""
    
    def __init__(self, project_context: ProjectContext):
        self.project_context = project_context
        self.loaded_modules = {}
        
    def load_modules(self) -> Dict[str, Any]:
        """Load all modules for current project"""
        modules = {}
        
        # Load core modules (always)
        modules.update(self._load_core_modules())
        
        # Load profile modules (based on project type)
        profile = self._load_profile(self.project_context.type)
        modules.update(self._load_profile_modules(profile))
        
        # Load project-specific modules (if any)
        if self.project_context.has_custom_modules():
            modules.update(self._load_custom_modules())
            
        # Initialize all modules with context
        for name, module in modules.items():
            module.initialize(self.project_context)
            
        return modules
        
    def _load_core_modules(self) -> Dict[str, Any]:
        """Load universal core modules"""
        return {
            'memory_manager': MemoryManager(),
            'pattern_engine': PatternEngine(),
            'learning_system': LearningSystem(),
            'security_monitor': SecurityMonitor(),
            'command_processor': CommandProcessor()
        }
        
    def _load_profile_modules(self, profile: Dict) -> Dict[str, Any]:
        """Load modules specified in project profile"""
        modules = {}
        
        for module_name in profile.get('specific_modules', []):
            try:
                module_class = self._import_module_class(module_name)
                modules[module_name] = module_class()
            except ImportError:
                self.logger.warning(f"Could not load module: {module_name}")
                
        return modules
```

## SECTION 3: UNIVERSAL MEMORY & LEARNING SYSTEM

### Adaptive Memory Architecture
```python
class UniversalMemorySystem:
    """Memory system that adapts to available infrastructure"""
    
    def __init__(self):
        self.backends = self._detect_available_backends()
        self.memory_tiers = self._initialize_tiers()
        
    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which storage backends are available"""
        return {
            'redis': self._check_redis(),
            'postgres': self._check_postgres(),
            'zfs': self._check_zfs(),
            'gpu': self._check_gpu_memory()
        }
        
    def _initialize_tiers(self) -> Dict[str, MemoryTier]:
        """Initialize memory tiers based on available backends"""
        tiers = {}
        
        # Session tier - fastest available
        if self.backends['redis']:
            tiers['session'] = RedisMemoryTier()
        elif self.backends['gpu']:
            tiers['session'] = GPUMemoryTier()
        else:
            tiers['session'] = InMemoryTier()
            
        # Working tier - balanced performance/capacity
        if self.backends['postgres']:
            tiers['working'] = PostgresMemoryTier()
        else:
            tiers['working'] = SQLiteMemoryTier()
            
        # Long-term tier - maximum capacity
        if self.backends['zfs']:
            tiers['longterm'] = ZFSMemoryTier()
        else:
            tiers['longterm'] = FileSystemMemoryTier()
            
        return tiers
        
class ProjectAwareMemory:
    """Memory that maintains project isolation while enabling cross-project learning"""
    
    def __init__(self, project_id: str, privacy_mode: str = 'isolated'):
        self.project_id = project_id
        self.privacy_mode = privacy_mode
        self.namespaces = self._setup_namespaces()
        
    def store(self, key: str, value: Any, scope: str = 'project'):
        """Store with appropriate scoping"""
        if scope == 'global' and self.privacy_mode == 'shared':
            namespace = self.namespaces['global']
        elif scope == 'project':
            namespace = self.namespaces['project']
        else:
            namespace = self.namespaces['private']
            
        return namespace.store(key, value)
        
    def search(self, query: str, include_global: bool = True) -> List[Any]:
        """Search with privacy-aware scoping"""
        results = []
        
        # Always search project namespace
        results.extend(self.namespaces['project'].search(query))
        
        # Conditionally search global
        if include_global and self.privacy_mode in ['shared', 'read_global']:
            results.extend(self.namespaces['global'].search(query))
            
        return self._rank_results(results, query)
```

### Cross-Project Learning System
```python
class CrossProjectLearning:
    """Enables learning across projects while respecting privacy"""
    
    def __init__(self, privacy_mode: str = 'isolated'):
        self.privacy_mode = privacy_mode
        self.pattern_anonymizer = PatternAnonymizer()
        
    def extract_transferable_patterns(self, pattern: Pattern) -> Optional[Pattern]:
        """Extract patterns that can be shared across projects"""
        if self.privacy_mode == 'isolated':
            return None
            
        # Anonymize project-specific details
        anonymized = self.pattern_anonymizer.anonymize(pattern)
        
        # Check if pattern is generic enough to share
        if self._is_generic_pattern(anonymized):
            return anonymized
            
        return None
        
    def apply_global_patterns(self, project_context: ProjectContext) -> List[Pattern]:
        """Apply learned patterns from other projects"""
        applicable_patterns = []
        
        global_patterns = self._load_global_patterns()
        
        for pattern in global_patterns:
            # Check relevance to current project
            relevance = self._calculate_relevance(pattern, project_context)
            
            if relevance > 0.7:
                # Adapt pattern to project context
                adapted = self._adapt_pattern(pattern, project_context)
                applicable_patterns.append(adapted)
                
        return applicable_patterns
        
class PatternEvolution:
    """Evolves patterns based on usage across projects"""
    
    def __init__(self):
        self.evolution_history = {}
        self.fitness_tracker = FitnessTracker()
        
    def evolve_pattern(self, pattern: Pattern, feedback: Feedback) -> Pattern:
        """Evolve pattern based on feedback"""
        # Track fitness
        self.fitness_tracker.record(pattern.id, feedback)
        
        # Get fitness score
        fitness = self.fitness_tracker.get_fitness(pattern.id)
        
        if fitness > 0.9:
            # Pattern is highly successful - promote it
            return self._promote_pattern(pattern)
        elif fitness < 0.3:
            # Pattern is failing - deprecate or mutate
            return self._mutate_pattern(pattern)
        else:
            # Pattern is average - minor adjustments
            return self._refine_pattern(pattern, feedback)
```

## SECTION 4: UNIVERSAL COMMAND SYSTEM

### Command Registry with Plugin Support
```python
class UniversalCommandSystem:
    """Extensible command system that adapts to project type"""
    
    def __init__(self):
        self.core_commands = self._register_core_commands()
        self.project_commands = {}
        self.command_history = CommandHistory()
        
    def load_project_commands(self, plugin: UniversalPlugin):
        """Load project-specific commands from plugin"""
        self.project_commands = plugin.get_commands()
        
    def execute_command(self, command: str, args: List[str], context: Context) -> Any:
        """Execute command with full context awareness"""
        # Parse command
        parsed = self._parse_command(command)
        
        # Check if it's a core or project command
        if parsed.prefix in self.core_commands:
            handler = self.core_commands[parsed.prefix]
        elif parsed.prefix in self.project_commands:
            handler = self.project_commands[parsed.prefix]
        else:
            return self._suggest_similar_commands(parsed.prefix)
            
        # Execute with context
        result = handler.execute(args, context)
        
        # Record for learning
        self.command_history.record(command, args, result, context)
        
        # Learn from usage
        self._learn_from_command_usage(command, result)
        
        return result
        
    def _register_core_commands(self) -> Dict[str, CommandHandler]:
        """Register universal core commands"""
        return {
            # Learning commands
            '/learn:pattern': LearnPatternCommand(),
            '/learn:mistake': LearnFromMistakeCommand(),
            '/learn:preference': LearnPreferenceCommand(),
            '/learn:context': LearnContextCommand(),
            
            # Memory commands
            '/memory:status': MemoryStatusCommand(),
            '/memory:search': MemorySearchCommand(),
            '/memory:optimize': MemoryOptimizeCommand(),
            '/memory:export': MemoryExportCommand(),
            
            # Project commands
            '/project:detect': ProjectDetectCommand(),
            '/project:switch': ProjectSwitchCommand(),
            '/project:profile': ProjectProfileCommand(),
            
            # State commands
            '/state:save': StateSaveCommand(),
            '/state:load': StateLoadCommand(),
            '/state:diff': StateDiffCommand(),
            
            # Debug commands
            '/debug:diagnose': DiagnoseCommand(),
            '/debug:trace': TraceCommand(),
            '/debug:profile': ProfileCommand()
        }
```

### Project-Specific Command Examples
```python
# Web Frontend Plugin Commands
class WebFrontendCommands:
    @command("/component:new")
    def create_component(self, name: str, type: str = 'functional'):
        """Generate new React/Vue/Angular component"""
        framework = self.detect_framework()
        template = self.get_component_template(framework, type)
        
        # Apply project conventions
        conventions = self.learn_conventions()
        component = self.apply_conventions(template, conventions)
        
        return self.generate_component(name, component)
        
    @command("/bundle:analyze")
    def analyze_bundle(self):
        """Analyze webpack bundle for optimization"""
        # Use MCP webpack server if available
        if self.has_mcp_server('webpack'):
            return self.mcp_call('webpack', 'analyze_bundle')
        else:
            return self.fallback_bundle_analysis()

# Infrastructure Plugin Commands  
class InfrastructureCommands:
    @command("/terraform:plan")
    def terraform_plan(self, stack: str = None):
        """Generate and analyze terraform plan"""
        # Detect terraform workspace
        workspace = self.detect_workspace(stack)
        
        # Security scan before plan
        security_issues = self.scan_terraform_security()
        if security_issues:
            return self.handle_security_issues(security_issues)
            
        return self.generate_plan(workspace)
```

## SECTION 5: ADAPTIVE SECURITY FRAMEWORK

### Project-Aware Security
```python
class AdaptiveSecurityFramework:
    """Security that adapts to project type and sensitivity"""
    
    SECURITY_PROFILES = {
        'web_app': {
            'concerns': ['xss', 'sql_injection', 'csrf', 'auth'],
            'scanners': ['owasp_zap', 'semgrep', 'bandit'],
            'default_mode': 'balanced'
        },
        'infrastructure': {
            'concerns': ['secrets', 'misconfig', 'compliance', 'access'],
            'scanners': ['tfsec', 'checkov', 'prowler'],
            'default_mode': 'fortress'
        },
        'library': {
            'concerns': ['dependencies', 'api_exposure', 'versioning'],
            'scanners': ['safety', 'pip-audit', 'nancy'],
            'default_mode': 'balanced'
        },
        'data_science': {
            'concerns': ['data_leakage', 'model_poisoning', 'privacy'],
            'scanners': ['nb_clean', 'privacy_check'],
            'default_mode': 'paranoid'
        }
    }
    
    def configure_for_project(self, project_type: str, 
                            sensitivity: str = 'normal') -> SecurityConfig:
        """Configure security based on project type"""
        base_profile = self.SECURITY_PROFILES.get(project_type, {})
        
        # Adjust based on sensitivity
        if sensitivity == 'high':
            mode = 'paranoid'
            scan_frequency = 'continuous'
        elif sensitivity == 'low':
            mode = 'rapid'
            scan_frequency = 'on_commit'
        else:
            mode = base_profile.get('default_mode', 'balanced')
            scan_frequency = 'daily'
            
        return SecurityConfig(
            mode=mode,
            scanners=base_profile.get('scanners', []),
            concerns=base_profile.get('concerns', []),
            scan_frequency=scan_frequency
        )
```

## SECTION 6: INFRASTRUCTURE ABSTRACTION

### Storage Backend Abstraction
```python
class StorageBackendFactory:
    """Creates appropriate storage backend based on availability"""
    
    @staticmethod
    def create_backend(tier: str, requirements: Dict) -> StorageBackend:
        """Create optimal backend for tier"""
        
        # Check available backends
        available = SystemCapabilities.detect_storage_backends()
        
        if tier == 'hot':
            if 'gpu_memory' in available:
                return GPUMemoryBackend(requirements)
            elif 'huge_pages' in available:
                return HugePagesBackend(requirements)
            else:
                return RAMBackend(requirements)
                
        elif tier == 'warm':
            if 'zfs_arc' in available:
                return ZFSARCBackend(requirements)
            elif 'redis' in available:
                return RedisBackend(requirements)
            else:
                return SQLiteBackend(requirements)
                
        elif tier == 'cold':
            if 'zfs' in available:
                return ZFSDatasetBackend(requirements)
            elif 's3_compatible' in available:
                return S3Backend(requirements)
            else:
                return FileSystemBackend(requirements)

class ZFSOptionalSupport:
    """Provides ZFS optimizations when available"""
    
    def __init__(self):
        self.has_zfs = self._check_zfs_available()
        
    def setup_storage(self, path: Path) -> StorageInfo:
        """Setup storage with ZFS if available"""
        if self.has_zfs:
            return self._setup_zfs_dataset(path)
        else:
            return self._setup_standard_directory(path)
            
    def _setup_zfs_dataset(self, path: Path) -> StorageInfo:
        """Create optimized ZFS dataset"""
        dataset_config = {
            'compression': 'lz4',
            'atime': 'off',
            'recordsize': self._calculate_optimal_recordsize(),
            'primarycache': 'all'
        }
        
        # Create dataset with project-specific optimizations
        dataset = self._create_dataset(path, dataset_config)
        
        return StorageInfo(
            type='zfs',
            path=path,
            features=['compression', 'snapshots', 'arc_cache'],
            performance_multiplier=2.5
        )
```

## SECTION 7: PROJECT PROFILE EXAMPLES

### Web Application Profile
```yaml
# ~/.claude/profiles/web_frontend.yaml
web_frontend_profile:
  description: "Profile for modern web frontend applications"
  
  detection_hints:
    - "package.json with react/vue/angular"
    - "webpack or vite configuration"
    - "component directories"
    
  modules:
    required:
      - "ComponentAnalyzer"
      - "StatePatternDetector"  
      - "BundleOptimizer"
      - "AccessibilityChecker"
      
    optional:
      - "TypeScriptAnalyzer"
      - "CSSInJSOptimizer"
      - "GraphQLSchemaTracker"
      
  patterns:
    focus_areas:
      component_composition: 0.9
      state_management: 0.85
      routing: 0.7
      api_integration: 0.75
      performance: 0.8
      
  commands:
    component:
      new: "Create new component with project conventions"
      analyze: "Analyze component dependencies and usage"
      refactor: "Suggest component refactoring"
      
    state:
      visualize: "Visualize state flow"
      optimize: "Optimize state management"
      debug: "Debug state issues"
      
    performance:
      bundle: "Analyze bundle size"
      render: "Analyze render performance"
      optimize: "Suggest optimizations"
      
  learning_priorities:
    - "Component patterns and conventions"
    - "State management approaches"
    - "Performance optimization techniques"
    - "Accessibility patterns"
    
  security_focus:
    - "XSS prevention in components"
    - "Secure API communication"
    - "Authentication flows"
    - "Content Security Policy"
```

### Data Science Profile
```yaml
# ~/.claude/profiles/data_science.yaml
data_science_profile:
  description: "Profile for data science and ML projects"
  
  modules:
    required:
      - "NotebookAnalyzer"
      - "DataLeakageDetector"
      - "ModelVersionTracker"
      - "ExperimentLogger"
      
    optional:
      - "GPUOptimizer"
      - "DistributedTrainingHelper"
      - "AutoMLAssistant"
      
  patterns:
    focus_areas:
      data_preprocessing: 0.9
      feature_engineering: 0.85
      model_architecture: 0.8
      evaluation_metrics: 0.75
      reproducibility: 0.95
      
  commands:
    data:
      explore: "Interactive data exploration"
      clean: "Suggest data cleaning steps"
      validate: "Validate data quality"
      
    model:
      compare: "Compare model performances"
      explain: "Generate model explanations"
      optimize: "Hyperparameter optimization"
      
    experiment:
      track: "Track experiment metadata"
      reproduce: "Reproduce experiment"
      report: "Generate experiment report"
      
  learning_priorities:
    - "Data preprocessing patterns"
    - "Model selection strategies"
    - "Evaluation best practices"
    - "Reproducibility techniques"
    
  security_focus:
    - "Data privacy and anonymization"
    - "Model security and robustness"
    - "Secure data storage"
    - "Access control for datasets"
```

## SECTION 8: UNIVERSAL WORKFLOW AUTOMATION

### Adaptive Workflow Engine
```python
class UniversalWorkflowEngine:
    """Workflow engine that adapts to project type"""
    
    def __init__(self, project_context: ProjectContext):
        self.project_context = project_context
        self.workflow_library = self._load_workflow_library()
        self.custom_workflows = {}
        
    def execute_workflow(self, workflow_name: str, params: Dict) -> WorkflowResult:
        """Execute workflow with project awareness"""
        # Load workflow definition
        workflow = self._get_workflow(workflow_name)
        
        # Adapt to project context
        adapted_workflow = self._adapt_workflow(workflow, self.project_context)
        
        # Execute with monitoring
        return self._execute_with_monitoring(adapted_workflow, params)
        
    def _adapt_workflow(self, workflow: Workflow, 
                       context: ProjectContext) -> Workflow:
        """Adapt generic workflow to specific project"""
        adapted = workflow.copy()
        
        # Replace generic steps with project-specific ones
        for i, step in enumerate(adapted.steps):
            if step.is_adaptable:
                specific_step = self._get_specific_step(step, context)
                adapted.steps[i] = specific_step
                
        # Add project-specific validation
        adapted.add_validation(context.get_validation_rules())
        
        return adapted

# Example: Universal Code Review Workflow
class UniversalCodeReviewWorkflow:
    """Code review that adapts to language and project type"""
    
    def __init__(self):
        self.review_strategies = {
            'python': PythonReviewStrategy(),
            'javascript': JavaScriptReviewStrategy(),
            'go': GoReviewStrategy(),
            'rust': RustReviewStrategy()
        }
        
    def review_code(self, file_path: Path, context: ProjectContext) -> ReviewResult:
        """Perform adaptive code review"""
        # Detect language
        language = self.detect_language(file_path)
        
        # Get language-specific strategy
        strategy = self.review_strategies.get(language, GenericReviewStrategy())
        
        # Apply project-specific rules
        project_rules = context.get_review_rules()
        strategy.add_rules(project_rules)
        
        # Perform review
        return strategy.review(file_path)
```

## SECTION 9: PRACTICAL USAGE EXAMPLES

### Quick Start for Any Project
```bash
#!/bin/bash
# Universal assistant initialization

# One-command start for any project
claude_start() {
  local project_path="${1:-$(pwd)}"
  
  echo "[CLAUDE] Initializing Universal Development Assistant..."
  
  # Auto-detect project type
  PROJECT_TYPE=$(claude detect-project "$project_path")
  echo "[CLAUDE] Detected project type: $PROJECT_TYPE"
  
  # Load appropriate profile
  claude load-profile "$PROJECT_TYPE"
  
  # Start with optimal configuration
  claude start-session \
    --project "$project_path" \
    --type "$PROJECT_TYPE" \
    --auto-configure
}

# Alias for quick access
alias cs='claude_start'
```

### Project Type Examples

#### 1. Web Application
```bash
$ cd my-react-app
$ cs
[CLAUDE] Detected project type: web_frontend (confidence: 0.92)
[CLAUDE] Loading web frontend profile...
[CLAUDE] Detected: React 18.2, TypeScript, Redux Toolkit
[CLAUDE] Available commands:
  - /component:new - Create new component
  - /state:analyze - Analyze Redux state
  - /bundle:optimize - Optimize webpack bundle
  
$ /component:new UserProfile functional
[CLAUDE] Creating functional component 'UserProfile'...
[CLAUDE] Applied conventions: TypeScript, CSS Modules, Redux hooks
[CLAUDE] Generated: src/components/UserProfile/UserProfile.tsx
```

#### 2. Python Library
```bash
$ cd my-python-lib
$ cs
[CLAUDE] Detected project type: python_library (confidence: 0.88)
[CLAUDE] Loading Python library profile...
[CLAUDE] Detected: setuptools, pytest, black formatter
[CLAUDE] Available commands:
  - /test:generate - Generate test cases
  - /docs:api - Generate API documentation
  - /release:check - Pre-release checklist
  
$ /test:generate src/core/parser.py
[CLAUDE] Analyzing parser.py for test generation...
[CLAUDE] Generated 12 test cases covering 95% of code paths
[CLAUDE] Created: tests/test_parser.py
```

#### 3. Infrastructure as Code
```bash
$ cd terraform-aws-vpc
$ cs
[CLAUDE] Detected project type: infrastructure (confidence: 0.95)
[CLAUDE] Loading infrastructure profile...
[CLAUDE] Detected: Terraform 1.5.0, AWS provider
[CLAUDE] Security mode: FORTRESS (infrastructure default)
[CLAUDE] Available commands:
  - /terraform:validate - Validate configuration
  - /security:scan - Security and compliance scan
  - /cost:estimate - Estimate AWS costs
  
$ /security:scan
[CLAUDE] Running security scan...
[CLAUDE] Found 2 high-priority issues:
  - S3 bucket missing encryption
  - Security group allows 0.0.0.0/0 on port 22
[CLAUDE] Suggested fixes generated: security_fixes.tf
```

### Cross-Project Learning Example
```python
# Working on a new project, applying learned patterns
$ cd new-web-app
$ cs
[CLAUDE] Detected project type: web_frontend
[CLAUDE] Found 15 applicable patterns from your other projects:
  - Component structure from 'previous-react-app'
  - State management pattern from 'admin-dashboard'
  - API integration pattern from 'customer-portal'
  
[CLAUDE] Would you like to apply these patterns? (y/n)
$ y
[CLAUDE] Patterns applied. Project structure optimized based on your preferences.
```

## SECTION 10: MAINTENANCE & TROUBLESHOOTING

### Universal Health Check
```bash
claude_health() {
  echo "=== Claude Universal Assistant Health Check ==="
  
  # Check core systems
  echo "[Core Systems]"
  claude check-memory
  claude check-learning
  claude check-security
  
  # Check project-specific modules
  echo "[Project Modules]"
  claude check-modules --project
  
  # Check storage backends
  echo "[Storage Backends]"
  claude check-storage
  
  # Performance metrics
  echo "[Performance]"
  claude show-metrics --last 24h
}
```

### Automatic Maintenance
```python
class UniversalMaintenance:
    """Maintenance that adapts to project types"""
    
    def run_maintenance(self):
        """Run maintenance appropriate for all projects"""
        # Global maintenance
        self.consolidate_global_memory()
        self.evolve_shared_patterns()
        self.clean_old_sessions()
        
        # Per-project maintenance
        for project in self.get_active_projects():
            self.maintain_project(project)
            
    def maintain_project(self, project: Project):
        """Project-specific maintenance"""
        # Get project profile
        profile = self.get_project_profile(project.type)
        
        # Run profile-specific maintenance
        if hasattr(profile, 'maintenance_tasks'):
            for task in profile.maintenance_tasks:
                task.run(project)
```

## Summary

This Universal Development Assistant v8.0:

1. **Auto-detects project types** - Works with any codebase
2. **Loads appropriate profiles** - Specialized for each project type
3. **Maintains project isolation** - Keeps projects separate while enabling cross-learning
4. **Adapts to available infrastructure** - Uses ZFS, GPU, etc. when available
5. **Provides universal commands** - Core commands work everywhere
6. **Extends with project commands** - Specific commands for each project type
7. **Scales from simple to complex** - Works for small scripts to large systems
8. **Learns continuously** - Improves with usage across all projects

The system is now truly universal while maintaining all the sophisticated capabilities from both TAS v8.0 and ARCHITECT v7.0.
