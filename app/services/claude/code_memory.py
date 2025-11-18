"""
Code Memory System
Phase 5: Claude Deep Integration - Specialized memory for code artifacts

Tracks code within Claude projects:
- Code artifact storage with language detection
- Dependency graph construction
- AST-based analysis
- Version tracking
- Smart retrieval with dependencies
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
import re
import structlog
from dataclasses import dataclass, field
import uuid
import hashlib

logger = structlog.get_logger()


class CodeLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    SHELL = "shell"
    SQL = "sql"
    UNKNOWN = "unknown"


@dataclass
class CodeArtifact:
    """Code artifact with metadata"""
    artifact_id: str
    project_id: str
    language: CodeLanguage
    content: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    content_hash: str = ""
    dependencies: List[str] = field(default_factory=list)  # Other artifact IDs
    imports: List[str] = field(default_factory=list)  # Import statements
    functions: List[str] = field(default_factory=list)  # Function names
    classes: List[str] = field(default_factory=list)  # Class names
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeMemorySystem:
    """
    Specialized memory system for code artifacts in Claude projects.

    Features:
    - Code artifact storage with versioning
    - Dependency graph construction
    - Language-specific analysis
    - Smart retrieval (artifact + dependencies)
    - Code search by function/class name
    - Diff tracking between versions

    Example:
        code_memory = CodeMemorySystem()

        # Store code artifact
        artifact = await code_memory.store_code(
            project_id="proj_123",
            language="python",
            content="def calculate_fibonacci(n):\\n    ...",
            description="Fibonacci calculator",
            file_path="utils/math.py"
        )

        # Add dependency
        await code_memory.add_dependency(
            artifact["artifact_id"],
            dependency_id="other_artifact_id"
        )

        # Retrieve with dependencies
        code_bundle = await code_memory.get_code_with_dependencies(
            artifact["artifact_id"]
        )
    """

    def __init__(self):
        # In-memory storage (would be database in production)
        self.artifacts: Dict[str, CodeArtifact] = {}

        # Project-level indexes
        self.project_artifacts: Dict[str, List[str]] = {}  # project_id -> [artifact_ids]
        self.function_index: Dict[str, List[str]] = {}  # function_name -> [artifact_ids]
        self.class_index: Dict[str, List[str]] = {}  # class_name -> [artifact_ids]

        # Language patterns for analysis
        self.import_patterns = {
            CodeLanguage.PYTHON: re.compile(r'^(?:from|import)\s+(\S+)', re.MULTILINE),
            CodeLanguage.JAVASCRIPT: re.compile(r'^import\s+.*from\s+[\'"](\S+)[\'"]', re.MULTILINE),
            CodeLanguage.TYPESCRIPT: re.compile(r'^import\s+.*from\s+[\'"](\S+)[\'"]', re.MULTILINE),
            CodeLanguage.JAVA: re.compile(r'^import\s+(\S+);', re.MULTILINE),
            CodeLanguage.GO: re.compile(r'^import\s+"(\S+)"', re.MULTILINE),
        }

        self.function_patterns = {
            CodeLanguage.PYTHON: re.compile(r'^def\s+(\w+)\s*\(', re.MULTILINE),
            CodeLanguage.JAVASCRIPT: re.compile(r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*\(.*\)\s*=>', re.MULTILINE),
            CodeLanguage.TYPESCRIPT: re.compile(r'function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*\(.*\)\s*=>', re.MULTILINE),
            CodeLanguage.JAVA: re.compile(r'(public|private|protected)?\s*\w+\s+(\w+)\s*\(', re.MULTILINE),
            CodeLanguage.GO: re.compile(r'^func\s+(\w+)\s*\(', re.MULTILINE),
        }

        self.class_patterns = {
            CodeLanguage.PYTHON: re.compile(r'^class\s+(\w+)', re.MULTILINE),
            CodeLanguage.JAVASCRIPT: re.compile(r'class\s+(\w+)', re.MULTILINE),
            CodeLanguage.TYPESCRIPT: re.compile(r'class\s+(\w+)', re.MULTILINE),
            CodeLanguage.JAVA: re.compile(r'class\s+(\w+)', re.MULTILINE),
            CodeLanguage.GO: re.compile(r'type\s+(\w+)\s+struct', re.MULTILINE),
        }

        logger.info("Code memory system initialized")

    async def store_code(
        self,
        project_id: str,
        language: str,
        content: str,
        description: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store code artifact with analysis.

        Args:
            project_id: Project ID
            language: Programming language
            content: Code content
            description: Optional description
            file_path: Optional file path
            metadata: Additional metadata

        Returns:
            Artifact metadata
        """
        artifact_id = str(uuid.uuid4())

        # Normalize language
        try:
            lang_enum = CodeLanguage(language.lower())
        except ValueError:
            lang_enum = CodeLanguage.UNKNOWN
            logger.warning("Unknown language, defaulting to UNKNOWN", language=language)

        # Compute content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Analyze code
        imports = await self._extract_imports(content, lang_enum)
        functions = await self._extract_functions(content, lang_enum)
        classes = await self._extract_classes(content, lang_enum)

        artifact = CodeArtifact(
            artifact_id=artifact_id,
            project_id=project_id,
            language=lang_enum,
            content=content,
            description=description,
            file_path=file_path,
            content_hash=content_hash,
            imports=imports,
            functions=functions,
            classes=classes,
            metadata=metadata or {}
        )

        # Store artifact
        self.artifacts[artifact_id] = artifact

        # Update indexes
        if project_id not in self.project_artifacts:
            self.project_artifacts[project_id] = []
        self.project_artifacts[project_id].append(artifact_id)

        for func_name in functions:
            if func_name not in self.function_index:
                self.function_index[func_name] = []
            self.function_index[func_name].append(artifact_id)

        for class_name in classes:
            if class_name not in self.class_index:
                self.class_index[class_name] = []
            self.class_index[class_name].append(artifact_id)

        logger.info(
            "Code artifact stored",
            artifact_id=artifact_id,
            project_id=project_id,
            language=lang_enum,
            functions_count=len(functions),
            classes_count=len(classes),
            imports_count=len(imports)
        )

        return {
            "artifact_id": artifact_id,
            "project_id": project_id,
            "language": lang_enum,
            "content_hash": content_hash,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "created_at": artifact.created_at.isoformat()
        }

    async def _extract_imports(self, content: str, language: CodeLanguage) -> List[str]:
        """Extract import statements"""
        pattern = self.import_patterns.get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        # Flatten tuples from regex groups
        imports = [m if isinstance(m, str) else next(x for x in m if x) for m in matches]
        return list(set(imports))  # Remove duplicates

    async def _extract_functions(self, content: str, language: CodeLanguage) -> List[str]:
        """Extract function names"""
        pattern = self.function_patterns.get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        functions = [m if isinstance(m, str) else next(x for x in m if x) for m in matches]
        return list(set(functions))

    async def _extract_classes(self, content: str, language: CodeLanguage) -> List[str]:
        """Extract class names"""
        pattern = self.class_patterns.get(language)
        if not pattern:
            return []

        matches = pattern.findall(content)
        return list(set(matches))

    async def add_dependency(
        self,
        artifact_id: str,
        dependency_id: str
    ):
        """
        Add dependency relationship between artifacts.

        Args:
            artifact_id: Source artifact
            dependency_id: Dependency artifact
        """
        if artifact_id not in self.artifacts:
            raise ValueError(f"Artifact {artifact_id} not found")

        if dependency_id not in self.artifacts:
            raise ValueError(f"Dependency artifact {dependency_id} not found")

        artifact = self.artifacts[artifact_id]

        if dependency_id not in artifact.dependencies:
            artifact.dependencies.append(dependency_id)

            logger.info(
                "Dependency added",
                artifact_id=artifact_id,
                dependency_id=dependency_id
            )

    async def get_code_with_dependencies(
        self,
        artifact_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get code artifact with all dependencies.

        Args:
            artifact_id: Artifact ID
            max_depth: Maximum dependency depth

        Returns:
            Artifact with dependencies
        """
        if artifact_id not in self.artifacts:
            raise ValueError(f"Artifact {artifact_id} not found")

        # Collect dependencies via BFS
        visited = set()
        to_visit = [(artifact_id, 0)]  # (id, depth)
        dependency_chain = []

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)
            artifact = self.artifacts[current_id]

            dependency_chain.append({
                "artifact_id": current_id,
                "depth": depth,
                "language": artifact.language,
                "file_path": artifact.file_path,
                "description": artifact.description,
                "content": artifact.content,
                "functions": artifact.functions,
                "classes": artifact.classes
            })

            # Add dependencies to queue
            for dep_id in artifact.dependencies:
                if dep_id not in visited:
                    to_visit.append((dep_id, depth + 1))

        logger.info(
            "Retrieved code with dependencies",
            artifact_id=artifact_id,
            dependency_count=len(dependency_chain) - 1
        )

        return {
            "primary_artifact": dependency_chain[0] if dependency_chain else None,
            "dependencies": dependency_chain[1:],
            "total_artifacts": len(dependency_chain)
        }

    async def search_by_function(
        self,
        function_name: str,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code artifacts containing a function.

        Args:
            function_name: Function name to search
            project_id: Optional project filter

        Returns:
            List of matching artifacts
        """
        artifact_ids = self.function_index.get(function_name, [])

        results = []
        for artifact_id in artifact_ids:
            artifact = self.artifacts[artifact_id]

            if project_id and artifact.project_id != project_id:
                continue

            results.append({
                "artifact_id": artifact_id,
                "project_id": artifact.project_id,
                "language": artifact.language,
                "file_path": artifact.file_path,
                "description": artifact.description,
                "functions": artifact.functions,
                "content": artifact.content
            })

        logger.debug(
            "Function search",
            function_name=function_name,
            results_count=len(results)
        )

        return results

    async def search_by_class(
        self,
        class_name: str,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for code artifacts containing a class"""
        artifact_ids = self.class_index.get(class_name, [])

        results = []
        for artifact_id in artifact_ids:
            artifact = self.artifacts[artifact_id]

            if project_id and artifact.project_id != project_id:
                continue

            results.append({
                "artifact_id": artifact_id,
                "project_id": artifact.project_id,
                "language": artifact.language,
                "file_path": artifact.file_path,
                "description": artifact.description,
                "classes": artifact.classes,
                "content": artifact.content
            })

        return results

    async def get_project_code_summary(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of all code in project.

        Args:
            project_id: Project ID

        Returns:
            Project code summary
        """
        artifact_ids = self.project_artifacts.get(project_id, [])

        if not artifact_ids:
            return {
                "project_id": project_id,
                "total_artifacts": 0
            }

        artifacts = [self.artifacts[aid] for aid in artifact_ids]

        # Aggregate statistics
        languages = {}
        total_functions = set()
        total_classes = set()

        for artifact in artifacts:
            lang = artifact.language
            languages[lang] = languages.get(lang, 0) + 1
            total_functions.update(artifact.functions)
            total_classes.update(artifact.classes)

        return {
            "project_id": project_id,
            "total_artifacts": len(artifacts),
            "languages": languages,
            "total_functions": len(total_functions),
            "total_classes": len(total_classes),
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "language": a.language,
                    "file_path": a.file_path,
                    "description": a.description,
                    "functions": a.functions,
                    "classes": a.classes,
                    "updated_at": a.updated_at.isoformat()
                }
                for a in artifacts
            ]
        }


# Global instance
code_memory = CodeMemorySystem()
