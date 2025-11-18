"""
Code Introspector
Phase 8.4: Code analysis and understanding

Analyzes code to understand:
- Structure and complexity
- Performance bottlenecks
- Code quality metrics
- Potential improvements
- Dependencies and side effects

Uses AST parsing for safe static analysis.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import ast
import inspect
from pathlib import Path
import structlog

logger = structlog.get_logger()


class CodeComplexity(Enum):
    """Code complexity levels"""
    TRIVIAL = "trivial"        # 1-5 lines, simple logic
    LOW = "low"                # 6-20 lines, straightforward
    MEDIUM = "medium"          # 21-50 lines, some branching
    HIGH = "high"              # 51-100 lines, complex logic
    VERY_HIGH = "very_high"    # 100+ lines, highly complex


@dataclass
class CodeMetrics:
    """Metrics for a piece of code"""
    file_path: str
    function_name: Optional[str] = None

    # Size metrics
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0

    # Complexity metrics
    cyclomatic_complexity: int = 0  # Number of independent paths
    cognitive_complexity: int = 0   # How hard to understand
    nesting_depth: int = 0

    # Quality metrics
    has_docstring: bool = False
    has_type_hints: bool = False
    has_tests: bool = False

    # Dependencies
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)

    # Potential issues
    potential_issues: List[str] = field(default_factory=list)

    @property
    def complexity_level(self) -> CodeComplexity:
        """Determine complexity level"""
        if self.lines_of_code <= 5:
            return CodeComplexity.TRIVIAL
        elif self.lines_of_code <= 20 and self.cyclomatic_complexity <= 5:
            return CodeComplexity.LOW
        elif self.lines_of_code <= 50 and self.cyclomatic_complexity <= 10:
            return CodeComplexity.MEDIUM
        elif self.lines_of_code <= 100:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH

    @property
    def quality_score(self) -> float:
        """Calculate quality score (0.0 to 1.0)"""
        score = 0.0

        # Docstring
        if self.has_docstring:
            score += 0.3

        # Type hints
        if self.has_type_hints:
            score += 0.2

        # Tests
        if self.has_tests:
            score += 0.3

        # Complexity penalty
        if self.complexity_level in [CodeComplexity.TRIVIAL, CodeComplexity.LOW]:
            score += 0.2
        elif self.complexity_level == CodeComplexity.MEDIUM:
            score += 0.1

        return min(1.0, score)


class CodeIntrospector:
    """
    Code Introspector for self-analysis.

    Analyzes Python code to understand structure, complexity,
    and potential improvements.

    Uses AST (Abstract Syntax Tree) for safe static analysis
    without executing code.

    Example:
        introspector = CodeIntrospector()

        # Analyze a function
        metrics = await introspector.analyze_function(my_function)

        print(f"Complexity: {metrics.complexity_level.value}")
        print(f"Quality: {metrics.quality_score:.2f}")

        # Analyze a file
        file_metrics = await introspector.analyze_file("module.py")
    """

    def __init__(self):
        """Initialize code introspector"""
        # Cache of analyzed code
        self.metrics_cache: Dict[str, CodeMetrics] = {}

        logger.info("Code introspector initialized")

    async def analyze_function(
        self,
        func: callable
    ) -> CodeMetrics:
        """
        Analyze a function.

        Args:
            func: Function to analyze

        Returns:
            Code metrics
        """
        func_name = func.__name__
        module = inspect.getmodule(func)
        file_path = inspect.getfile(func) if module else "unknown"

        # Check cache
        cache_key = f"{file_path}::{func_name}"
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]

        # Get source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            logger.warning(f"Could not get source for {func_name}")
            return CodeMetrics(file_path=file_path, function_name=func_name)

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error(f"Syntax error in {func_name}: {e}")
            return CodeMetrics(file_path=file_path, function_name=func_name)

        # Analyze
        metrics = self._analyze_ast(tree, file_path, func_name, source)

        # Check for docstring
        metrics.has_docstring = func.__doc__ is not None

        # Check for type hints
        metrics.has_type_hints = self._has_type_hints(func)

        # Cache
        self.metrics_cache[cache_key] = metrics

        logger.debug(
            "Function analyzed",
            function=func_name,
            complexity=metrics.complexity_level.value,
            quality=metrics.quality_score
        )

        return metrics

    async def analyze_file(
        self,
        file_path: str
    ) -> List[CodeMetrics]:
        """
        Analyze all functions in a file.

        Args:
            file_path: Path to Python file

        Returns:
            List of metrics for each function
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        # Read source
        source = path.read_text()

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []

        # Find all function definitions
        metrics_list = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_metrics = self._analyze_ast(
                    node, file_path, node.name, source
                )

                # Check for docstring
                func_metrics.has_docstring = ast.get_docstring(node) is not None

                metrics_list.append(func_metrics)

        logger.info(
            "File analyzed",
            file_path=file_path,
            functions_count=len(metrics_list)
        )

        return metrics_list

    async def find_bottlenecks(
        self,
        metrics_list: List[CodeMetrics]
    ) -> List[CodeMetrics]:
        """
        Identify performance bottlenecks.

        Args:
            metrics_list: List of code metrics

        Returns:
            Metrics for functions likely to be bottlenecks
        """
        bottlenecks = []

        for metrics in metrics_list:
            # High complexity = potential bottleneck
            if metrics.complexity_level in [CodeComplexity.HIGH, CodeComplexity.VERY_HIGH]:
                bottlenecks.append(metrics)

            # Deep nesting = potential inefficiency
            elif metrics.nesting_depth > 4:
                bottlenecks.append(metrics)

        return bottlenecks

    async def suggest_refactoring(
        self,
        metrics: CodeMetrics
    ) -> List[str]:
        """
        Suggest refactoring opportunities.

        Args:
            metrics: Code metrics

        Returns:
            List of refactoring suggestions
        """
        suggestions = []

        # Missing docstring
        if not metrics.has_docstring:
            suggestions.append("Add docstring to explain purpose and parameters")

        # Missing type hints
        if not metrics.has_type_hints:
            suggestions.append("Add type hints for better code clarity")

        # High complexity
        if metrics.cyclomatic_complexity > 10:
            suggestions.append(
                f"Reduce cyclomatic complexity ({metrics.cyclomatic_complexity}) "
                "by extracting functions"
            )

        # Deep nesting
        if metrics.nesting_depth > 3:
            suggestions.append(
                f"Reduce nesting depth ({metrics.nesting_depth}) "
                "by using early returns or extracting functions"
            )

        # Too long
        if metrics.lines_of_code > 50:
            suggestions.append(
                f"Function is long ({metrics.lines_of_code} lines) - "
                "consider breaking into smaller functions"
            )

        return suggestions

    # Private methods

    def _analyze_ast(
        self,
        tree: ast.AST,
        file_path: str,
        func_name: str,
        source: str
    ) -> CodeMetrics:
        """Analyze AST and extract metrics"""
        metrics = CodeMetrics(
            file_path=file_path,
            function_name=func_name
        )

        # Count lines
        lines = source.split('\n')
        metrics.lines_of_code = len([l for l in lines if l.strip()])
        metrics.lines_of_comments = len([l for l in lines if l.strip().startswith('#')])
        metrics.blank_lines = len([l for l in lines if not l.strip()])

        # Cyclomatic complexity
        metrics.cyclomatic_complexity = self._calculate_cyclomatic(tree)

        # Nesting depth
        metrics.nesting_depth = self._calculate_nesting_depth(tree)

        # Imports and calls
        metrics.imports = self._extract_imports(tree)
        metrics.calls = self._extract_calls(tree)

        # Dependencies
        metrics.dependencies = set(metrics.imports)

        return metrics

    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1

            # Exception handlers
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1

            # Boolean operations
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0

        def visit(node: ast.AST, depth: int):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            # Nodes that increase nesting
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With,
                                 ast.AsyncFor, ast.AsyncWith, ast.Try)):
                depth += 1

            for child in ast.iter_child_nodes(node):
                visit(child, depth)

        visit(tree, 0)
        return max_depth

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _extract_calls(self, tree: ast.AST) -> List[str]:
        """Extract function calls"""
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        return calls

    def _has_type_hints(self, func: callable) -> bool:
        """Check if function has type hints"""
        try:
            sig = inspect.signature(func)

            # Check return annotation
            if sig.return_annotation == inspect.Signature.empty:
                return False

            # Check parameter annotations
            for param in sig.parameters.values():
                if param.annotation == inspect.Parameter.empty:
                    # Ignore 'self' and 'cls'
                    if param.name not in ['self', 'cls']:
                        return False

            return True

        except (ValueError, TypeError):
            return False
