#!/usr/bin/env python3
"""
Causal Inference Engine for DSMIL Brain

Performs causal reasoning:
- Counterfactual reasoning ("What if X hadn't happened?")
- Root cause analysis chains
- Intervention modeling ("If we do X, then Y")
- Causal graph distinct from associative
- Do-calculus implementation
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Tuple, Callable
from datetime import datetime, timezone
from enum import Enum, auto
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships"""
    CAUSES = auto()          # Direct causation
    ENABLES = auto()         # Enabling condition
    PREVENTS = auto()        # Prevention
    INCREASES_RISK = auto()  # Risk increase
    DECREASES_RISK = auto()  # Risk decrease
    CONFOUNDS = auto()       # Confounding variable


@dataclass
class CausalVariable:
    """A variable in the causal graph"""
    var_id: str
    name: str
    var_type: str  # "binary", "continuous", "categorical"

    # Possible values
    values: List[Any] = field(default_factory=list)

    # Observed data
    observed_value: Optional[Any] = None

    # Metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """Edge in causal graph"""
    source_id: str
    target_id: str
    relation: CausalRelationType
    strength: float = 0.5  # Causal strength

    # Conditional probability
    cpd: Optional[Dict[Any, float]] = None  # Conditional prob distribution

    # Evidence
    evidence_count: int = 0
    confidence: float = 0.5


@dataclass
class Intervention:
    """An intervention (do-operator)"""
    intervention_id: str
    variable_id: str
    set_value: Any

    # Predicted effects
    predicted_effects: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Counterfactual:
    """A counterfactual query result"""
    query_id: str

    # The counterfactual scenario
    factual: Dict[str, Any]      # What actually happened
    counterfactual: Dict[str, Any]  # Hypothetical scenario

    # Result
    probability: float
    explanation: str

    # Causal path
    causal_path: List[str] = field(default_factory=list)


@dataclass
class CausalQuery:
    """A query to the causal inference engine"""
    query_id: str
    query_type: str  # "effect", "cause", "counterfactual", "intervention"

    # Query specifics
    target_variable: str
    given_variables: Dict[str, Any] = field(default_factory=dict)
    intervention_variable: Optional[str] = None
    intervention_value: Optional[Any] = None

    # Result
    result: Optional[Any] = None
    confidence: float = 0.0


class CausalGraph:
    """
    Directed Acyclic Graph for causal relationships

    Represents causal structure separate from correlation.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._variables: Dict[str, CausalVariable] = {}
        self._edges: Dict[Tuple[str, str], CausalEdge] = {}
        self._parents: Dict[str, Set[str]] = defaultdict(set)
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_variable(self, var: CausalVariable):
        """Add variable to graph"""
        with self._lock:
            self._variables[var.var_id] = var

    def add_edge(self, edge: CausalEdge):
        """Add causal edge"""
        with self._lock:
            key = (edge.source_id, edge.target_id)
            self._edges[key] = edge
            self._parents[edge.target_id].add(edge.source_id)
            self._children[edge.source_id].add(edge.target_id)

    def get_parents(self, var_id: str) -> Set[str]:
        """Get parent variables (direct causes)"""
        return self._parents.get(var_id, set())

    def get_children(self, var_id: str) -> Set[str]:
        """Get child variables (direct effects)"""
        return self._children.get(var_id, set())

    def get_ancestors(self, var_id: str) -> Set[str]:
        """Get all ancestor variables (indirect causes)"""
        ancestors = set()
        queue = list(self.get_parents(var_id))

        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.get_parents(current))

        return ancestors

    def get_descendants(self, var_id: str) -> Set[str]:
        """Get all descendant variables (indirect effects)"""
        descendants = set()
        queue = list(self.get_children(var_id))

        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(self.get_children(current))

        return descendants

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z

        D-separation implies conditional independence.
        """
        # Simplified d-separation check
        # Full implementation would use Bayes-Ball algorithm

        # If Z blocks all paths from X to Y, they're d-separated
        if z and (x in z or y in z):
            return True

        # Check if there's a directed path
        descendants_x = self.get_descendants(x)
        if y in descendants_x:
            # Path exists, check if blocked
            for node in z:
                if node in descendants_x:
                    return True

        return False

    def topological_sort(self) -> List[str]:
        """Return variables in topological order"""
        with self._lock:
            in_degree = {v: 0 for v in self._variables}
            for edge in self._edges.values():
                in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1

            queue = [v for v, d in in_degree.items() if d == 0]
            order = []

            while queue:
                node = queue.pop(0)
                order.append(node)

                for child in self._children.get(node, set()):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            return order

    def get_causal_effect(self, cause: str, effect: str) -> float:
        """Get causal effect strength from cause to effect"""
        edge = self._edges.get((cause, effect))
        if edge:
            return edge.strength

        # Check indirect path
        descendants = self.get_descendants(cause)
        if effect in descendants:
            # Calculate product of strengths along path
            # Simplified: would need path enumeration
            return 0.3  # Default indirect effect

        return 0.0


class CausalInferenceEngine:
    """
    Causal Inference Engine

    Performs causal reasoning including:
    - Effect estimation (What is P(Y|do(X))?)
    - Root cause analysis
    - Counterfactual queries
    - Intervention planning

    Usage:
        engine = CausalInferenceEngine()

        # Build causal model
        engine.add_variable(CausalVariable("rain", "Rain", "binary"))
        engine.add_variable(CausalVariable("sprinkler", "Sprinkler", "binary"))
        engine.add_variable(CausalVariable("wet_grass", "Wet Grass", "binary"))

        engine.add_causal_relation("rain", "wet_grass", CausalRelationType.CAUSES)
        engine.add_causal_relation("sprinkler", "wet_grass", CausalRelationType.CAUSES)

        # Query
        effect = engine.compute_causal_effect("rain", "wet_grass")

        # Counterfactual
        cf = engine.counterfactual_query("wet_grass",
            factual={"rain": True, "wet_grass": True},
            counterfactual={"rain": False})
    """

    def __init__(self):
        self._graph = CausalGraph()
        self._observations: List[Dict[str, Any]] = []
        self._interventions: List[Intervention] = []
        self._lock = threading.RLock()

        logger.info("CausalInferenceEngine initialized")

    def add_variable(self, var: CausalVariable):
        """Add a variable to the causal model"""
        self._graph.add_variable(var)

    def add_causal_relation(self, cause: str, effect: str,
                           relation: CausalRelationType = CausalRelationType.CAUSES,
                           strength: float = 0.5):
        """Add a causal relationship"""
        edge = CausalEdge(
            source_id=cause,
            target_id=effect,
            relation=relation,
            strength=strength,
        )
        self._graph.add_edge(edge)

    def add_observation(self, observation: Dict[str, Any]):
        """Add an observational data point"""
        with self._lock:
            self._observations.append(observation)

            # Update variable observed values
            for var_id, value in observation.items():
                if var_id in self._graph._variables:
                    self._graph._variables[var_id].observed_value = value

    def compute_causal_effect(self, cause: str, effect: str,
                              evidence: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute P(effect | do(cause))

        Uses do-calculus to estimate causal effect.
        """
        with self._lock:
            # Get direct causal effect
            direct_effect = self._graph.get_causal_effect(cause, effect)

            if direct_effect > 0:
                return direct_effect

            # Check for indirect effects via descendants
            descendants = self._graph.get_descendants(cause)
            if effect not in descendants:
                return 0.0

            # Compute via back-door criterion
            # Simplified implementation
            confounders = self._graph.get_parents(cause) & self._graph.get_parents(effect)

            if not confounders:
                # No confounding, direct estimation
                return self._estimate_from_observations(cause, effect)
            else:
                # Adjust for confounders
                return self._adjust_for_confounders(cause, effect, confounders)

    def _estimate_from_observations(self, cause: str, effect: str) -> float:
        """Estimate causal effect from observational data"""
        if not self._observations:
            return 0.5

        # Count co-occurrences
        cause_true = 0
        effect_given_cause = 0

        for obs in self._observations:
            if obs.get(cause):
                cause_true += 1
                if obs.get(effect):
                    effect_given_cause += 1

        if cause_true == 0:
            return 0.5

        return effect_given_cause / cause_true

    def _adjust_for_confounders(self, cause: str, effect: str,
                                confounders: Set[str]) -> float:
        """Adjust causal estimate for confounding variables"""
        # Back-door adjustment formula
        # P(Y|do(X)) = Σ_z P(Y|X,Z) P(Z)

        if not self._observations or not confounders:
            return self._estimate_from_observations(cause, effect)

        # Stratify by confounders (simplified)
        strata = defaultdict(list)

        for obs in self._observations:
            confounder_values = tuple(obs.get(c) for c in sorted(confounders))
            strata[confounder_values].append(obs)

        # Compute weighted average
        total_weight = 0
        weighted_effect = 0

        for stratum_key, stratum_obs in strata.items():
            weight = len(stratum_obs) / len(self._observations)

            # Effect within stratum
            cause_true = sum(1 for o in stratum_obs if o.get(cause))
            effect_given_cause = sum(1 for o in stratum_obs if o.get(cause) and o.get(effect))

            if cause_true > 0:
                stratum_effect = effect_given_cause / cause_true
                weighted_effect += weight * stratum_effect

            total_weight += weight

        return weighted_effect / total_weight if total_weight > 0 else 0.5

    def counterfactual_query(self, target: str,
                            factual: Dict[str, Any],
                            counterfactual: Dict[str, Any]) -> Counterfactual:
        """
        Answer a counterfactual query

        "Given that factual happened, what would target be if counterfactual?"

        Args:
            target: Variable we're querying
            factual: What actually happened
            counterfactual: Hypothetical changes

        Returns:
            Counterfactual result
        """
        query_id = hashlib.sha256(
            f"{target}:{factual}:{counterfactual}".encode()
        ).hexdigest()[:16]

        with self._lock:
            # Step 1: Abduction - infer exogenous variables from factual
            exogenous = self._abduction(factual)

            # Step 2: Action - apply intervention
            modified_model = self._apply_intervention(counterfactual)

            # Step 3: Prediction - compute target under counterfactual
            cf_value, probability = self._predict_counterfactual(
                target, modified_model, exogenous
            )

            # Find causal path
            changed_vars = set(counterfactual.keys())
            causal_path = self._find_causal_path(changed_vars, target)

            # Generate explanation
            explanation = self._generate_cf_explanation(
                target, factual, counterfactual, cf_value
            )

            return Counterfactual(
                query_id=query_id,
                factual=factual,
                counterfactual={**factual, **counterfactual, target: cf_value},
                probability=probability,
                explanation=explanation,
                causal_path=causal_path,
            )

    def _abduction(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Infer exogenous noise variables from evidence"""
        # Simplified: assume exogenous variables are residuals
        exogenous = {}

        for var_id in self._graph._variables:
            if var_id in evidence:
                # Compute residual
                parents = self._graph.get_parents(var_id)
                expected = self._compute_expected(var_id, {p: evidence.get(p) for p in parents})
                actual = 1.0 if evidence[var_id] else 0.0
                exogenous[f"U_{var_id}"] = actual - expected

        return exogenous

    def _compute_expected(self, var_id: str, parent_values: Dict[str, Any]) -> float:
        """Compute expected value given parent values"""
        parents = self._graph.get_parents(var_id)

        if not parents:
            return 0.5

        # Weighted sum of parent influences
        total = 0.0
        for parent in parents:
            edge = self._graph._edges.get((parent, var_id))
            if edge and parent in parent_values:
                parent_val = 1.0 if parent_values[parent] else 0.0
                total += edge.strength * parent_val

        return min(1.0, max(0.0, total))

    def _apply_intervention(self, intervention: Dict[str, Any]) -> Dict:
        """Apply intervention to model (graph surgery)"""
        # In full implementation, would modify graph structure
        return intervention

    def _predict_counterfactual(self, target: str, intervention: Dict,
                                exogenous: Dict) -> Tuple[Any, float]:
        """Predict target value under counterfactual"""
        # Propagate intervention effects
        ancestors = self._graph.get_ancestors(target)

        # Check if intervention affects target
        intervention_vars = set(intervention.keys())
        affecting_vars = intervention_vars & (ancestors | {target})

        if not affecting_vars:
            # Intervention doesn't affect target
            return self._graph._variables[target].observed_value, 0.8

        # Compute counterfactual value
        # Simplified: use structural equation
        cf_value = self._compute_expected(target, intervention)
        cf_binary = cf_value > 0.5

        # Confidence based on causal path length
        confidence = 0.9 ** len(affecting_vars)

        return cf_binary, confidence

    def _find_causal_path(self, sources: Set[str], target: str) -> List[str]:
        """Find causal path from sources to target"""
        for source in sources:
            path = self._bfs_path(source, target)
            if path:
                return path
        return []

    def _bfs_path(self, start: str, end: str) -> List[str]:
        """BFS to find path in causal graph"""
        if start == end:
            return [start]

        visited = {start}
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            for child in self._graph.get_children(current):
                if child == end:
                    return path + [child]
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        return []

    def _generate_cf_explanation(self, target: str, factual: Dict,
                                 counterfactual: Dict, cf_value: Any) -> str:
        """Generate human-readable explanation"""
        changes = []
        for var, val in counterfactual.items():
            if var in factual and factual[var] != val:
                changes.append(f"{var} changed from {factual[var]} to {val}")

        change_str = ", ".join(changes) if changes else "no changes"

        actual = factual.get(target, "unknown")

        return f"If {change_str}, then {target} would be {cf_value} (was {actual})"

    def root_cause_analysis(self, effect: str,
                           effect_value: Any) -> List[Tuple[str, float]]:
        """
        Identify root causes of an observed effect

        Returns:
            List of (cause, contribution_score) tuples
        """
        causes = []

        with self._lock:
            # Get all ancestors
            ancestors = self._graph.get_ancestors(effect)

            for ancestor in ancestors:
                # Compute causal contribution
                contribution = self.compute_causal_effect(ancestor, effect)

                if contribution > 0.1:
                    causes.append((ancestor, contribution))

            # Also include direct parents
            for parent in self._graph.get_parents(effect):
                if parent not in [c[0] for c in causes]:
                    contribution = self.compute_causal_effect(parent, effect)
                    causes.append((parent, contribution))

        # Sort by contribution
        causes.sort(key=lambda x: x[1], reverse=True)

        return causes

    def plan_intervention(self, target: str, desired_value: Any,
                         allowed_interventions: Optional[Set[str]] = None) -> List[Intervention]:
        """
        Plan interventions to achieve desired outcome

        Returns:
            List of recommended interventions
        """
        interventions = []

        with self._lock:
            # Get all ancestors that could affect target
            ancestors = self._graph.get_ancestors(target) | self._graph.get_parents(target)

            if allowed_interventions:
                ancestors &= allowed_interventions

            for ancestor in ancestors:
                effect = self.compute_causal_effect(ancestor, target)

                if effect > 0.3:  # Meaningful effect
                    intervention = Intervention(
                        intervention_id=f"int_{ancestor}_{target}",
                        variable_id=ancestor,
                        set_value=desired_value,  # Simplified
                        confidence=effect,
                    )
                    intervention.predicted_effects[target] = desired_value
                    interventions.append(intervention)

        # Sort by confidence
        interventions.sort(key=lambda i: i.confidence, reverse=True)

        return interventions

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "variables": len(self._graph._variables),
            "edges": len(self._graph._edges),
            "observations": len(self._observations),
            "interventions": len(self._interventions),
        }


if __name__ == "__main__":
    print("Causal Inference Self-Test")
    print("=" * 50)

    engine = CausalInferenceEngine()

    print("\n[1] Build Causal Model")
    # Classic sprinkler example
    engine.add_variable(CausalVariable("season", "Season", "categorical"))
    engine.add_variable(CausalVariable("rain", "Rain", "binary"))
    engine.add_variable(CausalVariable("sprinkler", "Sprinkler", "binary"))
    engine.add_variable(CausalVariable("wet_grass", "Wet Grass", "binary"))
    engine.add_variable(CausalVariable("slippery", "Slippery Sidewalk", "binary"))

    engine.add_causal_relation("season", "rain", strength=0.7)
    engine.add_causal_relation("season", "sprinkler", strength=0.6)
    engine.add_causal_relation("rain", "wet_grass", strength=0.9)
    engine.add_causal_relation("sprinkler", "wet_grass", strength=0.8)
    engine.add_causal_relation("wet_grass", "slippery", strength=0.7)

    print("    Built model with 5 variables and 5 causal relations")

    print("\n[2] Add Observations")
    observations = [
        {"season": "summer", "rain": False, "sprinkler": True, "wet_grass": True, "slippery": True},
        {"season": "summer", "rain": False, "sprinkler": False, "wet_grass": False, "slippery": False},
        {"season": "winter", "rain": True, "sprinkler": False, "wet_grass": True, "slippery": True},
        {"season": "winter", "rain": True, "sprinkler": False, "wet_grass": True, "slippery": False},
    ]
    for obs in observations:
        engine.add_observation(obs)
    print(f"    Added {len(observations)} observations")

    print("\n[3] Causal Effect Query")
    effect = engine.compute_causal_effect("rain", "wet_grass")
    print(f"    P(wet_grass | do(rain)) = {effect:.2f}")

    effect2 = engine.compute_causal_effect("sprinkler", "slippery")
    print(f"    P(slippery | do(sprinkler)) = {effect2:.2f}")

    print("\n[4] Counterfactual Query")
    cf = engine.counterfactual_query(
        target="slippery",
        factual={"rain": True, "wet_grass": True, "slippery": True},
        counterfactual={"rain": False}
    )
    print(f"    Query: {cf.explanation}")
    print(f"    Probability: {cf.probability:.2f}")
    print(f"    Causal path: {' -> '.join(cf.causal_path) if cf.causal_path else 'N/A'}")

    print("\n[5] Root Cause Analysis")
    causes = engine.root_cause_analysis("slippery", True)
    print("    Causes of slippery sidewalk:")
    for cause, contribution in causes[:3]:
        print(f"      - {cause}: {contribution:.2f}")

    print("\n[6] Intervention Planning")
    interventions = engine.plan_intervention("slippery", False)
    print("    To prevent slippery sidewalk:")
    for intv in interventions[:2]:
        print(f"      - Set {intv.variable_id} (confidence: {intv.confidence:.2f})")

    print("\n[7] Statistics")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Causal Inference test complete")

