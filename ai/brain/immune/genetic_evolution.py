#!/usr/bin/env python3
"""
Genetic Algorithm Evolution for DSMIL Brain

Evolving analysis strategies:
- Competing analysis strategies
- Survival of most effective
- Mutation for novel techniques
- Crossover of successful methods
"""

import random
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStrategy:
    """An evolved analysis strategy"""
    strategy_id: str
    name: str

    # Genes (parameters)
    genes: Dict[str, float] = field(default_factory=dict)

    # Fitness
    fitness: float = 0.0
    evaluations: int = 0

    # Lineage
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def mutate(self, rate: float = 0.1) -> "AnalysisStrategy":
        """Create mutated copy"""
        new_genes = {}
        for key, value in self.genes.items():
            if random.random() < rate:
                # Mutate
                new_genes[key] = max(0, min(1, value + random.gauss(0, 0.2)))
            else:
                new_genes[key] = value

        return AnalysisStrategy(
            strategy_id=hashlib.sha256(f"{self.strategy_id}:mutate:{random.random()}".encode()).hexdigest()[:16],
            name=f"{self.name}_m",
            genes=new_genes,
            generation=self.generation + 1,
            parent_ids=[self.strategy_id],
        )

    def crossover(self, other: "AnalysisStrategy") -> "AnalysisStrategy":
        """Create offspring from two parents"""
        new_genes = {}
        all_keys = set(self.genes.keys()) | set(other.genes.keys())

        for key in all_keys:
            if random.random() < 0.5:
                new_genes[key] = self.genes.get(key, 0.5)
            else:
                new_genes[key] = other.genes.get(key, 0.5)

        return AnalysisStrategy(
            strategy_id=hashlib.sha256(f"{self.strategy_id}:{other.strategy_id}:{random.random()}".encode()).hexdigest()[:16],
            name=f"{self.name}x{other.name}",
            genes=new_genes,
            generation=max(self.generation, other.generation) + 1,
            parent_ids=[self.strategy_id, other.strategy_id],
        )

    def evaluate(self, success: bool):
        """Update fitness based on evaluation"""
        self.evaluations += 1
        if success:
            self.fitness = (self.fitness * (self.evaluations - 1) + 1.0) / self.evaluations
        else:
            self.fitness = (self.fitness * (self.evaluations - 1) + 0.0) / self.evaluations


@dataclass
class StrategyPopulation:
    """Population of strategies"""
    population_id: str
    strategies: List[AnalysisStrategy] = field(default_factory=list)
    generation: int = 0


class GeneticEvolution:
    """
    Genetic Evolution System

    Evolves analysis strategies through genetic algorithms.

    Usage:
        evolution = GeneticEvolution(population_size=20)

        # Initialize population
        evolution.initialize_population()

        # Evaluate strategies
        for strategy in evolution.get_strategies():
            result = run_analysis(strategy)
            evolution.evaluate(strategy.strategy_id, result)

        # Evolve next generation
        evolution.evolve()
    """

    def __init__(self, population_size: int = 20,
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio

        self._strategies: Dict[str, AnalysisStrategy] = {}
        self._generation = 0
        self._lock = threading.RLock()

        # Default gene template
        self._gene_template = {
            "sensitivity": 0.5,
            "specificity": 0.5,
            "depth": 0.5,
            "breadth": 0.5,
            "speed": 0.5,
        }

        logger.info("GeneticEvolution initialized")

    def initialize_population(self):
        """Initialize random population"""
        with self._lock:
            for i in range(self.population_size):
                genes = {k: random.random() for k in self._gene_template}

                strategy = AnalysisStrategy(
                    strategy_id=hashlib.sha256(f"init:{i}:{random.random()}".encode()).hexdigest()[:16],
                    name=f"strategy_{i}",
                    genes=genes,
                    generation=0,
                )

                self._strategies[strategy.strategy_id] = strategy

    def evaluate(self, strategy_id: str, success: bool):
        """Record evaluation result"""
        with self._lock:
            if strategy_id in self._strategies:
                self._strategies[strategy_id].evaluate(success)

    def evolve(self) -> int:
        """
        Evolve to next generation

        Returns number of new strategies created
        """
        with self._lock:
            strategies = list(self._strategies.values())

            # Sort by fitness
            strategies.sort(key=lambda s: s.fitness, reverse=True)

            # Select elite
            elite_count = int(self.population_size * self.elite_ratio)
            elite = strategies[:elite_count]

            new_strategies = []

            # Keep elite
            for e in elite:
                new_strategies.append(e)

            # Generate offspring
            while len(new_strategies) < self.population_size:
                # Select parents (tournament selection)
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)

                # Crossover
                child = parent1.crossover(parent2)

                # Mutate
                if random.random() < self.mutation_rate:
                    child = child.mutate(self.mutation_rate)

                new_strategies.append(child)

            # Replace population
            self._strategies = {s.strategy_id: s for s in new_strategies}
            self._generation += 1

            return len(new_strategies) - elite_count

    def get_strategies(self) -> List[AnalysisStrategy]:
        """Get all strategies"""
        with self._lock:
            return list(self._strategies.values())

    def get_best_strategy(self) -> Optional[AnalysisStrategy]:
        """Get strategy with highest fitness"""
        with self._lock:
            if not self._strategies:
                return None
            return max(self._strategies.values(), key=lambda s: s.fitness)

    def get_stats(self) -> Dict:
        """Get evolution statistics"""
        with self._lock:
            if not self._strategies:
                return {"generation": 0, "population": 0}

            fitnesses = [s.fitness for s in self._strategies.values()]
            return {
                "generation": self._generation,
                "population": len(self._strategies),
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "max_fitness": max(fitnesses),
                "min_fitness": min(fitnesses),
            }


if __name__ == "__main__":
    print("Genetic Evolution Self-Test")
    print("=" * 50)

    evolution = GeneticEvolution(population_size=10)

    print("\n[1] Initialize Population")
    evolution.initialize_population()
    print(f"    Created {len(evolution.get_strategies())} strategies")

    print("\n[2] Simulate Evaluations")
    for strategy in evolution.get_strategies():
        # Simulate: higher sensitivity = better detection
        success = random.random() < strategy.genes.get("sensitivity", 0.5)
        evolution.evaluate(strategy.strategy_id, success)

    print("\n[3] Evolve")
    for gen in range(5):
        # Evaluate
        for strategy in evolution.get_strategies():
            success = random.random() < strategy.genes.get("sensitivity", 0.5)
            evolution.evaluate(strategy.strategy_id, success)

        # Evolve
        new_count = evolution.evolve()
        stats = evolution.get_stats()
        print(f"    Gen {stats['generation']}: avg={stats['avg_fitness']:.3f}, max={stats['max_fitness']:.3f}")

    print("\n[4] Best Strategy")
    best = evolution.get_best_strategy()
    if best:
        print(f"    ID: {best.strategy_id}")
        print(f"    Fitness: {best.fitness:.3f}")
        print(f"    Genes: {best.genes}")

    print("\n" + "=" * 50)
    print("Genetic Evolution test complete")

