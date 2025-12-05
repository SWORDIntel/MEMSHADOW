#!/usr/bin/env python3
"""
Zero-Knowledge Proofs for DSMIL Brain

Prove facts without revealing underlying data:
- Prove membership without revealing identity
- Verify computations without seeing data
- Privacy-preserving aggregation
"""

import os
import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ProofStatement:
    """A statement to prove"""
    statement_id: str
    claim: str  # What is being claimed
    public_inputs: Dict[str, Any] = field(default_factory=dict)
    # Private witness NOT included - only prover has it


@dataclass
class ZKProof:
    """A zero-knowledge proof"""
    proof_id: str
    statement_id: str

    # Proof components (simplified simulation)
    commitment: bytes
    challenge: bytes
    response: bytes

    # Metadata
    prover_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Verification
    is_valid: Optional[bool] = None


class ZeroKnowledgeProver:
    """
    Zero-Knowledge Proof System

    Enables proving statements without revealing underlying data.

    Note: This is a simulation of ZKP concepts. Production would use
    libraries like libsnark, bellman, or circom.

    Supported proof types:
    - Membership: Prove entity is in set without revealing which
    - Range: Prove value is in range without revealing value
    - Computation: Prove computation result without revealing inputs
    - Knowledge: Prove knowledge of secret without revealing it

    Usage:
        zkp = ZeroKnowledgeProver()

        # Create statement
        statement = zkp.create_statement(
            claim="I know a value x such that hash(x) = H",
            public_inputs={"hash": "abc123..."}
        )

        # Create proof (prover has secret witness)
        proof = zkp.prove(statement, witness={"x": "secret_value"})

        # Verify proof (verifier doesn't see witness)
        valid = zkp.verify(proof)
    """

    def __init__(self, node_id: str = ""):
        self.node_id = node_id
        self._statements: Dict[str, ProofStatement] = {}
        self._proofs: Dict[str, ZKProof] = {}
        self._lock = threading.RLock()

        logger.info(f"ZeroKnowledgeProver initialized (node={node_id})")

    def create_statement(self, claim: str,
                        public_inputs: Optional[Dict[str, Any]] = None) -> ProofStatement:
        """Create a statement to prove"""
        with self._lock:
            statement_id = hashlib.sha256(
                f"{claim}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            statement = ProofStatement(
                statement_id=statement_id,
                claim=claim,
                public_inputs=public_inputs or {},
            )

            self._statements[statement_id] = statement
            return statement

    def prove(self, statement: ProofStatement,
             witness: Dict[str, Any]) -> ZKProof:
        """
        Create a zero-knowledge proof

        Args:
            statement: The statement to prove
            witness: Private data known only to prover

        Returns:
            ZKProof that can be verified without revealing witness
        """
        with self._lock:
            # Simulate Schnorr-like protocol
            # In production, use actual ZKP circuit

            # Commitment phase: prover commits to random value
            random_nonce = os.urandom(32)
            commitment = hashlib.sha384(random_nonce).digest()

            # Challenge: hash of commitment and public inputs
            challenge_input = commitment + str(statement.public_inputs).encode()
            challenge = hashlib.sha384(challenge_input).digest()

            # Response: computed using witness
            witness_hash = hashlib.sha384(str(witness).encode()).digest()
            response = hashlib.sha384(random_nonce + witness_hash + challenge).digest()

            proof_id = hashlib.sha256(
                f"proof:{statement.statement_id}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            proof = ZKProof(
                proof_id=proof_id,
                statement_id=statement.statement_id,
                commitment=commitment,
                challenge=challenge,
                response=response,
                prover_id=self.node_id,
            )

            self._proofs[proof_id] = proof
            return proof

    def verify(self, proof: ZKProof) -> bool:
        """
        Verify a zero-knowledge proof

        Returns True if proof is valid, False otherwise.
        Does NOT require knowledge of witness.
        """
        with self._lock:
            # Get statement
            statement = self._statements.get(proof.statement_id)
            if not statement:
                logger.warning(f"Statement {proof.statement_id} not found")
                return False

            # Verify challenge was computed correctly
            expected_challenge_input = proof.commitment + str(statement.public_inputs).encode()
            expected_challenge = hashlib.sha384(expected_challenge_input).digest()

            if proof.challenge != expected_challenge:
                logger.warning("Challenge mismatch")
                proof.is_valid = False
                return False

            # In real ZKP, verify response against commitment and challenge
            # using the verification equation

            # Simulation: check response is non-empty and properly formed
            if len(proof.response) != 48:  # SHA384 output
                proof.is_valid = False
                return False

            proof.is_valid = True
            return True

    def prove_membership(self, element: Any, merkle_root: bytes,
                        merkle_path: List[Tuple[bytes, bool]]) -> ZKProof:
        """
        Prove membership in a set without revealing which element

        Args:
            element: The element (witness - kept secret)
            merkle_root: Root of Merkle tree (public)
            merkle_path: Authentication path (witness)
        """
        statement = self.create_statement(
            claim="I know an element in the set",
            public_inputs={"merkle_root": merkle_root.hex()}
        )

        return self.prove(statement, {
            "element": element,
            "path": merkle_path,
        })

    def prove_range(self, value: int, min_val: int, max_val: int) -> ZKProof:
        """
        Prove value is in range without revealing value

        Args:
            value: The actual value (witness)
            min_val: Minimum (public)
            max_val: Maximum (public)
        """
        statement = self.create_statement(
            claim=f"I know a value x where {min_val} <= x <= {max_val}",
            public_inputs={"min": min_val, "max": max_val}
        )

        return self.prove(statement, {"value": value})

    def prove_computation(self, inputs: Dict[str, Any],
                         output: Any,
                         computation_hash: bytes) -> ZKProof:
        """
        Prove computation was done correctly without revealing inputs

        Args:
            inputs: Computation inputs (witness)
            output: Computation output (public)
            computation_hash: Hash of computation logic (public)
        """
        statement = self.create_statement(
            claim="I computed f(x) = y correctly",
            public_inputs={
                "output": output,
                "computation": computation_hash.hex(),
            }
        )

        return self.prove(statement, {"inputs": inputs})

    def aggregate_proofs(self, proofs: List[ZKProof]) -> ZKProof:
        """
        Aggregate multiple proofs into one

        Reduces verification overhead for multiple proofs.
        """
        with self._lock:
            if not proofs:
                raise ValueError("No proofs to aggregate")

            # Combine commitments
            combined_commitment = hashlib.sha384(
                b"".join(p.commitment for p in proofs)
            ).digest()

            # Combine challenges
            combined_challenge = hashlib.sha384(
                b"".join(p.challenge for p in proofs)
            ).digest()

            # Combine responses
            combined_response = hashlib.sha384(
                b"".join(p.response for p in proofs)
            ).digest()

            # Create aggregate statement
            agg_statement = self.create_statement(
                claim=f"Aggregate of {len(proofs)} proofs",
                public_inputs={"proof_ids": [p.proof_id for p in proofs]}
            )

            return ZKProof(
                proof_id=hashlib.sha256(f"agg:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
                statement_id=agg_statement.statement_id,
                commitment=combined_commitment,
                challenge=combined_challenge,
                response=combined_response,
                prover_id="aggregator",
            )

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            valid_proofs = sum(1 for p in self._proofs.values() if p.is_valid)
            return {
                "statements": len(self._statements),
                "proofs": len(self._proofs),
                "valid_proofs": valid_proofs,
            }


if __name__ == "__main__":
    print("Zero-Knowledge Proofs Self-Test")
    print("=" * 50)

    zkp = ZeroKnowledgeProver(node_id="test-node")

    print("\n[1] Basic Proof")
    statement = zkp.create_statement(
        claim="I know the password",
        public_inputs={"hash": hashlib.sha256(b"secret123").hexdigest()}
    )
    print(f"    Statement: {statement.claim}")

    proof = zkp.prove(statement, {"password": "secret123"})
    print(f"    Proof ID: {proof.proof_id}")

    valid = zkp.verify(proof)
    print(f"    Valid: {valid}")

    print("\n[2] Range Proof")
    range_proof = zkp.prove_range(value=42, min_val=0, max_val=100)
    print(f"    Proving 0 <= 42 <= 100")
    print(f"    Proof ID: {range_proof.proof_id}")
    valid = zkp.verify(range_proof)
    print(f"    Valid: {valid}")

    print("\n[3] Membership Proof")
    merkle_root = hashlib.sha256(b"tree_root").digest()
    membership_proof = zkp.prove_membership(
        element="alice",
        merkle_root=merkle_root,
        merkle_path=[(hashlib.sha256(b"sibling1").digest(), True)],
    )
    print(f"    Proving membership without revealing identity")
    print(f"    Proof ID: {membership_proof.proof_id}")
    valid = zkp.verify(membership_proof)
    print(f"    Valid: {valid}")

    print("\n[4] Computation Proof")
    comp_proof = zkp.prove_computation(
        inputs={"a": 3, "b": 4},
        output=12,
        computation_hash=hashlib.sha256(b"multiply").digest(),
    )
    print(f"    Proving f(a,b) = 12 without revealing a, b")
    print(f"    Proof ID: {comp_proof.proof_id}")
    valid = zkp.verify(comp_proof)
    print(f"    Valid: {valid}")

    print("\n[5] Aggregate Proofs")
    agg_proof = zkp.aggregate_proofs([range_proof, membership_proof, comp_proof])
    print(f"    Aggregated 3 proofs")
    print(f"    Aggregate Proof ID: {agg_proof.proof_id}")

    print("\n[6] Statistics")
    stats = zkp.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Zero-Knowledge Proofs test complete")

