# Hybrid Local-Cloud Architecture for MEMSHADOW

This document outlines the conceptual hybrid local-cloud architecture for Project MEMSHADOW, particularly leveraging local NPU/CPU capabilities alongside VPS-based backend services. This architecture is inspired by ideas presented in `CLAUDEI2-Concepts.md`.

## 1. Rationale

A hybrid architecture aims to combine the strengths of local processing (low latency, privacy, offline access for some features) with the power and scalability of cloud/VPS resources (large-scale storage, heavy computation, centralized management). This is especially relevant for users with modern hardware equipped with NPUs (Neural Processing Units) or powerful CPUs.

## 2. Core Principles

-   **Tiered Memory:** Data and processing are distributed across local and remote tiers based on immediacy, computational cost, and data volume.
-   **Smart Routing:** Operations are routed to the most appropriate execution environment (local NPU, local CPU, or remote VPS).
-   **Differential Sync:** Minimize data transfer by syncing only deltas or essential information between local and remote stores.
-   **Graceful Degradation:** Provide core functionality even when offline, with full capabilities restored upon reconnection.
-   **Power Awareness:** Adapt processing strategies based on the device's power state (e.g., battery vs. plugged in).

## 3. Architectural Components & Tiers

```mermaid
graph LR
    subgraph Local Device (User's Machine)
        direction TB
        Local_NPU[Local NPU]
        Local_CPU[Local CPU]
        Local_RAM[L1 Cache (RAM - Working Memory)]
        Local_SSD[L2 Cache (NVMe SSD - Hot Memory Index)]
        Local_Client[MEMSHADOW Client (CLI/GUI/Extension)]
        Local_SyncAgent[Local Sync Agent]

        Local_Client --> Local_NPU
        Local_Client --> Local_CPU
        Local_NPU --> Local_RAM
        Local_CPU --> Local_RAM
        Local_CPU --> Local_SSD
        Local_RAM --> Local_Client
        Local_SSD --> Local_Client
        Local_SyncAgent -.-> Local_SSD
    end

    subgraph Remote VPS
        direction TB
        VPS_API[MEMSHADOW API (FastAPI)]
        VPS_CPU_GPU[VPS CPU/GPU (Heavy Processing)]
        VPS_DB[L3 Storage (PostgreSQL, ChromaDB - Cold Storage)]
        VPS_Enrichment[Enrichment Pipeline]
        VPS_Analytics[Cross-Project Analytics]
        VPS_KG[Global Knowledge Graph]
    end

    Local_SyncAgent <-->|Secure Sync Protocol (HTTPS/gRPC)| VPS_API

    VPS_API --> VPS_CPU_GPU
    VPS_API --> VPS_DB
    VPS_CPU_GPU --> VPS_Enrichment
    VPS_CPU_GPU --> VPS_Analytics
    VPS_CPU_GPU --> VPS_KG
    VPS_Enrichment --> VPS_DB
    VPS_Analytics --> VPS_DB
    VPS_KG --> VPS_DB
```

### 3.1 Local Device Components

*   **Local NPU (Neural Processing Unit):**
    *   **Responsibilities:** Real-time, power-efficient AI tasks.
        *   Instant embedding generation for new conversations/notes.
        *   Real-time sentiment tracking or intent classification during input.
        *   Accelerated semantic search on L2 cache.
        *   Memory relevance scoring for immediate context.
        *   Edge-based memory compression before sync.
        *   Running small, specialized ONNX models for quick analysis (e.g., code similarity, decision extraction).
    *   **Characteristics:** Optimized for low-latency inference with smaller models.

*   **Local CPU:**
    *   **Responsibilities:** More complex local processing.
        *   Managing L1 and L2 caches (indexing, updates).
        *   Syntax analysis, structured queries on local data.
        *   Running medium-sized local models (e.g., local Phi-3/Gemma for memory Q&A if VPS is unavailable).
        *   Memory deduplication on local data.
        *   Orchestrating tasks between NPU, RAM, and SSD.
    *   **Characteristics:** General-purpose processing, higher power consumption than NPU for sustained tasks.

*   **L1 Cache (RAM - Working Memory):**
    *   **Content:** Current session data, most frequently accessed memories, active project context.
    *   **Characteristics:** Fastest access, volatile.

*   **L2 Cache (NVMe SSD - Hot Memory Index):**
    *   **Content:** Recently accessed memories, frequently used project contexts, NPU-accelerated vector index for a subset of memories (e.g., last 30 days or most relevant projects).
    *   **Characteristics:** Fast access, persistent, larger capacity than RAM. Search can be NPU or CPU-accelerated.

*   **Local MEMSHADOW Client:**
    *   The user interface (CLI, GUI, Browser Extension).
    *   Captures input, displays results, interacts with local caches and processing units.

*   **Local Sync Agent:**
    *   Manages synchronization of memories and context between the local device and the VPS backend.
    *   Implements differential sync, conflict resolution (if applicable), and handles offline queuing.

### 3.2 Remote VPS Components

*   **MEMSHADOW API (FastAPI):**
    *   Central hub for all remote operations, authentication, and data consistency.
    *   Manages requests from local sync agents.

*   **VPS CPU/GPU (Heavy Processing):**
    *   **Responsibilities:** Computationally intensive tasks.
        *   Deep enrichment of memories (complex NLP, large model analysis).
        *   Cross-project analysis and pattern detection.
        *   Updating the global knowledge graph.
        *   Training or fine-tuning larger (potentially custom) embedding or analysis models.
        *   Batch processing of large memory archives.

*   **L3 Storage (PostgreSQL, ChromaDB - Cold Storage):**
    *   **Content:** The complete, canonical MEMSHADOW datastore. All memories, full vector index, metadata, user accounts.
    *   **Characteristics:** Largest capacity, source of truth, may have higher latency than local caches.

*   **Enrichment Pipeline, Analytics, Knowledge Graph:**
    *   As detailed in the main MEMSHADOW architecture, these run on the VPS utilizing its resources.

## 4. Key Hybrid Operations and Flows

### 4.1 Memory Ingestion (Local-First)
1.  **Local Client:** User inputs new information.
2.  **Local NPU/CPU:**
    *   (NPU) Real-time embedding generation.
    *   (CPU) Immediate indexing into L1/L2 cache.
    *   (NPU) Initial relevance scoring, sentiment analysis.
3.  **Local Sync Agent:**
    *   Queues the new memory for synchronization.
    *   (NPU) Compresses memory data if applicable.
    *   Sends compressed delta to VPS API.
4.  **VPS API:**
    *   Receives data, validates, stores in L3.
    *   Queues for heavy enrichment tasks on VPS CPU/GPU.

### 4.2 Memory Retrieval (Local-First)
1.  **Local Client:** User issues a query.
2.  **Local NPU/CPU:**
    *   (NPU) Query embedding generation.
    *   (NPU/CPU) Semantic search on L1 (RAM) and L2 (SSD) caches. Approximate Nearest Neighbor (ANN) on NPU for speed, refined by CPU.
3.  **Local Client:** Displays initial local results immediately.
4.  **Local Sync Agent (in parallel or if local results are insufficient):**
    *   Sends the query to VPS API.
5.  **VPS API:**
    *   Performs comprehensive search on L3 storage.
    *   Returns more complete or deeper semantic matches.
6.  **Local Client:** Integrates VPS results with local results, potentially refining the display.

### 4.3 Predictive Memory Loading
1.  **Local NPU:** Analyzes ongoing user activity (e.g., code being written, documents being read) in real-time.
2.  **Local NPU:** Predicts likely next memory needs based on learned patterns or current context.
3.  **Local CPU:** Preemptively loads these predicted memories from L2 cache into L1 cache, or even fetches them from VPS via Sync Agent if high confidence and not available locally.
4.  **Result:** Reduced latency when the user actually needs that context.

### 4.4 Distributed Memory Compute Pipeline Example
A task like "understanding a new code file" could be pipelined:
```yaml
memory_pipeline_task: "analyze_new_code_file.py"
  local_stage:
    - npu_ops: # <10ms per op
        - "semantic_embedding_of_code_chunks"
        - "quick_pattern_recognition (e.g., is it a test, util, model?)"
    - cpu_ops: # <50ms per op
        - "syntax_analysis_ast_generation"
        - "immediate_local_indexing"
        - "similarity_to_L2_cache_code"
  sync_to_vps: # Payload: code, local NPU/CPU analysis results
  remote_vps_stage: # Asynchronous
    - vps_ops:
        - "deep_enrichment (e.g., call larger CodeLLM for explanation, vulnerabilities)"
        - "cross_project_analysis (e.g., find similar functions in other projects)"
        - "update_global_knowledge_graph (e.g., new function dependencies)"
        - "model_distillation (if applicable, update local NPU models based on new findings)"
  sync_from_vps: # Optional: Send back insights or updated models to local device
```

## 5. Power-Aware Operations

The system can adapt its strategy based on power status:
```yaml
power_profiles:
  plugged_in:
    local_npu_usage: "maximum_performance"
    local_cpu_usage: "all_cores_active"
    sync_frequency: "continuous_realtime"
    local_heavy_tasks: "enabled" # e.g., local model fine-tuning on idle

  battery_normal:
    local_npu_usage: "balanced_performance_power"
    local_cpu_usage: "efficiency_cores_preferred"
    sync_frequency: "periodic_batched (e.g., every 5 mins)"
    local_heavy_tasks: "deferred_to_vps"

  battery_low:
    local_npu_usage: "minimal_essential_ops_only"
    local_cpu_usage: "single_core_low_power"
    sync_frequency: "manual_or_on_charge_only"
    local_processing: "minimal_retrieval_only"
```
The Local Client or Sync Agent would monitor power status and adjust behavior.

## 6. Benefits of Hybrid Architecture

-   **Reduced Latency:** Immediate access to relevant local memories.
-   **Offline Capability:** Core memory functions can work offline with L1/L2 caches.
-   **Enhanced Privacy:** Sensitive or recent data can be processed locally before selective sync.
-   **Reduced Cloud Costs:** Offloading computation to local devices can lower VPS operational costs.
-   **Personalization:** Local NPU/CPU can learn user-specific patterns for predictive loading and relevance.
-   **Efficient Use of Powerful Edge Hardware:** Leverages increasingly common NPUs in modern laptops/desktops.

## 7. Challenges and Solutions

-   **Complexity:** Managing distributed state, synchronization, and potential conflicts.
    -   **Solutions:**
        -   **Robust State Management:**
            -   *How:* For the meta-orchestrator managing state across NPU, CPU, and VPS, avoid overly complex distributed consensus (like full Raft/Paxos) for local operations if possible. Instead, consider a primary state manager (e.g., on the CPU or a dedicated local service) using a well-defined state transition diagram and versioning for state updates. For critical cross-component operations that must be atomic, employ patterns like 2-Phase Commit (2PC) if strict consistency is paramount, or Sagas if eventual consistency is acceptable and compensating transactions can be defined. Simpler locking mechanisms with timeouts and clear rollback procedures can also be effective for local resource coordination.
            -   *Conceptual Example (Locking for local atomicity):*
                ```plaintext
                function request_critical_local_op(op_details):
                  lock_id = acquire_meta_lock(components=[NPU, CPU_Orchestrator], timeout=500ms)
                  if lock_id:
                    try:
                      // Perform part 1 on NPU
                      result_npu = NPU.execute_part(op_details.part1)
                      // Perform part 2 on CPU, possibly using NPU result
                      result_cpu = CPU.execute_part(op_details.part2, npu_output=result_npu)
                      // Commit overall state change
                      MetaState.update(op_id, final_state=result_cpu)
                      release_meta_lock(lock_id)
                      return {status: "success", result: result_cpu}
                    catch error:
                      // Rollback logic for NPU and CPU parts
                      NPU.rollback(op_details.part1)
                      CPU.rollback(op_details.part2)
                      MetaState.update(op_id, final_state="error_rolled_back")
                      release_meta_lock(lock_id)
                      return {status: "failure", error: error.message}
                  else:
                    return {status: "timeout", message: "Could not acquire lock"}
                ```
        -   **Configuration Integrity & Security:**
            -   *How:*
                -   **Schema Validation:** Use standard libraries (e.g., `jsonschema` for Python with YAML/JSON, ` यमुना` for Go) to validate configuration files against predefined schemas. Schemas should be versioned alongside the application.
                -   **Cryptographic Signing:** Configuration manifests can be signed offline by a trusted entity. The application then verifies the signature at load time using a public key.
                    -   *Workflow:* `config_file` -> `hash(config_file)` (e.g., SHA256) -> `sign(hash, private_key)` -> `signature_file`.
                    -   *App on load:* Read `config_file`, calculate its hash. Read `signature_file`. `verify(calculated_hash, signature_data, public_key)`.
                -   **Least Privilege Access:** Configuration management systems (e.g., a central deployment pipeline or a Git repository with access controls) must enforce strict ACLs. Local client configuration modifications should be validated against its schema, and sensitive settings might be immutable or require elevated privileges.
        -   **Microkernel Modularity & Security:**
            -   *How:*
                -   **Well-Defined Interfaces:** Use abstract base classes (Python), interfaces (Go/Java), or traits (Rust) to define contracts for pluggable modules. A plugin loader can discover, register, and manage the lifecycle of these modules.
                -   **Input Validation:** Each module interface method must rigorously validate its inputs for type, range, format, and potential malicious content before processing.
                -   **Sandboxing (for high-risk modules):** Consider running less trusted or experimental modules in isolated environments. This could be separate processes with restricted permissions (e.g., using `multiprocessing` in Python with OS-level controls like `seccomp-bpf` on Linux), or leveraging WebAssembly (WASM) runtimes for their sandboxing capabilities.
                -   **Fuzz Testing:** Employ property-based fuzzing (e.g., Hypothesis for Python) for module APIs by defining data invariants that must always hold, regardless of input.
        -   **Controlled Fault Injection:**
            -   *How:*
                -   **Secure Control Plane:** Access to trigger fault injections (e.g., via a dedicated API endpoint or CLI tool) must be strictly authenticated (e.g., mTLS, OAuth2) and authorized (role-based access control).
                -   **Deterministic & Reproducible Faults:** If faults involve randomness, use pseudo-random number generators (PRNGs) seeded with known values to ensure reproducibility.
                -   **Comprehensive Auditing:** Log every fault injection event: who triggered it, what specific fault was injected (e.g., "NPU_latency_injection"), parameters (e.g., "delay_ms: 500"), target component, and timestamp.
                -   **Scope Limitation:** Ensure that fault injection capabilities are designed to affect only specific, targeted components or interactions during testing, unless the goal is to test system-wide cascading failures in a controlled manner.

-   **Data Consistency:** Ensuring consistency between local caches and the central VPS store.
    -   **Solutions:**
        -   **Timestamp & Clock Security:**
            -   *How:*
                -   **Authenticated NTP:** Configure NTP clients on all devices (local and VPS) to use multiple trusted time sources and enable NTP authentication (e.g., using symmetric keys) to prevent spoofing of time sources.
                -   **Vector Clock Validation:** When a replica receives an update with a vector clock, it must verify that the sender's clock component has incremented correctly (typically by 1) and that other components in the received clock are greater than or equal to its own corresponding components. Any deviation might indicate a problem or malicious activity.
                    -   *Conceptual check for replica `R` receiving update from sender `S` with vector clock `VC_S` (its own clock is `VC_R`):*
                        `VC_S[S] == VC_R_before_merge[S] + 1` (if `R` had `S`'s previous value)
                        `forall i != S: VC_S[i] <= VC_R_before_merge[i]` (sender shouldn't advance others' clocks beyond what receiver knows)
                -   **Signed Updates (Optional):** For high-value data, the data payload and its associated metadata (including vector clock or timestamp) can be digitally signed by the originating device. This allows verification of authenticity and integrity, though it adds computational overhead.
        -   **CRDT Robustness & Validation:**
            -   *How:*
                -   **Server-Side Validation (VPS):** The VPS, as the L3 store and central synchronization point, must validate incoming CRDT states or operations. This includes checking for type conformity, value constraints, and adherence to the specific CRDT's invariants before merging or storing.
                -   **Conflict Logging & Metrics:** Log instances where CRDT convergence results in non-trivial merges, potential data overwrites, or unexpected outcomes. Monitor metrics like the frequency of conflicts or divergence rates.
                -   **Bounded History/Undo:** For some CRDTs, particularly state-based ones, consider keeping a bounded history of states or operations to allow for rollback or analysis in case of data corruption or erroneous merges.
        -   **Merkle Tree & Hashing Integrity:**
            -   *How:*
                -   **Authenticated Merkle Trees:** Instead of just hashing data, each node in the Merkle tree can also incorporate a Message Authentication Code (MAC) using a shared secret (if only two parties need to verify), or be digitally signed if public verifiability is required. This prevents an attacker from constructing a valid-looking tree with malicious data.
                -   **Delta Validation & Proofs:** When transmitting deltas (differences), the sender should also provide Merkle proofs (the sibling hashes along the path to the root) for the changed data. The receiver can then use these proofs to verify that the deltas correctly correspond to the claimed Merkle root hash without needing the entire dataset.
        -   **Multi-Layered Data Validation:**
            -   *How:* Implement validation at each significant boundary:
                -   **Local Client (UI/Input):** Basic format validation (e.g., email regex, number ranges), primarily for good UX and catching simple errors early.
                -   **Local Sync Agent:** Schema validation against expected data structures, basic consistency checks against its own cached state before attempting to sync.
                -   **VPS API (Backend):** Comprehensive business logic validation, referential integrity checks against the L3 database, enforcement of security policies (e.g., user permissions for the data), and checks for data invariants.
                -   *Example Flow for a new memory item:* User types -> Client UI (checks for max length) -> Sync Agent (validates against memory item schema, checks for duplicates in local hot cache) -> VPS API (verifies user auth, checks against project quotas, ensures linked entities exist) -> L3 Store.

-   **Security:** Protecting data on local devices and during synchronization.
    -   **Solutions:**
        -   **End-to-End Encryption (E2EE):**
            -   *How:*
                -   **Local Data Encryption:** Use libraries like `SQLCipher` for encrypting local SQLite databases, or leverage full-disk encryption (BitLocker, LUKS). For specific sensitive fields within application data, use AES-GCM with unique per-item nonces.
                -   **Key Management:**
                    -   *Local Keys:* Data Encryption Keys (DEKs) can be derived from a user's passphrase using a strong KDF (PBKDF2, Argon2id). This master key can then be stored in the OS keychain or, ideally, protected by a TPM/Secure Enclave if available (the TPM would wrap/unwrap the master key).
                    -   *Synchronization Keys:* For E2EE of synced data, a shared secret key per user (or per group of memories) is needed. This key could be derived from the user's master key or established using a secure key exchange protocol (e.g., a simplified version of Signal's Double Ratchet for pairwise E2EE with the VPS acting as a message forwarder, or a symmetric key encrypted with the user's public key for data at rest on the VPS).
                -   **Transport Layer Security (TLS):** All communication between the local sync agent and the VPS API must use TLS 1.3 (or latest recommended version).
        -   **Side-Channel Mitigation:**
            -   *How:*
                -   **Constant-Time Operations:** When implementing or selecting cryptographic libraries, ensure that functions handling secret data (keys, plaintexts) execute in constant time, meaning their execution time and memory access patterns do not depend on the secret values.
                -   **Blinding:** For asymmetric cryptographic operations like RSA decryption or signing, introduce a random "blinding factor" at the beginning of the computation and remove its effect at the end. This randomizes intermediate values, making it harder to deduce secret information from side channels like power consumption or timing.
        -   **Hardware Security Module (HSM) Utilization (TPM/Secure Enclave):**
            -   *How:*
                -   **Platform APIs:** Utilize platform-specific APIs: Trusted Platform Module (TPM) via TSS (TCG Software Stack) on Linux/Windows, Secure Enclave via Keychain services on macOS/iOS.
                -   **Key Operations:** Store private keys (e.g., for E2EE data decryption, document signing, or device identity) within the HSM. Perform cryptographic operations (sign, decrypt) directly inside the HSM so that the keys are never exposed in the main system memory.
                -   **Device Attestation:** Use the HSM to generate a cryptographic quote (a signed statement about the device's boot state and configuration) that can be sent to the VPS to prove the device's integrity during registration or critical operations.
        -   **Zero-Knowledge Proof (ZKP) Integrity:**
            -   *How:*
                -   **Established Libraries & Schemes:** Use well-vetted ZKP libraries and cryptographic schemes (e.g., Groth16, PLONK, STARKs via libraries like `ZoKrates`, `Circom/snarkjs`, `arkworks`).
                -   **Parameter Generation (Trusted Setup):** For zk-SNARKs requiring a per-circuit trusted setup (like Groth16), ensure the "toxic waste" (randomness used in the ceremony) is securely destroyed by all participants. Alternatively, use schemes that have a universal or transparent setup.
                -   **Rigorous Proof Validation:** The verifier (e.g., VPS or another client) must meticulously validate proofs against the known circuit definition and public inputs, ensuring all cryptographic checks pass.
        -   **Differential Privacy (DP) Robustness:**
            -   *How:*
                -   **Calibrated Noise Addition:** Add noise drawn from a specific distribution (e.g., Laplace for L1 sensitivity, Gaussian for L2 sensitivity) to query results or aggregated data. The scale of the noise (e.g., `lambda` for Laplace) is calibrated based on the query's sensitivity (how much one individual's data can change the result) and the chosen privacy budget (epsilon).
                -   **Privacy Budget Accounting:** Implement a mechanism to track the cumulative epsilon spent for each user or dataset across multiple queries. This is crucial to prevent re-identification through repeated querying.
                -   **Local vs. Central DP:** For stronger privacy, consider Local Differential Privacy where each client adds noise to its data *before* sending it to the VPS for aggregation. This protects data even from the VPS itself.

-   **Resource Management:** Efficiently deciding what to process locally versus remotely, and protecting against resource exhaustion.
    -   **Solutions:**
        -   **Secure & Resilient Schedulers:**
            -   *How:*
                -   **Input Sanitization for ML Schedulers:** If an RL scheduler learns from user interaction patterns or system telemetry, sanitize these inputs to remove outliers or detect patterns indicative of adversarial manipulation aiming to poison the training data.
                -   **Anomaly Detection in Scheduling:** Monitor scheduling decisions against baseline performance or expected behavior. For example, if local NPU is available and suitable for a task, but the scheduler consistently routes it to the VPS without a clear reason (e.g., higher load on NPU), flag this as an anomaly.
                -   **Quotas & Rate Limiting:** Implement using token bucket or leaky bucket algorithms at API gateways or within the scheduler itself for tasks originating from different sources, users, or local components. This prevents any single entity from monopolizing resources.
        -   **Adaptive Predictive Budgeting:**
            -   *How:*
                -   **Circuit Breaker Pattern:** Implement using libraries (e.g., `Hystrix` in Java (maintenance mode), `resilience4j` in Java, `Polly` in .NET, or custom logic in Python). If the error rate or latency from a predictive resource model exceeds a configured threshold for a defined period, the circuit "trips," and the system falls back to a more stable, static resource allocation strategy.
                -   **Ensemble of Models:** Instead of relying on a single predictive model, use an ensemble of diverse models. If one model starts behaving erratically or provides outlier predictions, its influence can be down-weighted or ignored by a meta-learner or voting mechanism.
        -   **System Stability & Monitoring:**
            -   *How:*
                -   **Comprehensive Monitoring:** Utilize tools like Prometheus for metrics collection, Grafana for visualization, and ELK/Loki stack for logging. Monitor key performance indicators (KPIs) for local (CPU, NPU, memory, disk I/O, thermal) and remote resources.
                -   **Throttling & Backpressure:** Implement API rate limiting at the VPS. For internal distributed pipelines, use backpressure mechanisms: if a downstream component's processing queue is full, it should signal upstream components to slow down or stop sending new data temporarily.
                -   **Load Shedding:** Design the system to gracefully degrade service under extreme load. This involves identifying lower-priority tasks or requests that can be dropped or deferred to preserve resources for critical functions.
        -   **Secure Agent Communication & Consensus (for Swarm Intelligence):**
            -   *How:*
                -   **Authenticated Channels:** Use mutual TLS (mTLS) for all agent-to-agent communication to ensure authenticity and encryption. Each agent would have its own certificate.
                -   **Reputation Systems:** Agents can maintain local reputation scores for other agents based on past interactions (e.g., timeliness, accuracy of information). Agents consistently providing bad data or failing liveness checks can be progressively isolated or their inputs down-weighted.
                -   **Resilient Consensus Algorithms:** If swarm agents need to agree on routing decisions or shared state, use consensus algorithms designed for potentially adversarial environments (e.g., Byzantine Fault Tolerant protocols like a permissioned variant of Honey Badger BFT or Raft with appropriate leader election and log replication security).

-   **Cross-Device Sync:** If the user has multiple local devices (e.g., laptop, desktop), managing consistent L2 caches and state across them.
    -   **Solutions:**
        -   **Centralized Source of Truth:**
            -   *How:* The VPS L3 storage (e.g., PostgreSQL + ChromaDB) must always be treated as the canonical store. Local caches (L1/L2) on devices are considered replicas that may be temporarily stale but aim for eventual consistency. All permanent state changes must be committed to the VPS.
        -   **Optimistic Replication with Conflict Resolution:**
            -   *How:* Devices sync their changes to the VPS independently. The VPS is responsible for detecting and resolving conflicts.
                -   **Versioning:** Each data item (or memory entry) should have a version number (e.g., an integer counter incremented on each update) or a reliable timestamp from a globally synchronized clock.
                -   **Conflict Detection on VPS:** A conflict occurs when an incoming update from a device has a version that is older than, or concurrent with (but different content), the version stored on the VPS.
                -   **Resolution Strategies (handled by VPS):**
                    -   *Last-Write-Wins (LWW):* Simplest strategy, based on timestamps. Requires reliable, well-synchronized clocks across all devices and the VPS. The update with the newest timestamp "wins."
                    -   *CRDT-based Merge Logic:* If data types are modeled as CRDTs, the VPS applies the defined merge function to conflicting updates.
                    -   *Application-Specific Merge Logic:* For complex data structures, the VPS might store conflicting versions temporarily and flag them for user resolution via a client UI, or apply predefined business rules (e.g., "append comments, overwrite main content if LWW").
                    -   *Conceptual LWW on VPS:*
                        ```python
                        # Assume item has 'id', 'content', 'timestamp_utc', 'version'
                        def handle_device_update(item_update):
                            vps_item = db.get_item(item_update.id)
                            if not vps_item or \
                               item_update.timestamp_utc > vps_item.timestamp_utc:
                                db.store_item(item_update) # Store new or updated item
                                return "accepted"
                            elif item_update.timestamp_utc == vps_item.timestamp_utc and \
                                 item_update.content_hash != vps_item.content_hash:
                                # Concurrent update, potentially resolve or flag
                                return handle_concurrent_update(vps_item, item_update)
                            else:
                                return "rejected_stale" # Update is older
                        ```
        -   **Device-Specific Caching Strategies:**
            -   *How:*
                -   **User Configuration/System Learning:** Allow users to define caching profiles per device (e.g., "Laptop: cache full active projects, last 30 days", "Phone: cache only pinned items, today's calendar-linked memories"). The system could also learn typical access patterns per device to optimize these profiles.
                -   **Sync Agent Logic:** The local sync agent on each device uses this profile to determine what data to proactively fetch from the VPS, what to retain in its L2 cache, and what to prune to save space.
        -   **Secure Device Quorum/Consensus (for sensitive cross-device operations):**
            -   *How:* For operations like authorizing a new device to access the user's MEMSHADOW account, if policy requires M-of-N existing devices to approve:
                -   **Device Registration & Attestation:** Each device is cryptographically registered with the user's account on the VPS, potentially using its TPM/Secure Enclave to generate and store a unique device key.
                -   **VPS as Orchestrator:** The VPS initiates the quorum approval. It sends a request (e.g., "Approve new device X?") to N online, registered devices. Each device UI prompts the user. If approved, the device signs the approval message with its device key and sends it back to the VPS. The VPS collects M signed approvals before finalizing the operation.
        -   **Authenticated Device Registration & Sync:**
            -   *How:*
                -   **Initial Device Registration:** User logs into their MEMSHADOW account on the VPS (e.g., via web). VPS provides a temporary, one-time registration token/QR code. On the new device, the MEMSHADOW client is installed. User enters the token. The client generates a local keypair (private key stored securely, ideally in TPM/Enclave). Public key is sent to VPS and associated with the user's account and this specific device.
                -   **Ongoing Sync Authentication:** Each sync request from the device to the VPS must be authenticated. This can be done by signing the request (or parts of it, like key headers and a hash of the payload) with the device's private key. Alternatively, the device can exchange its long-term credential for a short-lived session token (e.g., JWT) from the VPS.

This hybrid local-cloud architecture offers a powerful and flexible approach to building the MEMSHADOW system, catering to modern hardware capabilities while retaining the benefits of a centralized backend.
