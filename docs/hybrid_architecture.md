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

## 7. Challenges

-   **Complexity:** Managing distributed state, synchronization, and potential conflicts.
-   **Data Consistency:** Ensuring consistency between local caches and the central VPS store.
-   **Security:** Protecting data on local devices and during synchronization.
-   **Resource Management:** Efficiently deciding what to process locally versus remotely.
-   **Cross-Device Sync:** If the user has multiple local devices, managing consistent L2 caches across them.

This hybrid local-cloud architecture offers a powerful and flexible approach to building the MEMSHADOW system, catering to modern hardware capabilities while retaining the benefits of a centralized backend.
