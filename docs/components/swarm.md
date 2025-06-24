# SWARM Project (HYDRA Phase 3 - Autonomous Agent Swarm)

**Document ID:** MEMSHADOW-OPLAN-001 (Adapted from SWARM.md)

The SWARM project represents Phase 3 ("Run") of PROJECT HYDRA. It details the operational plan for designing, deploying, and operationalizing a Minimum Viable Swarm (MVS) of autonomous agents. This swarm is intended to execute objective-based adversarial simulations against the MEMSHADOW system within a secure staging environment.

## 1. Executive Summary & Mission Objective

### 1.1 Mission
To deploy an MVS capable of automated reconnaissance and basic vulnerability analysis against the MEMSHADOW staging environment.

### 1.2 End State
A persistent, automated red team capability that continuously probes the staging environment, providing dynamic and adaptive security validation. This forms the foundation for advanced adversarial simulations.

### 1.3 Managing Complexity
Complexity is managed by defining five core subsystems:
1.  **The Arena:** Isolated operational environment.
2.  **The Coordinator:** Central swarm command and control (C2).
3.  **The Agents:** Distributed execution units.
4.  **The Communication Protocol:** Data bus connecting components.
5.  **The Objective Framework:** Mission definition language.

## 2. The Operational Environment: "The Arena"

The swarm operates in a secure, isolated, and instrumented environment.

*   **A. Network Isolation:** A dedicated Docker Compose network (`hydra_arena_net`) configured with `internal: true` to prevent external access.
*   **B. Target Deployment:** A production-identical replica of the MEMSHADOW stack (FastAPI, PostgreSQL, Redis, etc.) deployed within The Arena.
*   **C. Swarm Deployment:** All HYDRA swarm components (Coordinator, Agents) deployed as containers within the same isolated network.
*   **D. Instrumentation & Observability:** A dedicated logging aggregator (e.g., Fluentd container) captures `stdout/stderr` from all containers in The Arena, writing to a single time-series log file for analysis.

```mermaid
graph TD
    subgraph The Arena (hydra_arena_net - Docker Internal Network)
        direction LR
        subgraph MEMSHADOW Staging Stack
            MS_API[FastAPI]
            MS_DB[PostgreSQL]
            MS_Redis[Redis]
            MS_Other[Other Services]
        end

        subgraph HYDRA Swarm
            Coordinator[Coordinator Node]
            Agent_Recon[agent-recon]
            Agent_APIMapper[agent-apimapper]
            Agent_AuthTest[agent-authtest]
            Swarm_Redis[Swarm Redis (Blackboard)]
        end

        LogAggregator[Fluentd Log Aggregator]

        Coordinator --> Swarm_Redis
        Agent_Recon --> Swarm_Redis
        Agent_APIMapper --> Swarm_Redis
        Agent_AuthTest --> Swarm_Redis

        Agent_Recon --> MS_API
        Agent_APIMapper --> MS_API
        Agent_AuthTest --> MS_API

        MS_API --> LogAggregator
        Coordinator --> LogAggregator
        Agent_Recon --> LogAggregator
        Agent_APIMapper --> LogAggregator
        Agent_AuthTest --> LogAggregator
    end
    LogAggregator --> OutputLog[Time-Series Log File]
```

## 3. Swarm C2: The Coordinator Node

The centralized intelligence of the swarm; directs agents, does not perform attacks itself.

*   **A. Architecture:** Python application in its own Docker container. No exposed network ports. Communicates solely via the internal Swarm Redis instance.
*   **B. Core Functions:**
    1.  **Objective Ingest:** Reads a mission file (e.g., `mission.yaml`) mounted into its container on startup.
    2.  **State Management & "Blackboard":** Uses the dedicated Swarm Redis database as a mission blackboard. Posts tasks for agents and reads their results.
    3.  **Task Delegation:** Operates on an event loop: checks mission objectives, checks available agents, generates tasks, publishes tasks to a Redis task queue/channel.
    4.  **Reporting:** When mission `success_condition` is met or all paths are exhausted, generates a final Markdown mission report and terminates.

## 4. MVS Agent Roster & Design (Minimum Viable Swarm)

Agents are stateless, lightweight Docker containers from a common Python base image.

*   **Agent 1: `agent-recon`**
    *   **Capability:** Network Service Discovery.
    *   **Action:** Subscribes to tasks for network scanning. Uses `python-nmap` to scan a given CIDR block (e.g., The Arena's subnet) to identify hosts and open TCP ports.
    *   **Output:** Publishes a report to the blackboard (e.g., `{"task_id": "...", "agent_id": "recon-1", "status": "SUCCESS", "data": {"host": "172.19.0.5", "open_ports": [80, 5432]}}`).

*   **Agent 2: `agent-apimapper`**
    *   **Capability:** API Endpoint Enumeration.
    *   **Action:** Subscribes to tasks targeting a host/port from `agent-recon`. Attempts to fetch `openapi.json`. If fails, uses a predefined wordlist for common API paths (`/api`, `/v1`, `/auth`).
    *   **Output:** Publishes a list of discovered, valid API endpoints (not 404s) to the blackboard.

*   **Agent 3: `agent-authtest`**
    *   **Capability:** Basic Authentication & Authorization Testing.
    *   **Action:** Subscribes to tasks with a list of endpoints from `agent-apimapper`. For each endpoint, performs requests:
        *   `GET` with no `Authorization` header.
        *   `GET` with a malformed JWT.
        *   `GET` with a valid but low-privilege JWT.
    *   **Output:** Publishes a report classifying each endpoint (e.g., `{"endpoint": "/api/v1/health", "status_no_auth": 200, "status_malformed_jwt": 200, "status_low_priv_jwt": 200, "classification": "PUBLIC"}`).

## 5. Communication Protocol & Data Flow

Asynchronous communication via Redis Pub/Sub or Lists (acting as queues).

*   **A. Communication Channels (Redis Keys/Channels):**
    *   `hydra:tasks:{agent_type}` (e.g., `hydra:tasks:recon`): List used as a task queue for specific agent types. Coordinator pushes tasks here.
    *   `hydra:reports:ingress`: List used as a report queue. Agents push results here. Coordinator pulls from here.
    *   `hydra:blackboard:{key}`: General Redis keys for shared state, e.g., `hydra:blackboard:known_hosts`, `hydra:blackboard:discovered_endpoints:{host_id}`.

*   **B. Message Schema (JSON):**
    *   **Task Message (Coordinator -> Agents, via Redis List):**
        ```json
        {
          "task_id": "c7a8b2f0-1e8d-4a9f-8b2c-5d7e6f0a1b3d",
          "agent_type_target": "APIMapper", // Implicit from the list key, but good for clarity
          "payload": {
            "target_host": "172.19.0.5",
            "target_port": 80
            // ... other task-specific parameters
          },
          "timestamp": "2024-06-25T10:00:00Z"
        }
        ```
    *   **Report Message (Agents -> Coordinator, via Redis List):**
        ```json
        {
          "task_id": "c7a8b2f0-1e8d-4a9f-8b2c-5d7e6f0a1b3d",
          "agent_id": "agent-apimapper-03", // Unique ID of the agent instance
          "agent_type": "APIMapper",
          "status": "SUCCESS", // SUCCESS, FAILURE, PARTIAL_SUCCESS
          "data": {
            "discovered_endpoints": ["/api/v1/health", "/api/v1/ingest"]
            // ... other result data
          },
          "error_message": null, // If status is FAILURE
          "timestamp": "2024-06-25T10:05:00Z"
        }
        ```

*   **C. Architectural Flow (Simplified):**
    ```mermaid
    graph TD
        subgraph "The Arena (hydra_arena_net)"
            CoordinatorNode[Coordinator] -- "1. LPUSH Task to hydra:tasks:AgentType" --> SwarmRedis[("Swarm Redis<br>Task & Report Queues<br>Blackboard")]

            subgraph Agents
                direction LR
                ReconAgent[agent-recon]
                APIMapperAgent[agent-apimapper]
                AuthTestAgent[agent-authtest]
            end

            SwarmRedis -- "2. Agent BRPOP Task from hydra:tasks:AgentType" --> APIMapperAgent
            APIMapperAgent -- "3. Executes Task on Target" --> TargetServices[MEMSHADOW Staging Stack]
            APIMapperAgent -- "4. LPUSH Report to hydra:reports:ingress" --> SwarmRedis
            SwarmRedis -- "5. Coordinator BRPOP Report from hydra:reports:ingress" --> CoordinatorNode
            CoordinatorNode -- "6. Updates Blackboard" --> SwarmRedis
        end
    ```

## 6. OPLAN: First Mission - "Operation Initial Foothold"

The initial objective for the MVS.

*   **A. Mission Definition File (`initial_foothold.yaml`):**
    ```yaml
    mission_id: "MVS_Initial_Foothold_001"
    mission_name: "Initial Foothold v1"
    description: >
      Perform basic reconnaissance of The Arena to identify running services,
      enumerate API endpoints on the primary web service, and classify endpoint
      authentication requirements.
    objective_stages:
      - stage_id: "network_recon"
        description: "Identify all responsive hosts and their open ports in The Arena."
        tasks:
          - agent_type: "recon"
            params:
              target_cidr: "172.19.0.0/24" # Example, actual CIDR of hydra_arena_net
        success_criteria:
          - "blackboard_key_exists:known_hosts" # e.g., a list of {"host": ..., "ports": [...]}
          - "num_known_hosts >= 1"

      - stage_id: "api_mapping"
        description: "Enumerate all discoverable API endpoints on the primary web service."
        depends_on: "network_recon" # This stage runs after network_recon is successful
        tasks:
          - agent_type: "apimapper"
            # Parameters derived from 'known_hosts' (e.g., host with port 80/443/8000)
            params_from_blackboard:
              target_host_key: "primary_web_service_host" # Key where recon agent stores this
              target_port_key: "primary_web_service_port"
        success_criteria:
          - "blackboard_key_exists:discovered_api_endpoints:{primary_web_service_host}"
          - "num_discovered_endpoints >= 5" # Example minimum

      - stage_id: "auth_testing"
        description: "Classify the authentication requirements for each discovered API endpoint."
        depends_on: "api_mapping"
        tasks:
          - agent_type: "authtest"
            # Parameters: list of discovered endpoints
            params_from_blackboard:
              endpoints_key: "discovered_api_endpoints:{primary_web_service_host}"
        success_criteria:
          - "all_endpoints_auth_classified:{primary_web_service_host}" # Custom check by Coordinator

    overall_success_condition: >
      All stages completed successfully and their success_criteria met.
      Final report generated by the Coordinator.
    ```

*   **B. Execution Flow (Simplified):**
    1.  **Start:** Coordinator loads `initial_foothold.yaml`.
    2.  **Stage 1 (Network Recon):**
        *   Coordinator sees `network_recon` stage, creates `recon` task from `params`.
        *   Publishes task to `hydra:tasks:recon`.
        *   `agent-recon` picks up task, scans, publishes report to `hydra:reports:ingress`.
        *   Coordinator processes report, updates blackboard (e.g., `hydra:blackboard:known_hosts`, `hydra:blackboard:primary_web_service_host`). Checks `success_criteria`.
    3.  **Stage 2 (API Mapping):**
        *   If Stage 1 successful, Coordinator moves to `api_mapping`.
        *   Retrieves `target_host`, `target_port` from blackboard. Creates `apimapper` task.
        *   Publishes task. `agent-apimapper` executes, reports.
        *   Coordinator updates blackboard (e.g., `hydra:blackboard:discovered_api_endpoints:{host}`). Checks `success_criteria`.
    4.  **Stage 3 (Auth Testing):**
        *   If Stage 2 successful, Coordinator moves to `auth_testing`.
        *   Retrieves endpoint list from blackboard. Creates multiple `authtest` tasks.
        *   Publishes tasks. `agent-authtest` instances execute, report.
        *   Coordinator updates blackboard (e.g., `hydra:blackboard:auth_status:{endpoint}`). Checks `success_criteria`.
    5.  **Completion:** If all stages meet `success_criteria` and `overall_success_condition` is met, Coordinator compiles data from blackboard into a final Markdown report and terminates the mission.

This structured, OPLAN-driven approach for the SWARM project aims to validate the core swarm architecture effectively and pave the way for more advanced offensive agents and complex mission objectives. The MVS focuses on establishing the fundamental C2, communication, and agent execution capabilities.
