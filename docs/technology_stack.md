# Project MEMSHADOW Technology Stack

This document outlines the primary programming languages, databases, frameworks, and tools used or proposed for Project MEMSHADOW.

## 1. Backend Services & API

*   **Programming Language:** Python 3.9+
    *   **Rationale:** Rich ecosystem for web development, data science, machine learning, and AI. Rapid prototyping and development speed.
*   **Web Framework:** FastAPI
    *   **Rationale:** High performance (ASGI based on Starlette and Pydantic), automatic data validation, OpenAPI/Swagger UI generation, dependency injection, asynchronous support.
*   **Task Queuing:** Celery with Redis as a broker/backend
    *   **Rationale:** Robust distributed task queue system for handling asynchronous operations like embedding generation, enrichment, and notifications. Redis provides a fast and reliable message broker.
*   **ASGI Server:** Uvicorn
    *   **Rationale:** Lightning-fast ASGI server, recommended for FastAPI.

## 2. Data Storage

*   **Metadata Database:** PostgreSQL (version 15+)
    *   **Rationale:** Powerful, open-source relational database with strong support for JSONB (for flexible metadata), full-text search, and geospatial data (if needed in future). Reliable and scalable.
*   **Vector Database:** ChromaDB
    *   **Rationale:** Open-source vector database specifically designed for AI applications, enabling efficient semantic search and similarity queries on embeddings.
*   **Cache:** Redis
    *   **Rationale:** In-memory data store used for caching frequently accessed data, session management, rate limiting, and as a Celery message broker.
*   **Object Storage:** S3-compatible (e.g., AWS S3, MinIO)
    *   **Rationale:** For storing large binary artifacts, original documents, or other bulky data linked to memories, keeping primary databases lean and focused.
*   **Secure Archival Storage (for SDAP):** Operator-controlled server, filesystem based.
    *   **Rationale:** Data sovereignty and security for backups.

## 3. Frontend & Client-Side

*   **`memcli` (CLI Client):** Python
    *   **Libraries:** `click` (for CLI structure), `httpx` (for API communication), `npyscreen` (for TUI), `rich` (for enhanced console output).
    *   **Rationale:** Python is well-suited for CLI tools and can easily interact with the Python backend.
*   **Browser Extension:** JavaScript/TypeScript
    *   **Frameworks/Libraries:** Standard WebExtensions API. Potentially a lightweight framework like Preact or Svelte for UI components if complexity grows.
    *   **Rationale:** Native technologies for browser integration.
*   **API Client Library (if developed):** Python (initially), potentially other languages based on demand.

## 4. Machine Learning & NLP

*   **Primary Language:** Python
*   **Embedding Models:**
    *   Sentence Transformers (e.g., `all-mpnet-base-v2`): For local, open-source embedding generation.
    *   OpenAI Embeddings (e.g., `text-embedding-ada-002`): For high-quality remote embedding generation.
    *   Future: Code-specific models (CodeBERT, UniXcoder), multilingual models.
*   **NLP Libraries:**
    *   spaCy: For industrial-strength NLP tasks like Named Entity Recognition (NER), part-of-speech tagging, dependency parsing.
    *   Transformers (Hugging Face): For accessing a wide variety of pre-trained models for summarization, sentiment analysis, Q&A, etc.
*   **Local LLM Processing:**
    *   Libraries like `llama-cpp-python` for running GGUF-quantized models (e.g., Llama, Phi-3).
    *   Transformers library for running smaller models locally (e.g., Gemma, Phi-3 mini).
*   **ML Frameworks:**
    *   PyTorch: Often used by Sentence Transformers and many Hugging Face models.
    *   TensorFlow: (Less emphasis currently but could be used by specific models).
    *   Scikit-learn: For classical ML tasks, clustering, evaluation metrics.
*   **NPU Acceleration (Local Device):**
    *   ONNX Runtime: For running optimized models on NPUs that support ONNX.
    *   OpenVINO (Intel): For Intel NPUs.
    *   Core ML (Apple): For Apple Silicon NPUs.
    *   (Specific libraries depend on target NPU hardware).

## 5. Security

*   **Authentication:**
    *   FIDO2/WebAuthn: Libraries like `webauthn-lib` (Python backend), `fido2` (Python client).
    *   JWT (JSON Web Tokens): `python-jose` for encoding/decoding.
*   **Password Hashing:** Argon2 (via `argon2-cffi`), bcrypt (via `passlib`).
*   **Encryption:**
    *   GPG (GNU Privacy Guard): For SDAP archive encryption.
    *   Cryptography (Python library): For AES, RSA, Fernet (symmetric encryption).
*   **Static Analysis Security Testing (SAST):** Bandit (for Python).
*   **Dependency Vulnerability Scanning:** `pip-audit`.
*   **Container Vulnerability Scanning:** Trivy.

## 6. DevOps & Infrastructure

*   **Containerization:** Docker, Docker Compose
    *   **Rationale:** Consistent development and deployment environments, microservices architecture.
*   **CI/CD:** GitLab CI (as per examples), Jenkins, GitHub Actions, etc.
    *   **Rationale:** Automated building, testing, and deployment.
*   **Infrastructure as Code (IaC) (Production):** Terraform
    *   **Rationale:** Manage cloud infrastructure programmatically.
*   **Orchestration (Production):** Kubernetes (EKS as per Terraform example)
    *   **Rationale:** Scalable and resilient deployment of containerized applications.
*   **Reverse Proxy/Load Balancer:** NGINX
    *   **Rationale:** Efficiently handle HTTP traffic, SSL termination, load balancing.
*   **Monitoring & Observability:**
    *   Prometheus: For metrics collection.
    *   Grafana: For dashboards and visualization.
    *   Loki/Promtail: For log aggregation (mentioned in Docker Compose).
    *   Fluentd: For log aggregation (mentioned for SWARM Arena).
    *   OpenTelemetry: For distributed tracing and metrics.
    *   Structlog: For structured logging in Python.
*   **Operating System (VPS/Servers):** Linux (e.g., Ubuntu, Alpine for containers).
*   **Shell Scripting:** Bash (for `sdap_backup.sh` and other operational scripts).

## 7. Specialized Components

*   **HYDRA Swarm Agents & Coordinator:** Python
    *   **Rationale:** Ease of development, access to networking and security libraries.
*   **JANUS Protocol (Sealing Utility):** C/C++ or Rust could be considered if a standalone, high-performance, secure binary is needed for key derivation. Otherwise, Python or Bash for scripting existing tools (like `openssl kdf`).
*   **Code Dependency Graph (Claude Code Memory):** NetworkX (Python library).

## 8. Documentation

*   **Format:** Markdown
*   **Diagrams:** Mermaid.js

This technology stack is chosen to balance development speed, performance, scalability, security, and access to a rich ecosystem of tools and libraries, particularly in the Python data science and AI communities. Choices may evolve as the project matures and specific needs arise.
