"""
Prometheus Metrics for MEMSHADOW
Classification: UNCLASSIFIED
Exposes operational metrics for monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
from functools import wraps
import time
import structlog

logger = structlog.get_logger()

# Application info
memshadow_info = Info('memshadow', 'MEMSHADOW application information')
memshadow_info.info({
    'version': '2.1',
    'classification': 'UNCLASSIFIED',
    'component': 'offensive_security_platform'
})

# API Metrics
http_requests_total = Counter(
    'memshadow_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'memshadow_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# C2 Metrics
c2_sessions_active = Gauge(
    'memshadow_c2_sessions_active',
    'Number of active C2 sessions'
)

c2_sessions_total = Counter(
    'memshadow_c2_sessions_total',
    'Total C2 sessions created'
)

c2_tasks_sent = Counter(
    'memshadow_c2_tasks_sent_total',
    'Total C2 tasks sent',
    ['task_type']
)

c2_tasks_completed = Counter(
    'memshadow_c2_tasks_completed_total',
    'Total C2 tasks completed',
    ['task_type', 'status']
)

c2_data_exfiltrated_bytes = Counter(
    'memshadow_c2_data_exfiltrated_bytes_total',
    'Total bytes exfiltrated via C2'
)

c2_session_duration_seconds = Histogram(
    'memshadow_c2_session_duration_seconds',
    'C2 session duration in seconds',
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400)
)

# Mission Metrics
missions_total = Counter(
    'memshadow_missions_total',
    'Total missions executed',
    ['status']
)

missions_active = Gauge(
    'memshadow_missions_active',
    'Number of active missions'
)

mission_duration_seconds = Histogram(
    'memshadow_mission_duration_seconds',
    'Mission duration in seconds',
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400)
)

vulnerabilities_found = Counter(
    'memshadow_vulnerabilities_found_total',
    'Total vulnerabilities discovered',
    ['severity']
)

iocs_detected = Counter(
    'memshadow_iocs_detected_total',
    'Total IOCs detected',
    ['threat_level']
)

# SWARM Agent Metrics
swarm_agents_deployed = Gauge(
    'memshadow_swarm_agents_deployed',
    'Number of deployed SWARM agents',
    ['agent_type']
)

swarm_tasks_queued = Gauge(
    'memshadow_swarm_tasks_queued',
    'Number of queued SWARM tasks',
    ['agent_type']
)

swarm_tasks_completed = Counter(
    'memshadow_swarm_tasks_completed_total',
    'Total SWARM tasks completed',
    ['agent_type', 'status']
)

# LureCraft Metrics
lurecraft_campaigns_active = Gauge(
    'memshadow_lurecraft_campaigns_active',
    'Number of active phishing campaigns'
)

lurecraft_emails_sent = Counter(
    'memshadow_lurecraft_emails_sent_total',
    'Total phishing emails sent',
    ['template']
)

lurecraft_clicks = Counter(
    'memshadow_lurecraft_clicks_total',
    'Total payload clicks',
    ['template']
)

lurecraft_compromises = Counter(
    'memshadow_lurecraft_compromises_total',
    'Total successful compromises',
    ['template']
)

lurecraft_success_rate = Gauge(
    'memshadow_lurecraft_success_rate',
    'Campaign success rate percentage',
    ['campaign_id']
)

# Database Metrics
db_connections_active = Gauge(
    'memshadow_db_connections_active',
    'Number of active database connections',
    ['database']
)

db_query_duration_seconds = Histogram(
    'memshadow_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# Redis Metrics
redis_operations_total = Counter(
    'memshadow_redis_operations_total',
    'Total Redis operations',
    ['operation']
)

redis_keys_total = Gauge(
    'memshadow_redis_keys_total',
    'Total number of Redis keys'
)

# Vector Database Metrics
chromadb_collections = Gauge(
    'memshadow_chromadb_collections',
    'Number of ChromaDB collections'
)

chromadb_documents = Gauge(
    'memshadow_chromadb_documents_total',
    'Total documents in ChromaDB'
)

chromadb_query_duration_seconds = Histogram(
    'memshadow_chromadb_query_duration_seconds',
    'ChromaDB query duration in seconds',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# System Metrics
system_memory_usage_bytes = Gauge(
    'memshadow_system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_cpu_usage_percent = Gauge(
    'memshadow_system_cpu_usage_percent',
    'System CPU usage percentage'
)

# Sentinel Metrics
sentinel_alerts_total = Counter(
    'memshadow_sentinel_alerts_total',
    'Total Sentinel alerts',
    ['severity']
)

sentinel_monitors_active = Gauge(
    'memshadow_sentinel_monitors_active',
    'Number of active Sentinel monitors'
)


def track_request_metrics(func):
    """Decorator to track HTTP request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        # Extract request info (simplified)
        method = kwargs.get('method', 'UNKNOWN')
        endpoint = kwargs.get('endpoint', func.__name__)

        try:
            result = await func(*args, **kwargs)
            status = 'success'
            status_code = 200
        except Exception as e:
            status = 'error'
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

        return result

    return wrapper


async def get_metrics() -> Response:
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


async def update_system_metrics():
    """Update system resource metrics"""
    try:
        import psutil

        # Memory usage
        memory = psutil.virtual_memory()
        system_memory_usage_bytes.set(memory.used)

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_cpu_usage_percent.set(cpu_percent)

    except ImportError:
        logger.warning("psutil not installed, system metrics unavailable")
    except Exception as e:
        logger.error("Error updating system metrics", error=str(e))


async def update_c2_metrics(controller):
    """Update C2 framework metrics from controller"""
    try:
        # Get active sessions
        active_sessions = controller.get_active_sessions()
        c2_sessions_active.set(len(active_sessions))

        # Session stats
        for session in active_sessions:
            c2_data_exfiltrated_bytes.inc(session.data_exfiltrated_bytes)

    except Exception as e:
        logger.error("Error updating C2 metrics", error=str(e))


async def update_swarm_metrics(blackboard):
    """Update SWARM agent metrics from blackboard"""
    try:
        agents = ['recon', 'crawler', 'apimapper', 'authtest', 'bgp', 'blockchain', 'wifi', 'webscan']

        for agent in agents:
            task_count = blackboard.get_task_count(agent)
            swarm_tasks_queued.labels(agent_type=agent).set(task_count)

    except Exception as e:
        logger.error("Error updating SWARM metrics", error=str(e))
