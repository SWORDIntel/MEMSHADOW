-- MEMSHADOW Database Initialization Script
-- Classification: UNCLASSIFIED
-- Creates necessary extensions and initial schema

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- Create schema for MEMSHADOW
CREATE SCHEMA IF NOT EXISTS memshadow;

-- Set default schema
SET search_path TO memshadow, public;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(100) NOT NULL,
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255),
    user_id UUID,
    username VARCHAR(255),
    ip_address INET,
    status VARCHAR(50),
    details JSONB,
    severity VARCHAR(20),
    classification VARCHAR(50) DEFAULT 'UNCLASSIFIED',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on audit logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_severity ON audit_logs(severity);

-- Create C2 sessions table
CREATE TABLE IF NOT EXISTS c2_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    hostname VARCHAR(255),
    ip_address INET,
    os_info TEXT,
    username VARCHAR(255),
    encryption_key TEXT,
    status VARCHAR(50) DEFAULT 'active',
    established_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tasks_sent INTEGER DEFAULT 0,
    tasks_completed INTEGER DEFAULT 0,
    data_exfiltrated_bytes BIGINT DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on C2 sessions
CREATE INDEX IF NOT EXISTS idx_c2_sessions_agent_id ON c2_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_c2_sessions_status ON c2_sessions(status);
CREATE INDEX IF NOT EXISTS idx_c2_sessions_last_seen ON c2_sessions(last_seen DESC);

-- Create missions table
CREATE TABLE IF NOT EXISTS missions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mission_id VARCHAR(255) UNIQUE NOT NULL,
    mission_name VARCHAR(255) NOT NULL,
    classification VARCHAR(50) DEFAULT 'UNCLASSIFIED',
    status VARCHAR(50) DEFAULT 'pending',
    stages_completed INTEGER DEFAULT 0,
    stages_total INTEGER,
    vulnerabilities_found INTEGER DEFAULT 0,
    iocs_detected INTEGER DEFAULT 0,
    agents_deployed INTEGER DEFAULT 0,
    mission_data JSONB,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on missions
CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
CREATE INDEX IF NOT EXISTS idx_missions_started_at ON missions(started_at DESC);

-- Create indicators of compromise (IOC) table
CREATE TABLE IF NOT EXISTS iocs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ioc_type VARCHAR(50) NOT NULL,
    value TEXT NOT NULL,
    threat_level VARCHAR(20),
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context TEXT,
    mission_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on IOCs
CREATE INDEX IF NOT EXISTS idx_iocs_type ON iocs(ioc_type);
CREATE INDEX IF NOT EXISTS idx_iocs_threat_level ON iocs(threat_level);
CREATE INDEX IF NOT EXISTS idx_iocs_value_trgm ON iocs USING gin (value gin_trgm_ops);

-- Create vulnerabilities table
CREATE TABLE IF NOT EXISTS vulnerabilities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vuln_type VARCHAR(100) NOT NULL,
    cvss DECIMAL(3,1),
    url TEXT,
    description TEXT,
    remediation TEXT,
    mission_id VARCHAR(255),
    target_host VARCHAR(255),
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on vulnerabilities
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_cvss ON vulnerabilities(cvss DESC);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_type ON vulnerabilities(vuln_type);

-- Create LureCraft campaigns table
CREATE TABLE IF NOT EXISTS lurecraft_campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id VARCHAR(255) UNIQUE NOT NULL,
    campaign_name VARCHAR(255) NOT NULL,
    template VARCHAR(100),
    payload_url TEXT,
    emails_sent INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    compromises INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on campaigns
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON lurecraft_campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaigns_started_at ON lurecraft_campaigns(started_at DESC);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_c2_sessions_updated_at BEFORE UPDATE ON c2_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_missions_updated_at BEFORE UPDATE ON missions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON lurecraft_campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA memshadow TO memshadow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA memshadow TO memshadow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA memshadow TO memshadow;

-- Completion message
DO $$
BEGIN
    RAISE NOTICE 'MEMSHADOW database initialized successfully';
    RAISE NOTICE 'Classification: UNCLASSIFIED';
END $$;
