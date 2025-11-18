// MEMSHADOW Performance Test
// Classification: UNCLASSIFIED
// k6 load testing script

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const requestCount = new Counter('request_count');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 10 },   // Stay at 10 users
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500'], // 95% of requests must complete below 500ms
    'errors': ['rate<0.1'],              // Error rate must be below 10%
    'http_req_failed': ['rate<0.05'],    // Failed requests must be below 5%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testScenarios = {
  health: `${BASE_URL}/api/v1/health`,
  metrics: `${BASE_URL}/api/v1/metrics`,
  dashboard: `${BASE_URL}/api/v1/tempest/dashboard/status`,
  c2Sessions: `${BASE_URL}/api/v1/c2/sessions`,
  iocs: `${BASE_URL}/api/v1/tempest/dashboard/intel/iocs`,
  vulnerabilities: `${BASE_URL}/api/v1/tempest/dashboard/intel/vulnerabilities`,
};

export default function () {
  // Health Check
  group('Health Checks', function () {
    const healthRes = http.get(testScenarios.health);
    const healthCheck = check(healthRes, {
      'health status is 200': (r) => r.status === 200,
      'health response time < 200ms': (r) => r.timings.duration < 200,
    });
    errorRate.add(!healthCheck);
    apiLatency.add(healthRes.timings.duration);
    requestCount.add(1);
  });

  sleep(1);

  // Metrics Endpoint
  group('Prometheus Metrics', function () {
    const metricsRes = http.get(testScenarios.metrics);
    const metricsCheck = check(metricsRes, {
      'metrics status is 200': (r) => r.status === 200,
      'metrics has content': (r) => r.body.length > 0,
    });
    errorRate.add(!metricsCheck);
    apiLatency.add(metricsRes.timings.duration);
    requestCount.add(1);
  });

  sleep(1);

  // Dashboard Status
  group('Dashboard API', function () {
    const dashboardRes = http.get(testScenarios.dashboard);
    const dashboardCheck = check(dashboardRes, {
      'dashboard status is 200': (r) => r.status === 200,
      'dashboard has missions': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.missions !== undefined;
        } catch (e) {
          return false;
        }
      },
    });
    errorRate.add(!dashboardCheck);
    apiLatency.add(dashboardRes.timings.duration);
    requestCount.add(1);
  });

  sleep(1);

  // C2 Sessions List
  group('C2 Framework API', function () {
    const c2Res = http.get(testScenarios.c2Sessions);
    const c2Check = check(c2Res, {
      'c2 sessions status is 200': (r) => r.status === 200,
      'c2 sessions response time < 500ms': (r) => r.timings.duration < 500,
    });
    errorRate.add(!c2Check);
    apiLatency.add(c2Res.timings.duration);
    requestCount.add(1);
  });

  sleep(1);

  // Intelligence - IOCs
  group('Intelligence API - IOCs', function () {
    const iocsRes = http.get(testScenarios.iocs);
    const iocsCheck = check(iocsRes, {
      'iocs status is 200': (r) => r.status === 200,
      'iocs has data structure': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.classification !== undefined;
        } catch (e) {
          return false;
        }
      },
    });
    errorRate.add(!iocsCheck);
    apiLatency.add(iocsRes.timings.duration);
    requestCount.add(1);
  });

  sleep(1);

  // Intelligence - Vulnerabilities
  group('Intelligence API - Vulnerabilities', function () {
    const vulnRes = http.get(testScenarios.vulnerabilities);
    const vulnCheck = check(vulnRes, {
      'vulnerabilities status is 200': (r) => r.status === 200,
      'vulnerabilities response time < 1s': (r) => r.timings.duration < 1000,
    });
    errorRate.add(!vulnCheck);
    apiLatency.add(vulnRes.timings.duration);
    requestCount.add(1);
  });

  sleep(2);
}

export function handleSummary(data) {
  return {
    'summary.html': htmlReport(data),
    'summary.json': JSON.stringify(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function htmlReport(data) {
  const metrics = data.metrics;
  const thresholds = data.thresholds;

  let html = `
<!DOCTYPE html>
<html>
<head>
  <title>MEMSHADOW Performance Test Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    .banner { background: #000; color: #fff; padding: 20px; text-align: center; }
    .container { background: #fff; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1, h2 { color: #333; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #4CAF50; color: white; }
    .pass { color: green; font-weight: bold; }
    .fail { color: red; font-weight: bold; }
    .metric { font-size: 2em; font-weight: bold; color: #4CAF50; }
  </style>
</head>
<body>
  <div class="banner">
    <h1>MEMSHADOW v2.1 Performance Test Report</h1>
    <p>Classification: UNCLASSIFIED</p>
  </div>

  <div class="container">
    <h2>Test Summary</h2>
    <table>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
      <tr>
        <td>Total Requests</td>
        <td class="metric">${metrics.http_reqs.values.count}</td>
      </tr>
      <tr>
        <td>Failed Requests</td>
        <td>${metrics.http_req_failed.values.rate * 100}%</td>
      </tr>
      <tr>
        <td>Average Duration</td>
        <td>${metrics.http_req_duration.values.avg.toFixed(2)}ms</td>
      </tr>
      <tr>
        <td>95th Percentile Duration</td>
        <td>${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms</td>
      </tr>
      <tr>
        <td>Error Rate</td>
        <td>${(metrics.errors.values.rate * 100).toFixed(2)}%</td>
      </tr>
    </table>
  </div>

  <div class="container">
    <h2>Threshold Results</h2>
    <table>
      <tr>
        <th>Threshold</th>
        <th>Status</th>
      </tr>
  `;

  for (const [name, result] of Object.entries(thresholds)) {
    const status = result.ok ? '<span class="pass">✓ PASS</span>' : '<span class="fail">✗ FAIL</span>';
    html += `<tr><td>${name}</td><td>${status}</td></tr>`;
  }

  html += `
    </table>
  </div>

  <div class="banner">
    <p>End of Report - Classification: UNCLASSIFIED</p>
  </div>
</body>
</html>
  `;

  return html;
}

function textSummary(data, opts) {
  let summary = '\n';
  summary += '================================================================================\n';
  summary += '  MEMSHADOW v2.1 Performance Test Summary\n';
  summary += '  Classification: UNCLASSIFIED\n';
  summary += '================================================================================\n\n';

  const metrics = data.metrics;

  summary += `Total Requests:       ${metrics.http_reqs.values.count}\n`;
  summary += `Failed Requests:      ${(metrics.http_req_failed.values.rate * 100).toFixed(2)}%\n`;
  summary += `Average Duration:     ${metrics.http_req_duration.values.avg.toFixed(2)}ms\n`;
  summary += `95th Percentile:      ${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += `Error Rate:           ${(metrics.errors.values.rate * 100).toFixed(2)}%\n\n`;

  summary += 'Thresholds:\n';
  for (const [name, result] of Object.entries(data.thresholds)) {
    const status = result.ok ? '✓ PASS' : '✗ FAIL';
    summary += `  ${status} ${name}\n`;
  }

  summary += '\n================================================================================\n';

  return summary;
}
