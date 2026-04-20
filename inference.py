"""
SREBench Inference Script — v2.0 (Enhanced)
===================================
MANDATORY
- Uses OpenAI Client for all LLM calls
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- Emits [START], [STEP], [END] structured stdout logs
- Named inference.py in the root directory

ENHANCEMENTS (v2.0):
- Scoring-aware system prompt with explicit bonus checklist
- Dynamic temperature scheduling (explore → exploit)
- Structured investigation state tracking (Reflexion-inspired)
- Postmortem template injection for keyword-rich documentation
- Task-adaptive prompt addons based on initial observations
- Few-shot exemplar for high-scoring investigation pattern
- Priority-based context compression preserving milestones
"""

import json
import os
import re
import sys
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Set these in your system or .env file ────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")

# Optional (for local Docker deployment)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# SREBench Space URL
SPACE_URL = os.getenv("SREBENCH_URL", "https://neuralninja110-srebench.hf.space")

BENCHMARK = "srebench"
SEED = 42

TASKS = [
    "task1_memory_leak",
    "task2_db_cascade",
    "task3_race_condition",
    "task4_dns_failure",
    "task5_cert_expiry",
    "task6_network_partition",
    "task7_kafka_lag",
    "task8_redis_failover",
    "task9_disk_full",
    "task10_rate_limit",
    "task11_db_migration_lock",
    "task12_health_flap",
    "task13_pod_eviction",
    "task14_cascading_timeout",
    "task15_secret_rotation",
    "task16_log_storm",
]

MAX_STEPS_MAP = {
    "task1_memory_leak": 20,
    "task2_db_cascade": 30,
    "task3_race_condition": 40,
    "task4_dns_failure": 25,
    "task5_cert_expiry": 35,
    "task6_network_partition": 40,
    "task7_kafka_lag": 25,
    "task8_redis_failover": 30,
    "task9_disk_full": 25,
    "task10_rate_limit": 25,
    "task11_db_migration_lock": 35,
    "task12_health_flap": 30,
    "task13_pod_eviction": 35,
    "task14_cascading_timeout": 30,
    "task15_secret_rotation": 25,
    "task16_log_storm": 35,
}


# ── System Prompt with Scoring Awareness ─────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SRE on-call engineer debugging a production incident.

## INVESTIGATION METHODOLOGY (follow this order STRICTLY)
1. ORIENT: list_services + check_alerts to understand scope
2. ASSESS: check_slo to see which services are burning error budget fastest, then classify_severity (SEV1-SEV4)
3. COMMUNICATE: update_status_page with initial status so stakeholders are informed
4. GATHER: read_logs + check_metrics on degraded services — focus on the MOST degraded service first
5. TRACE: trace_request to see distributed request flow — identify where latency/errors originate
6. TRACE DEPENDENCIES: check_dependencies on the degraded service AND on suspected root cause service (do this at least TWICE on different services)
7. CORRELATE: check_deployments to find recent changes. Compare deploy timestamps with error start times.
8. DIAGNOSE: run_diagnostic to confirm root cause. Match diagnostic type to symptoms (see PATTERN MATCHING below).
9. FIX: apply_fix targeting the ROOT CAUSE service, not symptom services. Include deploy_id when rolling back.
10. COMMUNICATE: update_status_page with resolution status
11. DOCUMENT: write_postmortem BEFORE verifying — MUST mention the specific root cause, affected services, and remediation steps in detail. Use ALL technical terms from your investigation.
12. VERIFY: verify_health AFTER documenting — this ENDS the episode, so do it LAST.

## SCORING COMPONENTS (maximize ALL of these)
1. Investigation milestones (50-55%): Hit ALL milestone flags in correct order
2. Evidence breadth (8%): Gather >=4 DISTINCT (action_type, service) pairs BEFORE fixing
3. Postmortem quality (6%): Include specific root cause keywords, affected services, timeline
4. Communication (4%): classify_severity + update_status_page BEFORE attempting fix
5. SLO awareness (2%): Fix before any SLO breaches
6. Time efficiency (5-10%): Complete within 50% of max steps
7. Blast radius (3%): check_dependencies on >=2 DIFFERENT services
8. Efficient investigation (4%): Identify root cause quickly

## BONUS CHECKLIST (ensure ALL are done before verify_health):
□ classify_severity (do this early, after initial investigation)
□ update_status_page (MUST be done BEFORE apply_fix for bonus points)
□ check_slo (during assessment phase)
□ check_dependencies on >=2 different services (blast radius assessment)
□ write_postmortem with detailed technical content (BEFORE verify_health)
□ Gather >=4 distinct evidence sources (different action+service combinations)

## CRITICAL RULES
- NEVER apply_fix as first action. You need at least 2 evidence sources first (investigation gating).
- When multiple services are degraded, the ROOT CAUSE is usually a LEAF service (databases). Errors CASCADE UPWARD.
- If errors started at time T, check what was deployed at T±5 minutes — timing correlation = likely cause.
- After ANY fix, ALWAYS: update_status_page → write_postmortem → verify_health (in that order).
- verify_health ENDS the episode — anything after it is ignored.

## SEVERITY CLASSIFICATION
- SEV1: Complete service outage or data loss affecting all users (payment down, auth completely broken)
- SEV2: Significant degradation affecting many users (intermittent errors, partial outages, data staleness)
- SEV3: Minor impact, workaround available
- SEV4: Cosmetic or negligible user impact

## PATTERN MATCHING (match symptoms to diagnostic type)
- "OOMKilled", "heap space", "memory exceeded" → memory leak → fix_type="restart"
- "connection pool exhausted", "HikariPool" → DB pool issue → trace to DATABASE → fix_type="increase_pool_size"
- "SSL handshake", "certificate expired", "TLS failed" → expired TLS cert → run_diagnostic type="tls" → fix_type="renew_cert"
- "DNS resolution failed", "NXDOMAIN", "stale DNS" → DNS cache → run_diagnostic type="dns" → fix_type="flush_dns"
- "connection timed out", "network unreachable", "split-brain" → network partition → run_diagnostic type="iptables" → fix_type="rollback_deploy" then reconcile_data
- "consumer lag", "offset behind", "kafka" → Kafka consumer lag → fix_type="restart_consumers" or "increase_partitions"
- "redis", "failover", "READONLY", "replication" → Redis failover → fix_type="trigger_failover" or "promote_replica"
- "disk", "no space", "filesystem full" → Disk full → fix_type="cleanup_disk" or "expand_volume"
- "rate limit", "429", "throttled" → Rate limiting → fix_type="increase_rate_limit" or "add_circuit_breaker"
- "migration", "lock", "blocking", "deadlock" → DB migration lock → fix_type="kill_migration" or "rollback_migration"
- "health check", "flapping", "CrashLoopBackOff" → Health check flap → fix_type="adjust_health_check" or "restart"
- "evict", "OOMKilled", "node pressure" → Pod eviction → fix_type="increase_resources" or "reschedule"
- "timeout", "cascading", "circuit breaker" → Cascading timeout → fix_type="increase_timeout" or "add_circuit_breaker"
- "secret", "rotation", "expired key", "401" → Secret rotation failure → fix_type="rotate_secrets" or "rollback_config"
- "log storm", "disk io", "logging flood" → Log storm → fix_type="adjust_log_level" or "add_rate_limiter"
- Error spike correlating with deploy → config change → run_diagnostic type="config_diff" WITH deploy_id → fix_type="rollback"

## CASCADE TRACING
When alert points to a gateway/frontend service:
1. check_dependencies on the alerted service
2. Trace toward LEAF services (databases)
3. check_metrics on EACH service in chain — highest error_rate = likely root cause
4. trace_request to visualise the full request flow and pinpoint exact failure span
Example: api-gateway → order-service → payment-service → payment-db(pool exhausted) = ROOT CAUSE

## RESPONSE FORMAT
Output exactly one JSON object per turn:
{
  "action_type": "<action>",
  "parameters": { ... },
  "reasoning": "What I observe → What I suspect → Why this action"
}

## AVAILABLE ACTIONS
- list_services: {} — List all services and statuses
- check_alerts: {} — View active alerts
- read_logs: {"service": "<name>"} — Read application logs
- check_metrics: {"service": "<name>", "metric": "<type>"} — Get service metrics
- check_deployments: {"last_n": 5} or {"service": "<name>"} — Recent deploys
- check_dependencies: {"service": "<name>"} — Service dependency graph
- run_diagnostic: {"service": "<name>", "type": "<diag_type>"} — Run diagnostics
- trace_request: {"service": "<name>"} — Distributed trace waterfall showing request flow
- check_slo: {} — SLO dashboard with error budget burn rates for all services
- classify_severity: {"severity": "SEV1|SEV2|SEV3|SEV4"} — Classify incident severity
- update_status_page: {"status": "investigating|identified|monitoring|resolved", "message": "<text>"} — Update public status page
- apply_fix: {"service": "<name>", "fix_type": "<type>"} — Apply remediation
- verify_health: {"service": "<name>"} or {} — Verify resolution
- write_postmortem: {"content": "<detailed text>"} — Document the incident
- escalate: {} — Get a hint (costs points)

Output valid JSON only — no markdown fences, no commentary outside the JSON object.
"""


# ── Few-Shot Exemplar ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLE = """
EXAMPLE of a high-scoring (0.94) investigation on a similar incident:
Step 1: list_services → Found order-service DOWN, auth-service degraded
Step 2: check_alerts → Alert: order-service OOMKilled 3x in 15 minutes
Step 3: check_slo → order-service burning 4.2x error budget
Step 4: classify_severity → SEV1 (complete service outage)
Step 5: update_status_page → "Investigating order-service outage affecting all orders"
Step 6: read_logs(order-service) → OOM errors, heap at 98%
Step 7: check_metrics(order-service, memory) → Memory linearly increasing
Step 8: check_dependencies(order-service) → upstream: api-gateway, downstream: order-db
Step 9: check_deployments → Deploy D-003 at T-10min changed memory limits
Step 10: trace_request(order-service) → Span shows 4500ms before OOM
Step 11: check_dependencies(api-gateway) → Confirms cascade from order-service
Step 12: run_diagnostic(order-service, type=memory) → Confirmed memory leak
Step 13: apply_fix(order-service, fix_type=restart) → Fix applied
Step 14: update_status_page → "Fix applied, monitoring recovery"
Step 15: write_postmortem → "Root cause: OOM Kill due to memory leak in order-service heap. Deploy D-003 reduced memory limits causing heap exhaustion. Affected services: order-service (primary), api-gateway (cascade). Fixed by restarting order-service. Prevention: add memory alerts, review deployment memory configs."
Step 16: verify_health → All services healthy. EPISODE ENDS.
Key: Used 6+ evidence sources, classified severity early, updated status page twice, wrote detailed postmortem with keywords.
"""


# ── Logging helpers (mandatory format) ───────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ──────────────────────────────────────────────────────────────

def extract_json(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and extra text."""
    content = content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid JSON found in: {content[:200]}")


def get_temperature(step: int, max_steps: int) -> float:
    """Dynamic temperature scheduling: explore early, exploit late."""
    progress = step / max(max_steps, 1)
    if progress < 0.15:
        return 0.3   # Exploration: consider multiple hypotheses
    elif progress < 0.5:
        return 0.1   # Investigation: focused, precise actions
    else:
        return 0.05  # Exploitation: maximum determinism for fix/verify


def call_llm(client: OpenAI, messages: list, temperature: float = 0.1) -> str:
    """Call the LLM via OpenAI client and return content."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=4096,
            stream=False,
        )
    except Exception as e:
        if "max_tokens" in str(e) or "unsupported_parameter" in str(e):
            # Reasoning models need max_completion_tokens and don't support temperature/top_p
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_completion_tokens=4096,
                stream=False,
            )
        else:
            raise
    return completion.choices[0].message.content


# ── Investigation State Tracker (Reflexion-inspired) ─────────────────────────

class InvestigationState:
    """Tracks investigation progress for structured context management."""

    def __init__(self):
        self.services_checked: dict[str, str] = {}   # service → key finding
        self.evidence_sources: set[str] = set()       # "action:service" pairs
        self.hypothesis: str = ""
        self.root_cause_found: bool = False
        self.fix_applied: bool = False
        self.severity_classified: bool = False
        self.status_page_updated: bool = False
        self.postmortem_written: bool = False
        self.deps_checked: set[str] = set()           # services checked for deps
        self.slo_checked: bool = False
        self.deployments_checked: bool = False
        self.actions_taken: list[str] = []

    def record_action(self, action_type: str, service: str, finding: str = ""):
        self.actions_taken.append(f"{action_type}({service})")
        if service and action_type not in ("apply_fix", "verify_health", "write_postmortem",
                                            "escalate", "classify_severity", "update_status_page"):
            self.evidence_sources.add(f"{action_type}:{service}")
            if finding:
                self.services_checked[service] = finding
        if action_type == "check_dependencies":
            self.deps_checked.add(service)
        if action_type == "check_slo":
            self.slo_checked = True
        if action_type == "check_deployments":
            self.deployments_checked = True
        if action_type == "classify_severity":
            self.severity_classified = True
        if action_type == "update_status_page":
            self.status_page_updated = True
        if action_type == "apply_fix":
            self.fix_applied = True
        if action_type == "write_postmortem":
            self.postmortem_written = True

    def get_missing_bonuses(self) -> list[str]:
        """Return list of bonus actions not yet completed."""
        missing = []
        if not self.severity_classified:
            missing.append("classify_severity (early, after initial investigation)")
        if not self.status_page_updated:
            missing.append("update_status_page (BEFORE applying fix)")
        if not self.slo_checked:
            missing.append("check_slo (assess error budget burns)")
        if len(self.deps_checked) < 2:
            missing.append(f"check_dependencies on more services (have {len(self.deps_checked)}, need >=2)")
        if len(self.evidence_sources) < 4:
            missing.append(f"gather more evidence (have {len(self.evidence_sources)}, want >=4)")
        if not self.deployments_checked:
            missing.append("check_deployments (correlate with error timeline)")
        return missing

    def get_summary(self) -> str:
        """Generate concise investigation summary for context compression."""
        parts = [
            "INVESTIGATION STATE:",
            f"  Evidence sources: {len(self.evidence_sources)} ({', '.join(sorted(self.evidence_sources))})",
            f"  Hypothesis: {self.hypothesis or 'Not yet formed'}",
        ]
        if self.services_checked:
            parts.append("  Key findings:")
            for svc, finding in self.services_checked.items():
                parts.append(f"    {svc}: {finding[:100]}")
        milestones = []
        if self.severity_classified:
            milestones.append("severity_classified")
        if self.status_page_updated:
            milestones.append("status_page_updated")
        if self.slo_checked:
            milestones.append("slo_checked")
        if self.deployments_checked:
            milestones.append("deployments_checked")
        if self.root_cause_found:
            milestones.append("root_cause_found")
        if self.fix_applied:
            milestones.append("fix_applied")
        if self.postmortem_written:
            milestones.append("postmortem_written")
        parts.append(f"  Milestones: {', '.join(milestones) if milestones else 'none'}")
        missing = self.get_missing_bonuses()
        if missing:
            parts.append(f"  STILL NEEDED for max score: {'; '.join(missing)}")
        return "\n".join(parts)


# ── Observation Formatting ───────────────────────────────────────────────────

def get_task_prompt_addon(initial_obs: dict) -> str:
    """Generate task-specific prompt guidance based on initial alert."""
    alert = initial_obs.get("alert_summary", "").lower()
    statuses = initial_obs.get("service_statuses", {})
    degraded = [
        (name, s) for name, s in statuses.items()
        if isinstance(s, dict) and s.get("status") in ("degraded", "down")
    ]

    addons = []

    # Pattern-based guidance from alert text
    if any(kw in alert for kw in ["oom", "memory", "heap", "restart"]):
        addons.append("PATTERN HINT: Alert mentions memory/OOM — likely a memory leak. "
                      "Check heap metrics and look for recent deploys that changed memory limits.")
    if any(kw in alert for kw in ["pool", "connection", "hikari"]):
        addons.append("PATTERN HINT: Connection pool issue — trace to DATABASE leaf services.")
    if any(kw in alert for kw in ["ssl", "tls", "cert", "handshake"]):
        addons.append("PATTERN HINT: TLS/certificate issue — run_diagnostic type='tls'.")
    if any(kw in alert for kw in ["dns", "nxdomain", "resolution"]):
        addons.append("PATTERN HINT: DNS resolution failure — run_diagnostic type='dns'.")
    if any(kw in alert for kw in ["network", "partition", "unreachable", "timeout"]):
        addons.append("PATTERN HINT: Network partition — run_diagnostic type='iptables', check for deploy changes.")
    if any(kw in alert for kw in ["kafka", "lag", "consumer", "offset"]):
        addons.append("PATTERN HINT: Kafka consumer lag — check consumer groups and partition leadership.")
    if any(kw in alert for kw in ["redis", "failover", "readonly", "replica"]):
        addons.append("PATTERN HINT: Redis failover — check replication status and promote replica if needed.")
    if any(kw in alert for kw in ["disk", "space", "filesystem", "storage"]):
        addons.append("PATTERN HINT: Disk full — check disk usage and identify large files/logs consuming space.")
    if any(kw in alert for kw in ["rate limit", "429", "throttl"]):
        addons.append("PATTERN HINT: Rate limiting — check traffic patterns and rate limit configuration.")
    if any(kw in alert for kw in ["migration", "lock", "deadlock", "blocking"]):
        addons.append("PATTERN HINT: DB migration lock — check for long-running migrations blocking queries.")
    if any(kw in alert for kw in ["flap", "crashloop", "health check"]):
        addons.append("PATTERN HINT: Health check flapping — check health check config and readiness probes.")
    if any(kw in alert for kw in ["evict", "node pressure", "pod"]):
        addons.append("PATTERN HINT: Pod eviction — check node resource pressure and batch job scheduling.")
    if any(kw in alert for kw in ["secret", "rotation", "expired key", "401"]):
        addons.append("PATTERN HINT: Secret rotation failure — check credential expiry and rotation configs.")
    if any(kw in alert for kw in ["log storm", "logging", "flood"]):
        addons.append("PATTERN HINT: Log storm — check log levels and identify the source of excessive logging.")

    # Multi-service cascade hint
    if len(degraded) > 3:
        addons.append("CASCADING FAILURE DETECTED: Multiple services degraded. "
                      "Root cause is likely in a LEAF service (database). "
                      "Trace the dependency chain from gateway → leaf to find the source.")

    # Service-specific hints from statuses
    for name, s in degraded:
        if isinstance(s, dict):
            restarts = s.get("restarts_last_hour", 0)
            if restarts >= 3:
                addons.append(f"HIGH RESTARTS: {name} has {restarts} restarts — possible OOM or crash loop.")

    return "\n".join(addons)


def format_observation(obs: dict, step_num: int, reward: float,
                       is_initial: bool = False, inv_state: "InvestigationState" = None) -> str:
    """Format observation to highlight key signals for the LLM."""
    parts = []
    if is_initial:
        parts.append(f"INCIDENT ALERT: {obs.get('alert_summary', 'Unknown incident')}")
        parts.append(f"\nAvailable actions: {obs.get('available_actions', [])}")
    else:
        result = obs.get("last_action_result", "")
        parts.append(f"Result: {result}")
        parts.append(f"Phase: {obs.get('incident_phase', '?')} | Step reward: {reward}")

    statuses = obs.get("service_statuses", {})
    degraded = []
    healthy = []
    items = statuses.values() if isinstance(statuses, dict) else statuses
    for s in items:
        if isinstance(s, str):
            healthy.append(s)
            continue
        name = s.get("name", "?")
        status = s.get("status", "")
        err = s.get("error_rate", 0)
        lat = s.get("latency_p99_ms", 0)
        restarts = s.get("restarts_last_hour", 0)
        pool = s.get("connection_pool_usage", 0)
        if status in ("degraded", "down"):
            extras = []
            if restarts > 0:
                extras.append(f"restarts={restarts}")
            if pool and pool > 0.8:
                extras.append(f"pool={pool} CRITICAL")
            extra_str = f", {', '.join(extras)}" if extras else ""
            degraded.append(f"  {name}: {status} (err={err}, lat={lat}ms{extra_str})")
        else:
            healthy.append(name)

    if degraded:
        parts.append("\nDEGRADED/DOWN services (investigate these):")
        parts.extend(degraded)
    if healthy:
        parts.append(f"Healthy: {', '.join(healthy)}")

    # Pattern detection hints from action results
    result_text = obs.get("last_action_result", "")
    if any(kw in result_text.lower() for kw in ["ssl", "tls", "certificate", "cert"]):
        parts.append("\nTLS/cert keywords detected -> consider run_diagnostic type='tls'")
    if any(kw in result_text.lower() for kw in ["dns", "nxdomain", "getaddrinfo", "stale ip"]):
        parts.append("\nDNS keywords detected -> consider run_diagnostic type='dns'")
    if any(kw in result_text.lower() for kw in ["connection pool", "hikaripool", "pool exhausted"]):
        parts.append("\nConnection pool issue -> trace to the DATABASE service (leaf node)")
    if any(kw in result_text.lower() for kw in ["network unreachable", "connection timed out", "iptables", "split-brain"]):
        parts.append("\nNetwork partition keywords -> consider run_diagnostic type='iptables'")
    if any(kw in result_text.lower() for kw in ["oom", "memory exceeded", "heap space"]):
        parts.append("\nOOM/memory keywords detected -> likely memory leak, check heap metrics")
    if any(kw in result_text.lower() for kw in ["kafka", "consumer lag", "offset behind"]):
        parts.append("\nKafka lag detected -> check consumer groups and partition status")
    if any(kw in result_text.lower() for kw in ["disk", "no space", "filesystem full"]):
        parts.append("\nDisk space issue detected -> check disk usage and cleanup")

    # Reflexion: remind agent of missing bonus actions
    if inv_state and not is_initial:
        missing = inv_state.get_missing_bonuses()
        if missing and not inv_state.fix_applied:
            parts.append(f"\n📋 BONUS ACTIONS STILL NEEDED: {'; '.join(missing[:3])}")
        elif inv_state.fix_applied and not inv_state.postmortem_written:
            parts.append("\n⚠️ FIX APPLIED — Now write_postmortem with detailed technical content, THEN verify_health.")
        elif inv_state.postmortem_written and not is_initial:
            parts.append("\n✅ Postmortem written — Now verify_health to complete the episode.")

    # Periodic strategic reminders
    if step_num > 0 and step_num % 6 == 0:
        parts.append("\nReminder: Have you checked deployments? Correlated timing? Run diagnostics?")
    if step_num > 0 and step_num % 10 == 0:
        parts.append("Reminder: After fixing root cause, ALWAYS write_postmortem THEN verify_health.")

    return "\n".join(parts)


def build_postmortem_prompt(inv_state: "InvestigationState") -> str:
    """Generate a postmortem guidance prompt with accumulated investigation data."""
    findings = []
    for svc, finding in inv_state.services_checked.items():
        findings.append(f"- {svc}: {finding[:150]}")

    return (
        "\nPOSTMORTEM GUIDANCE — Write a DETAILED postmortem including ALL these sections:\n"
        "1. INCIDENT SUMMARY: What happened, when, scope of impact\n"
        "2. ROOT CAUSE: Specific technical root cause (mention exact service names, error types, "
        "technical terms like OOM, memory leak, connection pool, DNS, TLS, etc.)\n"
        "3. AFFECTED SERVICES: List ALL services that were impacted\n"
        "4. TIMELINE: Key events with approximate timestamps\n"
        "5. REMEDIATION: Exact fix applied and why it resolved the issue\n"
        "6. PREVENTION: What changes would prevent recurrence\n"
        "\nYour investigation findings:\n" +
        "\n".join(findings) +
        "\n\nCRITICAL: Include as many specific technical terms as possible from your investigation "
        "(service names, error messages, metric values, deploy IDs). Higher detail = higher score.\n"
        'Output: {"action_type": "write_postmortem", "parameters": {"content": "<your detailed postmortem>"}, '
        '"reasoning": "Documenting incident before verification"}'
    )


# ── Main episode runner ──────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task episode with enhanced investigation strategy. Returns the grader score."""
    max_steps = MAX_STEPS_MAP[task_id]

    # Reset environment
    reset_resp = httpx.post(
        f"{SPACE_URL}/reset",
        json={"task_id": task_id, "seed": SEED},
        timeout=30,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()["observation"]

    # Initialize investigation state tracker
    inv_state = InvestigationState()

    # Build initial message with task-adaptive guidance and few-shot
    initial_msg = format_observation(obs, 0, 0.0, is_initial=True, inv_state=inv_state)
    task_addon = get_task_prompt_addon(obs)
    if task_addon:
        initial_msg += f"\n\n{task_addon}"
    initial_msg += f"\n\n{FEW_SHOT_EXAMPLE}"
    initial_msg += "\n\nFollow the investigation methodology. Start with Step 1 (ORIENT)."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_msg},
    ]

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step_num in range(1, max_steps + 1):
            # Dynamic temperature
            temperature = get_temperature(step_num, max_steps)

            # Get LLM action
            raw_action = None
            content = ""
            for attempt in range(3):
                try:
                    content = call_llm(client, messages, temperature=temperature)
                    if not content or not content.strip():
                        continue
                    raw_action = extract_json(content)
                    if "reasoning" not in raw_action:
                        raw_action["reasoning"] = "No explicit reasoning provided."
                    break
                except Exception as e:
                    print(f"[DEBUG] step={step_num} attempt={attempt+1} error: {e}", file=sys.stderr)
                    if attempt == 2:
                        break

            if raw_action is None:
                raw_action = {
                    "action_type": "escalate",
                    "parameters": {},
                    "reasoning": "Fallback after parse error",
                }

            action_type = raw_action.get("action_type", "?")
            service = raw_action.get("parameters", {}).get("service", "")

            # Track in investigation state
            inv_state.record_action(action_type, service)

            action_str = f"{action_type}({json.dumps(raw_action.get('parameters', {}))})"

            # Step environment
            try:
                step_resp = httpx.post(
                    f"{SPACE_URL}/step",
                    json={"action": raw_action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] HTTP step error: {e}", file=sys.stderr)
                log_step(step=step_num, action=action_str, reward=0.0, done=False, error=str(e))
                rewards.append(0.0)
                steps_taken = step_num
                break

            obs = step_data["observation"]
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            error = obs.get("last_action_error") if isinstance(obs, dict) else None

            rewards.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            # Extract key findings for investigation state
            result_text = obs.get("last_action_result", "") if isinstance(obs, dict) else ""
            if service and result_text:
                # Extract first meaningful line as finding summary
                first_line = result_text.split("\n")[0][:120] if result_text else ""
                if first_line and service not in inv_state.services_checked:
                    inv_state.services_checked[service] = first_line

            # Check if root cause indicators are present
            if any(kw in result_text.lower() for kw in ["root cause", "confirmed", "identified", "leak detected",
                                                          "pool exhausted confirmed", "cert expired confirmed"]):
                inv_state.root_cause_found = True

            # Update message history
            messages.append({"role": "assistant", "content": content})

            # Build observation message with reflexion guidance
            obs_msg = format_observation(obs, step_num, reward, inv_state=inv_state)

            # Inject postmortem template guidance when fix is applied but postmortem not written
            if inv_state.fix_applied and not inv_state.postmortem_written:
                obs_msg += build_postmortem_prompt(inv_state)

            messages.append({"role": "user", "content": obs_msg})

            # Context window management with priority-based retention
            if len(messages) > 20:
                # Use investigation state for rich summary instead of raw action list
                summary = inv_state.get_summary()

                # Add recent action log
                recent_actions = inv_state.actions_taken[-8:]
                summary += f"\n  Recent actions: {', '.join(recent_actions)}"
                summary += "\n\nContinue investigation. Remember: write_postmortem BEFORE verify_health."

                # Keep system prompt + summary + last 14 messages
                messages = [messages[0], {"role": "user", "content": summary}] + messages[-14:]

            if done:
                break

        # Get final grader score
        grader_resp = httpx.get(f"{SPACE_URL}/grader", timeout=30)
        grader_resp.raise_for_status()
        score = grader_resp.json().get("episode_score", 0.0)

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} exception: {exc}", file=sys.stderr)
        score = sum(rewards) / max(1, len(rewards)) if rewards else 0.001
        score = min(max(score, 0.001), 0.999)
        log_end(
            success=False,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )
        return score

    score = min(max(score, 0.001), 0.999)
    success = score > 0.001

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
        timeout=120.0,
    )

    results = {}
    for task_id in TASKS:
        score = run_task(client, task_id)
        results[task_id] = round(score, 4)

    print("\n=== SREBench Final Scores ===", file=sys.stderr)
    for task_id, score in results.items():
        print(f"  {task_id:<30} {score:.4f}", file=sys.stderr)
    avg = sum(results.values()) / len(results)
    print(f"  {'Average':<30} {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
