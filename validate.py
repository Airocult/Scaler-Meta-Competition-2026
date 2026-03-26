#!/usr/bin/env python3
"""
Pre-submission Validator for SREBench
Checks all competition requirements before deployment.
Exit 0 = all checks pass, Exit 1 = failures found.
"""
import asyncio
import json
import os
import subprocess
import sys
import traceback

# Project root = directory containing this script
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
WARN = "\033[93m⚠️  WARN\033[0m"

results = []


def check(name, passed, detail=""):
    tag = PASS if passed else FAIL
    results.append({"name": name, "passed": passed, "detail": detail})
    print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
    return passed


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─── 1. File structure ───────────────────────────────────────

def check_files():
    section("1. Required Files")
    required = [
        "Dockerfile",
        "requirements.txt",
        "openenv.yaml",
        "README.md",
        "app/main.py",
        "app/environment.py",
        "app/models.py",
        "app/scenarios/task1_memory_leak.py",
        "app/scenarios/task2_db_cascade.py",
        "app/scenarios/task3_race_condition.py",
        "app/graders/task1_grader.py",
        "app/graders/task2_grader.py",
        "app/graders/task3_grader.py",
        "baseline/run_baseline.py",
        "client.py",
        "pyproject.toml",
    ]
    root = PROJECT_ROOT
    for f in required:
        path = os.path.join(root, f)
        check(f"File exists: {f}", os.path.isfile(path))


# ─── 2. openenv.yaml validation ─────────────────────────────

def check_openenv_yaml():
    section("2. openenv.yaml Validation")
    import yaml
    root = PROJECT_ROOT
    path = os.path.join(root, "openenv.yaml")
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
        check("openenv.yaml parseable", True)
    except Exception as e:
        check("openenv.yaml parseable", False, str(e))
        return

    check("Has 'name' field", "name" in config, config.get("name"))
    check("Has 'tasks' list", isinstance(config.get("tasks"), list))
    check("3+ tasks defined", len(config.get("tasks", [])) >= 3,
          f"{len(config.get('tasks', []))} tasks")
    check("Has 'action_space'", "action_space" in config)
    check("Has 'observation_space'", "observation_space" in config)

    for task in config.get("tasks", []):
        check(f"Task '{task['id']}' has max_steps",
              "max_steps" in task, f"max_steps={task.get('max_steps')}")


# ─── 3. Typed models (Pydantic + OpenEnv) ───────────────────

def check_models():
    section("3. Typed Models (Pydantic + OpenEnv)")
    try:
        from app.models import SREAction, SREObservation, SREState
        check("SREAction importable", True)
        check("SREObservation importable", True)
        check("SREState importable", True)

        # Verify they extend OpenEnv types
        from openenv.core.env_server.types import Action, Observation, State
        check("SREAction extends OpenEnv Action",
              issubclass(SREAction, Action))
        check("SREObservation extends OpenEnv Observation",
              issubclass(SREObservation, Observation))
        check("SREState extends OpenEnv State",
              issubclass(SREState, State))

        # Validate construction
        a = SREAction(action_type="list_services", reasoning="test")
        check("SREAction constructable", a.action_type == "list_services")

    except Exception as e:
        check("Models import", False, str(e))


# ─── 4. Environment class ───────────────────────────────────

def check_environment():
    section("4. Environment Class")
    try:
        from app.environment import SREBenchEnvironment
        from openenv.core.env_server.interfaces import Environment
        check("SREBenchEnvironment importable", True)
        check("Extends OpenEnv Environment",
              issubclass(SREBenchEnvironment, Environment))

        env = SREBenchEnvironment()
        check("Instantiable", True)

        # Test reset
        obs = env.reset(task_id="task1_memory_leak", seed=42)
        check("reset() returns observation", obs is not None)

        # Test step
        from app.models import SREAction
        action = SREAction(action_type="list_services", reasoning="testing")
        obs = env.step(action)
        check("step() returns observation", obs is not None)

        # Test state
        state = env.state
        check("state property works", state is not None)

    except Exception as e:
        check("Environment", False, traceback.format_exc())


# ─── 5. API endpoints (ASGI) ────────────────────────────────

async def check_endpoints():
    section("5. API Endpoints (OpenEnv + Custom)")
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # OpenEnv standard endpoints
        for path, method in [
            ("/health", "GET"),
            ("/schema", "GET"),
            ("/metadata", "GET"),
            ("/docs", "GET"),
            ("/openapi.json", "GET"),
        ]:
            r = await c.request(method, path)
            check(f"{method} {path} → {r.status_code}",
                  r.status_code == 200)

        # Reset
        r = await c.post("/reset", json={"task_id": "task1_memory_leak", "seed": 42})
        data = r.json()
        check("/reset returns {observation, reward, done}",
              all(k in data for k in ("observation", "reward", "done")))

        # Step
        r = await c.post("/step", json={
            "action": {"action_type": "list_services", "reasoning": "validate"}})
        data = r.json()
        check("/step returns {observation, reward, done}",
              all(k in data for k in ("observation", "reward", "done")))

        # State
        r = await c.get("/state")
        check("/state returns dict", r.status_code == 200)

        # Custom: /tasks
        r = await c.get("/tasks")
        tasks = r.json()
        check("/tasks returns 6 tasks",
              len(tasks.get("tasks", [])) == 6,
              f"{len(tasks.get('tasks', []))} tasks")
        check("/tasks has action_schema",
              "action_schema" in tasks)

        # Custom: /grader
        r = await c.get("/grader")
        score = r.json().get("episode_score")
        check("/grader returns score in [0,1]",
              isinstance(score, (int, float)) and 0 <= score <= 1,
              f"score={score}")

        # Custom: /baseline exists
        check("/baseline endpoint exists",
              any("/baseline" in str(route.path) for route in app.routes))


# ─── 6. Graders produce valid scores ────────────────────────

async def check_graders():
    section("6. Graders (6 tasks, scores in 0.0–1.0)")
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    tasks = ["task1_memory_leak", "task2_db_cascade", "task3_race_condition",
             "task4_dns_failure", "task5_cert_expiry", "task6_network_partition"]

    async with AsyncClient(transport=transport, base_url="http://test") as c:
        for task_id in tasks:
            # Reset
            await c.post("/reset", json={"task_id": task_id, "seed": 42})
            # Do a minimal episode
            await c.post("/step", json={
                "action": {"action_type": "list_services", "reasoning": "grader-test"}})
            # Get score
            r = await c.get("/grader")
            score = r.json().get("episode_score", -1)
            check(f"Grader {task_id}: score={score:.4f} in [0,1]",
                  0.0 <= score <= 1.0)


# ─── 7. Baseline script ─────────────────────────────────────

def check_baseline_script():
    section("7. Baseline Script")
    root = PROJECT_ROOT
    path = os.path.join(root, "baseline", "run_baseline.py")
    check("baseline/run_baseline.py exists", os.path.isfile(path))

    # Check it's parseable Python
    try:
        with open(path) as f:
            compile(f.read(), path, "exec")
        check("baseline/run_baseline.py is valid Python", True)
    except SyntaxError as e:
        check("baseline/run_baseline.py is valid Python", False, str(e))

    # Check it has --json flag
    with open(path) as f:
        src = f.read()
    check("Supports --json flag", "--json" in src)
    check("Uses OpenEnv format (action wrapper)",
          '"action"' in src or "'action'" in src)


# ─── 8. Dockerfile ───────────────────────────────────────────

def check_dockerfile():
    section("8. Dockerfile")
    root = PROJECT_ROOT
    path = os.path.join(root, "Dockerfile")
    with open(path) as f:
        content = f.read()

    check("Exposes port 7860", "7860" in content)
    check("Has CMD/ENTRYPOINT", "CMD" in content or "ENTRYPOINT" in content)
    check("Has HEALTHCHECK", "HEALTHCHECK" in content)
    check("Installs requirements.txt", "requirements.txt" in content)


# ─── 9. README HF Space metadata ────────────────────────────

def check_readme_metadata():
    section("9. README.md HF Space Metadata")
    root = PROJECT_ROOT
    path = os.path.join(root, "README.md")
    with open(path) as f:
        content = f.read()

    check("Has YAML frontmatter (---)", content.startswith("---"))
    check("sdk: docker", "sdk: docker" in content)
    check("Has title", "title:" in content)


# ─── MAIN ────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  SREBench Pre-Submission Validator")
    print("=" * 60)

    check_files()
    check_openenv_yaml()
    check_models()
    check_environment()
    await check_endpoints()
    await check_graders()
    check_baseline_script()
    check_dockerfile()
    check_readme_metadata()

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  VALIDATION: {passed}/{total} checks passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed:
        print("\n  Failed checks:")
        for r in results:
            if not r["passed"]:
                print(f"    ❌ {r['name']}: {r['detail']}")
        print()
        return 1
    else:
        print("\n  ✅ All checks passed — ready for submission!\n")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
