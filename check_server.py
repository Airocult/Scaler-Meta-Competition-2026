#!/usr/bin/env python3
"""Quick script to check server and run baseline."""
import sys
import httpx

ports = [7860, 7861]
for port in ports:
    try:
        r = httpx.get(f"http://localhost:{port}/health", timeout=2)
        print(f"Port {port}: OK - {r.json()}")
    except Exception as e:
        print(f"Port {port}: FAILED - {e}")
