"""
Fixed microservices dependency graph — 8 services forming a DAG.
The topology is identical across all tasks; only service statuses change.
"""
from collections import deque


class ServiceGraph:
    SERVICES: dict[str, dict] = {
        "api-gateway": {"downstream": ["auth-service", "order-service"]},
        "auth-service": {"downstream": ["user-db"]},
        "order-service": {"downstream": ["inventory-service", "payment-service"]},
        "inventory-service": {"downstream": ["inventory-db"]},
        "payment-service": {"downstream": ["payment-db"]},
        "user-db": {"downstream": []},
        "inventory-db": {"downstream": []},
        "payment-db": {"downstream": []},
    }

    def get_all_services(self) -> list[str]:
        return list(self.SERVICES.keys())

    def get_downstream(self, service: str) -> list[str]:
        return self.SERVICES.get(service, {}).get("downstream", [])

    def get_upstream_of(self, service: str) -> list[str]:
        """Returns all services that call the given service."""
        return [s for s, v in self.SERVICES.items() if service in v["downstream"]]

    def get_dependency_chain(self, service: str) -> list[str]:
        """BFS from service downward through the graph."""
        visited = []
        queue = deque([service])
        seen = {service}
        while queue:
            current = queue.popleft()
            visited.append(current)
            for dep in self.get_downstream(current):
                if dep not in seen:
                    seen.add(dep)
                    queue.append(dep)
        return visited

    def get_full_graph_text(self) -> str:
        """Returns a human-readable representation of the service graph."""
        lines = []
        for svc, info in self.SERVICES.items():
            deps = info["downstream"]
            if deps:
                lines.append(f"  {svc} → {', '.join(deps)}")
            else:
                lines.append(f"  {svc} (leaf — no downstream)")
        return "\n".join(lines)
