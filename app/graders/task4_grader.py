"""
Grader for Task 4: DNS Resolution Failure.
"""
from app.scenarios.task4_dns_failure import DNSFailureScenario


class Task4Grader:
    """Deterministic grader for the DNS failure scenario."""

    @staticmethod
    def grade(scenario: DNSFailureScenario) -> float:
        return scenario.get_grader_score()
