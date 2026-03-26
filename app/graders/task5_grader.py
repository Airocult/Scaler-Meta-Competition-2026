"""
Grader for Task 5: Certificate Expiry Chain.
"""
from app.scenarios.task5_cert_expiry import CertExpiryScenario


class Task5Grader:
    """Deterministic grader for the certificate expiry scenario."""

    @staticmethod
    def grade(scenario: CertExpiryScenario) -> float:
        return scenario.get_grader_score()
