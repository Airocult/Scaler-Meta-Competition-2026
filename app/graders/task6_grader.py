"""
Grader for Task 6: Split-Brain Network Partition.
"""
from app.scenarios.task6_network_partition import NetworkPartitionScenario


class Task6Grader:
    """Deterministic grader for the network partition scenario."""

    @staticmethod
    def grade(scenario: NetworkPartitionScenario) -> float:
        return scenario.get_grader_score()
