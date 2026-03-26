"""
Grader for Task 1: Memory Leak OOM Kill.
"""
from app.scenarios.task1_memory_leak import MemoryLeakScenario


class Task1Grader:
    """Deterministic grader for the memory leak scenario."""

    @staticmethod
    def grade(scenario: MemoryLeakScenario) -> float:
        return scenario.get_grader_score()
