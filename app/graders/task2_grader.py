"""
Grader for Task 2: Cascading DB Pool Exhaustion.
"""
from app.scenarios.task2_db_cascade import DBCascadeScenario


class Task2Grader:
    """Deterministic grader for the DB cascade scenario."""

    @staticmethod
    def grade(scenario: DBCascadeScenario) -> float:
        return scenario.get_grader_score()
