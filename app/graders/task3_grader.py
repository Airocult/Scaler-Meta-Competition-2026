"""
Grader for Task 3: Distributed Race Condition via Config Change.
"""
from app.scenarios.task3_race_condition import RaceConditionScenario


class Task3Grader:
    """Deterministic grader for the race condition scenario."""

    @staticmethod
    def grade(scenario: RaceConditionScenario) -> float:
        return scenario.get_grader_score()
