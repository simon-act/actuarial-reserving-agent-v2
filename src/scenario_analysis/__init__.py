"""
Scenario Analysis and Stress Testing Module

This module provides comprehensive scenario analysis and stress testing
capabilities for actuarial reserving:

- Factor Stress Testing: Shocks to development factors
- Loss Ratio Stress Testing: Adverse loss ratio scenarios
- Tail Risk Analysis: Focus on immature years
- Combined Scenario Framework: Multiple stress factors
"""

from .stress_testing import StressTestFramework
from .scenario_generator import ScenarioGenerator
from .tail_risk import TailRiskAnalyzer

__all__ = [
    'StressTestFramework',
    'ScenarioGenerator',
    'TailRiskAnalyzer'
]
