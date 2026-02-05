# Agents package

# Intelligent Agent Framework
from agents.intelligent_base import (
    IntelligentAgent,
    Analysis,
    Decision,
    Critique,
    AgentThought,
    ConfidenceLevel
)

# Agents
from agents.orchestrator import Orchestrator
from agents.methodology import MethodologyAgent
from agents.intelligent_selection import IntelligentSelectionAgent
from agents.reserving import ReservingExecutionAgent
from agents.validation import ValidationAgent
from agents.reporting import ReportingAgent
from agents.qa import QASpecialistAgent
from agents.code_agent import CodeAgent
from agents.llm_utils import LLMClient

# Backward compatibility
SelectionAgent = IntelligentSelectionAgent

__all__ = [
    # Framework
    "IntelligentAgent",
    "Analysis",
    "Decision",
    "Critique",
    "AgentThought",
    "ConfidenceLevel",
    # Agents
    "Orchestrator",
    "MethodologyAgent",
    "IntelligentSelectionAgent",
    "SelectionAgent",  # Alias
    "ReservingExecutionAgent",
    "ValidationAgent",
    "ReportingAgent",
    "QASpecialistAgent",
    "CodeAgent",
    "LLMClient",
]
