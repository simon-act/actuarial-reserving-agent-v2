import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from agents.intelligent_base import IntelligentAgent, AgentRole, Analysis, Decision, ConfidenceLevel

# Mock Agent Implementation
class MockAgent(IntelligentAgent):
    def _get_system_prompt(self) -> str:
        return "You are a Mock Agent."
    
    def _format_data_for_analysis(self, data: dict) -> str:
        return str(data)

def test_agent_initialization():
    agent = MockAgent(role=AgentRole.EXECUTION)
    assert agent.role == AgentRole.EXECUTION
    assert agent.name == "Execution_Agent"
    assert agent.llm is not None

def test_fallback_analysis_when_llm_unavailable():
    agent = MockAgent(role=AgentRole.EXECUTION)
    # Simulate LLM unavailable
    agent.llm.is_available = MagicMock(return_value=False)
    
    data = {"foo": "bar"}
    analysis = agent.analyze(data)
    
    assert analysis.confidence == ConfidenceLevel.UNCERTAIN
    assert "LLM not available" in analysis.observations[0]
    assert "Fallback mode" in analysis.raw_reasoning

def test_fallback_decision_when_llm_unavailable():
    agent = MockAgent(role=AgentRole.EXECUTION)
    agent.llm.is_available = MagicMock(return_value=False)
    
    options = ["Option A", "Option B"]
    decision = agent.decide(None, options) # analysis is ignored in fallback
    
    assert decision.choice == "Option A"
    assert decision.confidence == ConfidenceLevel.UNCERTAIN
    assert "LLM not available" in decision.reasoning

@patch('agents.llm_utils.LLMClient.is_available')
@patch('agents.llm_utils.LLMClient.get_completion')
def test_successful_analysis_parsing(mock_get_completion, mock_is_available):
    agent = MockAgent(role=AgentRole.EXECUTION)
    mock_is_available.return_value = True
    
    # Mock LLM response with valid JSON
    mock_response = json.dumps({
        "observations": ["Trend is up"],
        "patterns": ["Linear growth"],
        "anomalies": [],
        "confidence": "high",
        "reasoning": "Data looks clean"
    })
    mock_get_completion.return_value = mock_response
    
    analysis = agent.analyze({"data": [1, 2, 3]})
    
    assert analysis.confidence == ConfidenceLevel.HIGH
    assert analysis.observations == ["Trend is up"]
    assert analysis.patterns == ["Linear growth"]

@patch('agents.llm_utils.LLMClient.is_available')
@patch('agents.llm_utils.LLMClient.get_completion')
def test_json_parsing_resilience(mock_get_completion, mock_is_available):
    """Test that it handles Markdown code blocks in JSON response"""
    agent = MockAgent(role=AgentRole.EXECUTION)
    mock_is_available.return_value = True
    
    # Mock LLM response wrapped in markdown code blocks (common LLM behavior)
    mock_response = """
    Here is the analysis:
    ```json
    {
        "observations": ["Wrapped in markdown"],
        "patterns": [],
        "anomalies": [],
        "confidence": "medium",
        "reasoning": "Parsing check"
    }
    ```
    """
    mock_get_completion.return_value = mock_response
    
    # We need to make sure _parse_json_response handles this, OR check if the agent logic needs improvement.
    # The current implementation in intelligent_base.py does basic { } search.
    # It finds first { and last }.
    
    analysis = agent.analyze({"data": []})
    
    assert analysis.observations == ["Wrapped in markdown"]
    assert analysis.confidence == ConfidenceLevel.MEDIUM
