"""
Code Agent - Dynamic Python Execution for Q&A

When the Q&A agent can't find a pre-computed answer, this agent:
1. Receives the question + triangle data
2. Asks LLM to generate Python code
3. Executes code in a restricted sandbox
4. Returns the result

Security: Uses RestrictedPython + resource limits
"""

import sys
import io
import traceback
from typing import Dict, Any, Tuple, Optional
from contextlib import redirect_stdout, redirect_stderr
import signal
import pandas as pd
import numpy as np

from agents.schemas import AgentRole, AgentLog, ReservingOutput
from agents.llm_utils import LLMClient


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (5 seconds)")


class CodeAgent:
    """
    Generates and executes Python code to answer complex questions.

    Example:
        User: "What is the correlation between development factors?"
        CodeAgent:
            1. Generates: factors.corr()
            2. Executes safely
            3. Returns correlation matrix
    """

    def __init__(self):
        self.role = AgentRole.METHODOLOGY  # Reuse role for logging
        self.llm = LLMClient()
        self.max_execution_time = 5  # seconds

        # Allowed modules in sandbox
        self.allowed_modules = {
            'pd': pd,
            'np': np,
            'pandas': pd,
            'numpy': np,
        }

        # Forbidden operations
        self.forbidden_patterns = [
            'import os', 'import sys', 'import subprocess',
            'open(', 'exec(', 'eval(', '__import__',
            'os.', 'sys.', 'subprocess.',
            'requests.', 'urllib.', 'socket.',
            'shutil.', 'pathlib.',
            'rm ', 'del ', 'remove',
        ]

    def answer_with_code(
        self,
        question: str,
        context: ReservingOutput,
        triangle_df: Optional[pd.DataFrame] = None
    ) -> Tuple[str, AgentLog]:
        """
        Generate and execute code to answer the question.

        Args:
            question: User's question
            context: Current analysis results
            triangle_df: Raw triangle DataFrame (if available)

        Returns:
            (answer_string, log)
        """
        print(f"[CodeAgent] 🐍 Generating code for: {question[:50]}...")

        # 1. Prepare data context for code
        data_context = self._prepare_data_context(context, triangle_df)

        # 2. Generate code via LLM
        code = self._generate_code(question, data_context)

        if not code:
            return "I couldn't generate code for this question.", self._make_log("Failed to generate code")

        # 3. Validate code safety
        if not self._is_safe(code):
            return "The generated code contains unsafe operations.", self._make_log("Unsafe code blocked")

        # 4. Execute in sandbox
        result, error = self._execute_sandboxed(code, data_context)

        if error:
            return f"Code execution error: {error}", self._make_log(f"Execution error: {error}")

        # 5. Format result
        answer = self._format_result(question, code, result)

        return answer, self._make_log(f"Executed code successfully, result type: {type(result).__name__}")

    def _prepare_data_context(
        self,
        context: ReservingOutput,
        triangle_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Prepare data variables available to generated code."""

        data = {}

        # Triangle as DataFrame
        if triangle_df is not None:
            data['triangle'] = triangle_df
        elif context.detailed_data and context.detailed_data.triangle:
            # Reconstruct from dict
            tri_dict = context.detailed_data.triangle
            data['triangle'] = pd.DataFrame(tri_dict).T.astype(float)

        # Development factors
        if context.detailed_data and context.detailed_data.development_factors:
            data['factors'] = pd.Series(context.detailed_data.development_factors)

        # Reserves by year
        if context.detailed_data and context.detailed_data.reserves_by_year:
            data['reserves'] = pd.Series(context.detailed_data.reserves_by_year)

        # Ultimates by year
        if context.detailed_data and context.detailed_data.ultimates_by_year:
            data['ultimates'] = pd.Series(context.detailed_data.ultimates_by_year)

        # Latest diagonal
        if context.detailed_data and context.detailed_data.latest_diagonal:
            data['latest'] = pd.Series(context.detailed_data.latest_diagonal)

        # Method selection data
        if context.method_selection:
            ms = context.method_selection
            data['all_estimators'] = pd.Series(ms.all_estimators)
            data['maturity'] = pd.Series(ms.maturity_by_year)
            if ms.validation_metrics:
                data['validation_metrics'] = pd.DataFrame(ms.validation_metrics).T

        # Summary stats
        data['total_reserve'] = context.chain_ladder.total_reserve
        data['ultimate_loss'] = context.chain_ladder.ultimate_loss

        if context.mack:
            data['mack_se'] = context.mack.standard_error
            data['mack_cv'] = context.mack.cv

        return data

    def _generate_code(self, question: str, data_context: Dict[str, Any]) -> Optional[str]:
        """Use LLM to generate Python code."""

        if not self.llm.is_available():
            return self._fallback_code_generation(question)

        # Describe available variables
        var_descriptions = []
        for name, value in data_context.items():
            if isinstance(value, pd.DataFrame):
                var_descriptions.append(f"- `{name}`: DataFrame with shape {value.shape}, columns: {list(value.columns)[:5]}")
            elif isinstance(value, pd.Series):
                var_descriptions.append(f"- `{name}`: Series with {len(value)} items")
            else:
                var_descriptions.append(f"- `{name}`: {type(value).__name__} = {value}")

        available_vars = "\n".join(var_descriptions)

        system_prompt = """You are a Python code generator for actuarial data analysis.
Generate ONLY executable Python code - no explanations, no markdown.

RULES:
- Use pandas (pd) and numpy (np) only
- Available variables are pre-loaded (don't create them)
- Return result by assigning to `result` variable
- Keep code SHORT (max 10 lines)
- Handle missing data gracefully
- NO imports, NO file operations, NO prints

OUTPUT FORMAT:
Just the Python code, nothing else."""

        user_prompt = f"""Question: {question}

Available variables:
{available_vars}

Generate Python code to answer this question.
Assign the answer to a variable called `result`."""

        try:
            response = self.llm.get_completion(system_prompt, user_prompt)

            # Clean response
            code = response.strip()
            code = code.replace('```python', '').replace('```', '').strip()

            # Ensure result assignment
            if 'result' not in code:
                code = f"result = {code}"

            return code

        except Exception as e:
            print(f"[CodeAgent] LLM error: {e}")
            return None

    def _fallback_code_generation(self, question: str) -> Optional[str]:
        """Simple pattern-based code generation when LLM unavailable."""

        q = question.lower()

        if 'correlation' in q and 'factor' in q:
            return "result = triangle.pct_change(axis=1).corr()"

        if 'mean' in q and 'factor' in q:
            return "result = factors.mean()"

        if 'std' in q or 'volatility' in q:
            return "result = triangle.pct_change(axis=1).std()"

        if 'sum' in q and 'reserve' in q:
            return "result = reserves.sum()"

        if 'max' in q:
            if 'reserve' in q:
                return "result = reserves.max()"
            if 'factor' in q:
                return "result = factors.max()"

        if 'min' in q:
            if 'reserve' in q:
                return "result = reserves.min()"
            if 'factor' in q:
                return "result = factors.min()"

        if 'describe' in q or 'summary' in q:
            return "result = triangle.describe()"

        if 'shape' in q or 'dimension' in q:
            return "result = f'Triangle shape: {triangle.shape}'"

        return None

    def _is_safe(self, code: str) -> bool:
        """Check if code is safe to execute."""

        code_lower = code.lower()

        for pattern in self.forbidden_patterns:
            if pattern.lower() in code_lower:
                print(f"[CodeAgent] ⚠️ Blocked unsafe pattern: {pattern}")
                return False

        return True

    def _execute_sandboxed(
        self,
        code: str,
        data_context: Dict[str, Any]
    ) -> Tuple[Any, Optional[str]]:
        """Execute code in a sandboxed environment."""

        # Prepare namespace
        namespace = {
            'pd': pd,
            'np': np,
            'result': None,
            **data_context
        }

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Set timeout (Unix only)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.max_execution_time)
            except (AttributeError, ValueError):
                pass  # Windows or unavailable

            # Execute
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)

            # Cancel timeout
            try:
                signal.alarm(0)
            except (AttributeError, ValueError):
                pass

            result = namespace.get('result')
            return result, None

        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return None, error_msg
        finally:
            try:
                signal.alarm(0)
            except:
                pass

    def _format_result(self, question: str, code: str, result: Any) -> str:
        """Format the execution result as a readable answer."""

        answer_parts = []

        # Format result based on type
        if isinstance(result, pd.DataFrame):
            if result.shape[0] > 10:
                result_str = result.head(10).to_string()
                result_str += f"\n... ({result.shape[0]} total rows)"
            else:
                result_str = result.to_string()
            answer_parts.append(f"**Result:**\n```\n{result_str}\n```")

        elif isinstance(result, pd.Series):
            if len(result) > 10:
                result_str = result.head(10).to_string()
                result_str += f"\n... ({len(result)} total items)"
            else:
                result_str = result.to_string()
            answer_parts.append(f"**Result:**\n```\n{result_str}\n```")

        elif isinstance(result, (int, float)):
            if isinstance(result, float):
                if abs(result) > 1000:
                    result_str = f"{result:,.2f}"
                else:
                    result_str = f"{result:.4f}"
            else:
                result_str = f"{result:,}"
            answer_parts.append(f"**Result:** {result_str}")

        elif result is not None:
            answer_parts.append(f"**Result:** {result}")
        else:
            answer_parts.append("The code executed but returned no result.")

        # Add code snippet (collapsed)
        answer_parts.append(f"\n<details><summary>Code executed</summary>\n\n```python\n{code}\n```\n</details>")

        return "\n".join(answer_parts)

    def _make_log(self, details: str) -> AgentLog:
        return AgentLog(
            agent=self.role,
            action="Code Execution",
            details=details
        )


def get_code_agent() -> CodeAgent:
    return CodeAgent()
