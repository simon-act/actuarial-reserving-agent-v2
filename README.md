# 🧠 Intelligent Reserving Agent

**AI-Powered Actuarial Reserving with Intelligent Agents**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intelligent-reserving.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Philosophy

Traditional reserving systems use **hardcoded thresholds** and **if-then rules**. This system is different:

| Traditional Approach | Intelligent Agent Approach |
|---------------------|---------------------------|
| `if volatility > 0.15: smooth` | LLM sees data and *reasons* about whether to smooth |
| `pick method with min(MSE)` | LLM considers context, diagnostics, pattern characteristics |
| No explanation | Full reasoning for every decision |
| Fixed rules | Adapts to data characteristics |

**Core Principles:**
- 🚫 **NO hardcoded thresholds** - The LLM decides what's "significant"
- 🚫 **NO if-then rules** - All decisions go through intelligent reasoning
- ✅ **TRANSPARENT reasoning** - Every decision is explained
- ✅ **SELF-CRITIQUE** - Agents evaluate their own decisions

---

## 🌐 Live Demo

**Try the app online:** [https://intelligent-reserving.streamlit.app](https://intelligent-reserving.streamlit.app)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/simon-act/intelligent-reserving-agent.git
cd intelligent-reserving-agent

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

---

## 🧠 How It Works

### The Thinking Process

Every intelligent agent follows: **Analyze → Decide → Critique**

```
┌─────────────────────────────────────────────────────────────┐
│                        RAW DATA                              │
│  (factors, metrics, patterns, triangle characteristics)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      1. ANALYZE                              │
│  • What patterns exist?                                      │
│  • What anomalies are concerning?                           │
│  • What's the overall data quality?                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      2. DECIDE                               │
│  • Which option fits the evidence?                          │
│  • What are the risks?                                      │
│  • What would change this decision?                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      3. CRITIQUE                             │
│  • What are the weaknesses in this reasoning?               │
│  • What alternative interpretations exist?                  │
│  • What additional data would help?                         │
└─────────────────────────────────────────────────────────────┘
```

### Intelligent Pattern Analysis

When the agent sees development factors like:
```
Period 1: 2.1543
Period 2: 1.4521
Period 3: 1.2876  ← decrease
Period 4: 1.3102  ← increase! (non-monotonic)
Period 5: 1.1254
```

**Traditional system:** `if non_monotonic_count > 2: apply_smoothing()`

**Intelligent agent:**
1. Sees the raw numbers
2. Analyzes: "Period 4 shows a 1.8% increase from Period 3. This is relatively small compared to the overall downward trend..."
3. Considers context: triangle size, volatility, diagnostic results
4. Decides: "Apply 70% exponential decay smoothing because..."
5. Critiques: "A weakness of this decision is..."

---

## 📁 Project Structure

```
intelligent-reserving-agent/
├── app.py                          # Main Streamlit app
├── pages/
│   ├── 1_Reported_Claims.py       # Triangles and method selection
│   └── 2_Summary.py               # Results and scenarios
│
├── src/
│   ├── agents/
│   │   ├── intelligent_base.py         # 🧠 Base framework
│   │   ├── intelligent_selection.py    # 🧠 Method selection agent
│   │   ├── reserving.py                # Execution agent
│   │   ├── llm_utils.py                # LLM client
│   │   └── schemas.py                  # Data schemas
│   │
│   ├── pattern_analysis/
│   │   ├── pattern_analyzer.py         # 🧠 Intelligent pattern analysis
│   │   └── curve_fitting.py            # Smoothing tools (exp, power, weibull)
│   │
│   ├── chain_ladder.py                 # Core Chain-Ladder
│   ├── stochastic_reserving/           # Mack & Bootstrap
│   ├── alternative_methods/            # Cape Cod
│   ├── tail_fitting/                   # Tail factor estimation
│   ├── model_selection/                # Factor estimators & CV
│   ├── diagnostics/                    # Model diagnostics
│   └── scenario_analysis/              # Stress testing
│
└── data/
    └── sample_triangle.csv
```

---

## 🔧 Reserving Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Chain-Ladder | Classic development factor method | Standard reserving |
| Mack Model | Distribution-free stochastic CL | Uncertainty quantification |
| Bootstrap | Simulation-based distributions | Full reserve distribution |
| Cape Cod | Implicit ELR from data | Homogeneous portfolios |

### Factor Estimators

- Volume Weighted
- Simple Average
- Medial (excludes extremes)
- Geometric Mean
- Regression-based
- Exponential Weighted

### Pattern Smoothing Methods

- Exponential Decay
- Inverse Power
- Weibull
- Monotonic Spline
- Linear Decay

---

## 🛠️ The Intelligent Agent Framework

All intelligent agents inherit from `IntelligentAgent`:

```python
class MyAgent(IntelligentAgent):
    def _get_system_prompt(self) -> str:
        """Define the agent's expertise and perspective."""
        return "You are an expert in..."

    def _format_data_for_analysis(self, data: Dict) -> str:
        """Format raw data for LLM to see."""
        return f"FACTORS:\n{data['factors']}\n..."
```

The base class provides:
- `analyze(data)` → LLM identifies patterns and anomalies
- `decide(analysis, options)` → LLM chooses with full reasoning
- `critique(decision)` → LLM critiques its own decision
- `think(data, options)` → Complete cycle: analyze → decide → critique

---

## 📚 References

1. Mack, T. (1993). *Distribution-free calculation of the standard error of chain ladder reserve estimates*. ASTIN Bulletin.

2. England, P. & Verrall, R. (2002). *Stochastic claims reserving in general insurance*. British Actuarial Journal.

3. Sherman, R. (1984). *Extrapolating, smoothing and interpolating development factors*. Proceedings of the CAS.

---

## 📄 License

MIT License

---

## 👤 Author

Simone Pirovano

---

*Built with Python for intelligent actuarial analysis*
