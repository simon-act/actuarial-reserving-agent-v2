# 📋 Actuarial Reserving Project - Cheat Sheet

## Quick Reference Card

---

## 🚀 Come Runnare

```bash
# Dalla cartella del progetto
cd ~/ai-portfolio/reserving/src

# Workflow completo
python enhanced_workflow.py

# Dashboard (richiede: pip install streamlit plotly)
streamlit run dashboard/app.py
```

---

## 📁 Struttura Progetto

```
reserving/
├── src/
│   ├── chain_ladder.py           # Core Chain-Ladder
│   ├── enhanced_workflow.py      # 🎯 MAIN ENTRY POINT
│   ├── extract_triangle.py       # Data extraction
│   ├── visualizer.py             # Grafici matplotlib
│   │
│   ├── stochastic_reserving/     # Modelli stocastici
│   │   ├── mack_model.py         # Mack's Chain-Ladder (SE, CI)
│   │   └── bootstrap.py          # Bootstrap simulation
│   │
│   ├── alternative_methods/      # Metodi alternativi
│   │   ├── bornhuetter_ferguson.py
│   │   └── cape_cod.py
│   │
│   ├── tail_fitting/             # 🆕 Tail factor automatico
│   │   └── tail_estimator.py     # 7 metodi di fitting
│   │
│   ├── model_selection/          # Selezione modello
│   │   ├── model_selector.py     # Orchestratore
│   │   ├── factor_estimators.py  # 7 aggregation rules
│   │   ├── windowed_estimators.py
│   │   ├── kfold_validation.py   # K-fold CV
│   │   └── statistical_tests.py  # DM test, MCS
│   │
│   ├── diagnostics/              # Diagnostica
│   │   ├── residual_analysis.py
│   │   ├── volatility_analysis.py
│   │   └── diagnostic_tests.py
│   │
│   ├── scenario_analysis/        # Stress testing
│   │   ├── stress_testing.py
│   │   ├── scenario_generator.py
│   │   └── tail_risk.py
│   │
│   ├── utils/
│   │   └── stats_utils.py        # Funzioni statistiche native
│   │
│   └── dashboard/
│       └── app.py                # Streamlit dashboard
│
├── data/
│   ├── raw/                      # swiss_re_2023_triangles.xlsx
│   ├── processed/                # CSV triangoli
│   └── inputs/                   # expected_loss_ratios.csv
│
├── outputs/                      # Risultati
└── docs/                         # Documentazione
```

---

## 🔧 Moduli e Funzionalità

| Modulo | Classe/Funzione | Descrizione |
|--------|-----------------|-------------|
| `chain_ladder` | `ChainLadder` | Chain-Ladder classico |
| `mack_model` | `MackChainLadder` | Standard Error & Confidence Intervals |
| `bootstrap` | `BootstrapChainLadder` | Distribuzione riserve via simulazione |
| `bornhuetter_ferguson` | `BornhuetterFerguson` | BF con ELR stimato/manuale |
| `cape_cod` | `CapeCod` | Stanard-Bühlmann method |
| `tail_estimator` | `TailEstimator` | Tail fitting (7 curve) |
| `kfold_validation` | `KFoldTriangleValidator` | Cross-validation triangoli |
| `stress_testing` | `StressTestFramework` | Stress test scenari |
| `diagnostic_tests` | `DiagnosticTests` | Test adeguatezza modello |

---

## 📊 Metodi di Reserving

### 1. Chain-Ladder (Base)
```python
from chain_ladder import ChainLadder
cl = ChainLadder(triangle)
cl.run_full_analysis()
print(cl.summary())
```

### 2. Mack (con incertezza)
```python
from stochastic_reserving.mack_model import MackChainLadder
mack = MackChainLadder(triangle)
mack.fit()
ci = mack.get_confidence_intervals(alpha=0.95)
```

### 3. Bootstrap
```python
from stochastic_reserving.bootstrap import BootstrapChainLadder
boot = BootstrapChainLadder(triangle, n_simulations=10000)
boot.fit()
percentiles = boot.get_percentiles([75, 90, 95, 99])
```

### 4. Bornhuetter-Ferguson
```python
from alternative_methods.bornhuetter_ferguson import BornhuetterFerguson
bf = BornhuetterFerguson(triangle, earned_premium)
bf.fit()
print(bf.summary())
```

### 5. Cape Cod
```python
from alternative_methods.cape_cod import CapeCod
cc = CapeCod(triangle, earned_premium)
cc.fit()
print(f"Implicit ELR: {cc.cape_cod_elr:.2%}")
```

### 6. Tail Fitting
```python
from tail_fitting import TailEstimator
tail = TailEstimator(triangle)
tail.fit()
tail.print_summary()
print(f"Tail Factor: {tail.tail_factor:.4f}")
```

---

## 🎯 Model Selection

### Factor Estimators (7 metodi)
| Metodo | Descrizione |
|--------|-------------|
| Simple Average | Media semplice |
| Volume Weighted | Pesato per volume |
| Medial | Esclude min/max |
| Geometric | Media geometrica |
| Harmonic | Media armonica |
| Regression | Regressione OLS |
| Exponential | Pesi esponenziali decrescenti |

### Windowed Selection
```python
from model_selection.model_selector import ModelSelector

selector = ModelSelector.create_with_windowed_grid(
    triangle=triangle,
    min_window=3,
    max_window=10,
    recent_only=True
)
results = selector.run_windowed_analysis(selection_criterion='RMSE')
print(f"Best: {results['best_model']}")
```

---

## 📈 Diagnostica

```python
from diagnostics.diagnostic_tests import DiagnosticTests

diag = DiagnosticTests(triangle, development_factors)
diag.run_all_tests()
score = diag.get_model_adequacy_score()
print(f"Score: {score['adequacy_score']}% - {score['rating']}")
```

**Test eseguiti:**
- Calendar year effect
- Accident year effect
- Development period independence
- Proportionality assumption
- Variance structure

---

## ⚡ Stress Testing

```python
from scenario_analysis.stress_testing import StressTestFramework

stress = StressTestFramework(triangle, factors)
stress.run_standard_scenarios()
stress.run_regulatory_scenarios()  # Solvency II
print(stress.get_summary_table())
```

**Scenari standard:**
- Uniform shock (+5%, +10%, +20%)
- Early period shock
- Late period shock
- Tail shock

---

## 🎛️ Dashboard

```bash
# Avvia dashboard
cd src
streamlit run dashboard/app.py
```

**Features:**
- Triangolo interattivo (heatmap)
- Confronto metodi
- Bootstrap distribution
- Diagnostica visuale
- Stress test charts

---

## 📐 Tail Fitting Methods

| Metodo | Formula | Uso |
|--------|---------|-----|
| Exponential | `f(k) = 1 + a·e^(-bk)` | Decay rapido |
| Inverse Power | `f(k) = 1 + a/k^b` | Long-tail |
| Weibull | `f(k) = 1 + a·e^(-(k/b)^c)` | Flessibile |
| Sherman | `f(k) = 1 + a/(b+k)` | Hyperbolic |
| Bondy | `tail = f^(f/(f-1))` | Quick estimate |
| Linear Decay | `f(k) = 1 + (a + bk)` | Semplice |
| Log-Linear | `log(f-1) = a + bk` | Exponential fit |

---

## 🔑 Output Chiave

| File | Contenuto |
|------|-----------|
| `ultimate_and_reserves.csv` | Ultimate e riserve per AY |
| `selected_factors.csv` | Fattori selezionati |
| `method_comparison.csv` | Confronto metodi |
| `FINAL_REPORT.txt` | Report completo |
| `tail_fitting_comparison.csv` | Confronto curve tail |

---

## ⚠️ Note Importanti

1. **No scipy** - Tutte le funzioni statistiche sono native (`utils/stats_utils.py`)
2. **Dati** - Triangolo Swiss Re Property Reinsurance 2023
3. **Unità** - Valori in milioni ($m)
4. **Validazione** - Holdout validation (ultima diagonale)

---

## 📚 Quick Commands

```python
# Workflow completo
from enhanced_workflow import EnhancedReservingWorkflow
workflow = EnhancedReservingWorkflow(triangle, earned_premium)
results = workflow.run_complete_analysis()
workflow.save_all_results('outputs/enhanced/')

# Solo tail fitting
from tail_fitting import TailEstimator
tail = TailEstimator(triangle)
tail.fit()
full_factors = tail.get_full_cumulative_factors()
```
