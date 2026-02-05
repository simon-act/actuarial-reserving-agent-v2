# Dashboard Refactor Prompt

## Contesto
App Streamlit per actuarial reserving (`src/dashboard/app.py`). Attualmente ha: triangle display, key metrics, diagnostics (prima dell'automatic selection), model selection (CL patterns + cross-model), ML section separata (anomaly detection, GBM, scenarios).

---

## Richiesta: Riorganizzare il flusso della dashboard in modo sequenziale e aggiungere funzionalità

### 1. Nuovo Flusso Sequenziale

Riordinare le sezioni del dashboard in questo ordine:

1. **📁 Data Selection** (sidebar - già presente)
2. **📐 Loss Development Triangles** - Heatmap, tabella, Age-to-Age factors (già presente, spostare all'inizio)
3. **🔍 Anomaly Detection** - Spostare dalla sezione ML qui. Mostrare subito dopo il triangolo per identificare eventuali outlier prima della modellizzazione
4. **🎯 Automatic Model Selection** - Mantenere i due tab esistenti (CL Pattern Selection + Cross-Model Comparison), **ma aggiungere GBM come modello comparabile** nel tab "Cross-Model Comparison"
5. **📊 Model Diagnostics & Residuals** *(NUOVA SEZIONE - vedi sotto)*
6. **📋 Summary Table** *(NUOVA SEZIONE - vedi sotto)*

---

### 2. Nuova Sezione: Model Diagnostics dopo la Selezione Manuale

Dopo l'Automatic Model Selection, aggiungere una nuova sezione dove l'utente può:

#### a) Selezionare manualmente il modello da analizzare tramite dropdown:
- Chain-Ladder (con varie combinazioni metodo/window)
- Mack Model
- Bornhuetter-Ferguson (se premium disponibile)
- Cape Cod (se premium disponibile)
- **GBM**

#### b) Visualizzare i Model Diagnostics per il modello selezionato:
- **Adequacy Score Gauge** (già implementato in `DiagnosticTests`)
- **Residual Histogram** con curva normale sovrapposta
- **Q-Q Plot** dei residui standardizzati
- **Residuals vs Fitted Values** (scatter plot)
- **Residuals vs Development Period** (per verificare pattern temporali)
- **Residuals vs Accident Year** (per verificare trend)

#### c) Triangolo dei Residui:
- Heatmap del triangolo con i residui standardizzati
- Colorscale divergente (rosso = residui positivi alti, blu = residui negativi alti, bianco = ~0)
- Tabella numerica dei residui con evidenziazione celle anomale (|z| > 2)

#### d) Test Statistici:
- Tabella con risultati dei test (già in `DiagnosticTests`):
  - Durbin-Watson (autocorrelazione)
  - Shapiro-Wilk (normalità)
  - Breusch-Pagan (omoschedasticità)
  - Calendar Year Effect test
- Indicare Pass/Fail per ogni test con icone ✅/❌

---

### 3. GBM nel Cross-Model Comparison

Nel tab "Cross-Model Comparison", aggiungere GBM alla lista dei modelli validabili:
- Usare la classe `GBMFactorPredictor` già esistente
- Calcolare RMSE/MAE sulla holdout diagonal come per gli altri modelli
- Includere nel ranking e nel grafico a barre

---

### 4. Nuova Sezione: Tabella Dettaglio Ultimates & Reported

Dopo i diagnostics, aggiungere una tabella riassuntiva per il modello selezionato:

| Accident Year | Reported (Latest) | Ultimate | IBNR Reserve | % Reported | Development Age |
|--------------|-------------------|----------|--------------|------------|-----------------|
| 2010         | 1,234,567         | 1,245,678| 11,111       | 99.1%      | 180 months      |
| 2011         | 1,345,678         | 1,400,000| 54,322       | 96.1%      | 168 months      |
| ...          | ...               | ...      | ...          | ...        | ...             |
| **TOTAL**    | **XX,XXX,XXX**    | **XX,XXX,XXX** | **X,XXX,XXX** | **XX.X%** | - |

Opzioni:
- Toggle per mostrare/nascondere anni completamente sviluppati
- Export CSV della tabella

---

### 5. Dettagli Implementativi

- Usare `st.session_state` per memorizzare il modello selezionato e i suoi risultati
- I diagnostics devono aggiornarsi dinamicamente quando l'utente cambia modello nel dropdown
- Per GBM, i residui vanno calcolati come differenza tra valori osservati e predetti dal modello
- Mantenere la sezione "Economic Scenarios" alla fine (opzionale/avanzata)

---

### 6. Struttura Finale Desiderata

```
1. [Sidebar] Data Selection
2. [Main] Key Metrics (quick summary)
3. [Main] Loss Development Triangles (tabs: heatmap, table, ATA factors)
4. [Main] Anomaly Detection (heatmap + lista anomalie)
5. [Main] Automatic Model Selection
   ├── Tab 1: CL Pattern Selection (metodo × window)
   └── Tab 2: Cross-Model Comparison (CL, Mack, BF, Cape Cod, GBM)
6. [Main] Model Diagnostics (NUOVO)
   ├── Dropdown selezione modello manuale
   ├── Diagnostic plots (residuals histogram, Q-Q, vs fitted, vs period, vs year)
   ├── Residuals Triangle (heatmap + table)
   └── Statistical Tests summary
7. [Main] Ultimates & Reported Detail Table (NUOVO)
8. [Main] Economic Scenarios (opzionale, in fondo)
```

---

## Classi/Moduli Esistenti da Utilizzare

- `src/diagnostics/diagnostic_tests.py` → `DiagnosticTests`
- `src/diagnostics/residual_analysis.py` → `ResidualAnalyzer`
- `src/ml_models/anomaly_detector.py` → `TriangleAnomalyDetector`
- `src/ml_models/gradient_boosting_factors.py` → `GBMFactorPredictor`
- `src/model_selection/model_selector.py` → `ModelSelector`
- `src/chain_ladder.py` → `ChainLadder`
- `src/stochastic_reserving/mack_model.py` → `MackChainLadder`
