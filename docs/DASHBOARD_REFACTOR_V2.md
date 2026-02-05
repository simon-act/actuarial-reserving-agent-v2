# Dashboard Refactor v2 - Prompt

## Contesto
Dashboard Streamlit (`src/dashboard/app.py`) da ristrutturare in app multi-pagina con sidebar diagnostics.

---

## Struttura Multi-Pagina

| Pagina | Nome | Contenuto |
|--------|------|-----------|
| **Page 1** | **Reported Claims** | Triangles, Anomaly Detection, Model Selection, Diagnostics (sidebar) |
| **Page 2** | **Summary** | Ultimates & Reported Table, Economic Scenarios |

---

## Page 1: Reported Claims

### 1. Rimuovere Key Metrics
Eliminare completamente la sezione "📊 Key Metrics".

---

### 2. Sezione "📐 Loss Development Triangles"

**Tab structure:**

| Tab | Contenuto |
|-----|-----------|
| **Cumulative Triangle** | Heatmap del triangolo cumulato |
| **Incremental Triangle** | Heatmap del triangolo incrementale (`C[i,j] - C[i,j-1]`) |
| **Lag Factors + Anomalies** | Heatmap dei lag factors colorati con anomaly detection overlay |

Il terzo tab mostra i lag factors (age-to-age) con colorazione basata sullo z-score dell'anomaly detection.

---

### 3. Sezione "🔍 Anomaly Detection" - AUTO-RUN

**Comportamento:**
- Eseguita **automaticamente** al caricamento dati (no button)
- Cachare con `@st.cache_data` o `st.session_state`
- Button opzionale "🔄 Re-run" solo se l'utente vuole ri-triggerare

**Layout:**
```
┌─────────────────────────────────────────────────────┐
│  [Heatmap Anomaly Z-Score - GRANDE]                 │
├─────────────────────────────────────────────────────┤
│  Cells Analyzed: XX | Anomalies: XX | Rate: XX%     │
├─────────────────────────────────────────────────────┤
│  [Tabella anomalie rilevate - se presenti]          │
└─────────────────────────────────────────────────────┘
```

---

### 4. Sezione "🎯 Automatic Model Selection" - AUTO-RUN

#### 4a. Chain-Ladder Pattern Selection
- **Auto-run** con `min_window=3`, `max_window=10`
- Sliders per modificare parametri + button "🔄 Re-run"
- Risultati visibili immediatamente

#### 4b. Cross-Model Comparison
- Posizionato **SOTTO** CL Pattern Selection (stesso container, non tab)
- **Auto-run** con tutti i modelli: Chain-Ladder, Mack, **GBM**, BF, Cape Cod
- **GBM sempre incluso**

#### 4c. Model Ranking Table
Tabella semplice con i **migliori 7 modelli**:

| Rank | Model | RMSE | MAE |
|------|-------|------|-----|
| 🥇 1 | Volume Weighted [window=5] | 0.0234 | 1,234 |
| 🥈 2 | GBM | 0.0256 | 1,456 |
| 🥉 3 | Mack Model | 0.0278 | 1,567 |
| 4 | ... | ... | ... |

#### 4d. Settlement Speed Chart (NUOVO)
- Grafico **% of Ultimate Reported** per i modelli comparati
- Asse X: Development Period
- Asse Y: % Reported (0-100%)
- Una linea per ogni modello con colori distinti
- Legenda chiara

---

### 5. Model Diagnostics - SIDEBAR LATERALE

**Comportamento:**
- **Tendina/drawer laterale** che si apre con un clic (usando `st.sidebar` o `st.expander` laterale)
- Contiene un **dropdown per selezionare il modello** da analizzare
- Mostra diagnostics solo per il modello selezionato

**Contenuto della sidebar:**
```
┌─────────────────────────┐
│ 📊 Model Diagnostics    │
│ ─────────────────────── │
│ Select Model: [dropdown]│
│                         │
│ [Adequacy Gauge]        │
│ Rating: Good            │
│ Issues: ...             │
│                         │
│ [Residuals Histogram]   │
│                         │
│ ▶ Residual Plots        │
│ ▶ Residuals Triangle    │
│ ▶ Statistical Tests     │
└─────────────────────────┘
```

- Usare `st.expander` dentro la sidebar per i sotto-contenuti
- Il dropdown include: Chain-Ladder, Mack, GBM, BF, Cape Cod (se disponibili)

---

## Page 2: Summary

### 1. Ultimates & Reported Detail Table
- Tabella completa come attualmente implementata
- Toggle "Show all years"
- Export CSV

### 2. Economic Scenarios
- Pre-defined Scenarios
- Monte Carlo Simulation
- Non più in expander, sezione normale della pagina

---

## Struttura File

```
src/dashboard/
├── app.py                    # Entry point con page navigation
├── pages/
│   ├── 1_Reported_Claims.py  # Page 1
│   └── 2_Summary.py          # Page 2
└── components/
    └── diagnostics_sidebar.py # Componente sidebar diagnostics (opzionale)
```

**Oppure** struttura single-file con `st.navigation` / `st.page_link` (Streamlit 1.30+).

---

## Note Implementative

### Auto-run pattern
```python
# All'inizio della pagina, dopo aver caricato il triangolo:
if 'anomaly_results' not in st.session_state:
    detector = TriangleAnomalyDetector(anomaly_threshold=2.5)
    detector.fit(triangle, epochs=300)
    st.session_state['anomaly_results'] = detector

# Poi renderizza i risultati
detector = st.session_state['anomaly_results']
```

### Triangolo Incrementale
```python
incremental = triangle.copy()
for j in range(1, len(triangle.columns)):
    col_curr = triangle.columns[j]
    col_prev = triangle.columns[j-1]
    incremental[col_curr] = triangle[col_curr] - triangle[col_prev]
# Prima colonna rimane uguale (è il primo valore)
```

### Settlement Speed Chart
```python
def factors_to_pattern(factors):
    cum_factors = np.cumprod(factors.values[::-1])[::-1]
    return (1 / cum_factors) * 100

# Plot
fig = go.Figure()
for model_name, factors in model_factors.items():
    pattern = factors_to_pattern(factors)
    fig.add_trace(go.Scatter(x=periods, y=pattern, name=model_name))
fig.update_layout(
    title="Settlement Speed (% of Ultimate Reported)",
    xaxis_title="Development Period",
    yaxis_title="% Reported"
)
```

### Sidebar Diagnostics
```python
with st.sidebar:
    st.header("📊 Model Diagnostics")
    selected_model = st.selectbox("Select Model:", available_models)

    # Diagnostics content
    diag = DiagnosticTests(triangle, model_factors[selected_model])
    score = diag.get_model_adequacy_score()

    # Gauge, histogram, etc.
    with st.expander("Residual Plots"):
        # plots
    with st.expander("Statistical Tests"):
        # tests table
```

---

## Riepilogo Modifiche

| Elemento | Prima | Dopo |
|----------|-------|------|
| Key Metrics | Presente | **Rimosso** |
| Triangles | 3 tab (heatmap, table, ATA) | 3 tab (cumulative, incremental, lag+anomalies) |
| Anomaly Detection | Button trigger | **Auto-run** |
| Model Selection | 2 tab, button trigger | **Auto-run**, sotto unico container |
| Model Ranking | Top 3 cards | **Table top 7** |
| Settlement Chart | Non presente | **Nuovo** |
| Diagnostics | Sezione main | **Sidebar laterale** |
| Ultimates Table | Stessa pagina | **Page 2: Summary** |
| Economic Scenarios | Expander | **Page 2: Summary** (normale) |
