# 6. Documentation – Design, Workflow & Rationale

> **Scope**: This document outlines the assumptions, detailed workflow, file mapping, performance considerations, testing strategy, visualization, and justifications for the campaign data integration, completion, and estimation project. Implementation is in Python, modular, with robust error handling and unit testing (pytest).

---

## Project Structure

```
data_engineering_assignment_Adge/
├─ artifacts/run_120219464067020404
|  |-stage1_core.csv
|  |-stage1_events_action_values.csv
|  |-stage1_events_video_play_actions.csv
|  |-stage1_events_video_avg_time_watched_actions.csv
|  | -stage1_events_video_thruplay_watched_actions.csv
|  | -stage2_cartesian_summary.json
|  |- stage2_domain_report.csv
├─ data/
│  ├─ test_raw_data_age_gender (1).csv
│  ├─ test_raw_data_publisher_platform.csv
|- scripts
|  | __init__.py
|  | -check_cartesian_counts.py
|  | -export_stage1.py
|  |- run_ad_pipeline.py
├─ src/
│  ├─ __init__.py
│  ├─ cartesian_missing.py
│  ├─ data_loading.py
│  ├─ mappings.py
│  ├─ value_estimatio.py   
│  ├─ visualization.py
│  └─ tests/
│     ├─ test_cartesian_counts_integration.py
│     ├─ test_cartesian_missing.py
│     ├─ test_value_estimation.py   
│     └─ test_data_loading.py
├─ main.py
├─ README.md
├─ requirements.txt
```
---

## 1. Assumptions

- **Primary Key** – `ad_id` is the main entity; the same ad may appear with multiple categorical variants.  
- **JSON Parsing** – All JSON-like fields (`actions`, `action_values`, `video_play_actions`, etc.) can be malformed and are parsed using a defensive `safe_parse_actions` method.  
- **Null Handling** – Missing numeric metrics are imputed with `0`.  
- **Total Preservation** – The value estimation step preserves total sums for each metric per `ad_id`.  
- **Processing Order** – Data is aggregated into a per-`ad_id` dictionary **before** being normalized into structured DataFrames for modularity.  
- **NA as Wildcard** – When detecting missing categorical combinations, `NA` is treated as a wildcard to avoid false gaps.

---

## 2. Workflow & Implementation

### **Step 1 – Data Loading & Master Dictionary**
- **Goal:** Consolidate all data sources by `ad_id`, including nested JSON arrays.  
- **Approach:**  
  - Parse & normalize categorical and metric fields.  
  - Store in a hierarchical Python dictionary keyed by `ad_id`.  
  - Flatten into a master DataFrame and separate event tables.  
- **Files:** `data_loading.py`, `mappings.py`  
- **Tests:** `test_data_loading.py`

---

### **Step 2 – Cartesian Product Generation** *(Critical Fix Applied)*
- **Goal:** Identify all valid **categorical** combinations for each ad from:
  - `objective`, `age`, `gender`, `publisher_platform`
- **Fix:** Explicitly **exclude** `date_start` and `date_stop` from the Cartesian product to avoid unnecessary expansion.  
- **Output:** `stage2_cartesian_summary.json`, `stage2_domain_report.csv`  
- **Files:** `cartesian_missing.py`  
- **Tests:** `test_cartesian_missing.py`, `test_cartesian_counts_integration.py`

---

### **Step 3 – Missing Value Estimation**
- **Goal:** Impute missing metric rows **without altering overall totals**.  
- **Strategy:**  
  - Use marginal distributions & neighborhood-based estimation per `ad_id`.  
  - Normalize filled values so the total sum matches the original dataset.  
  - Merge cleanly into a single DataFrame (avoiding duplicated columns).  
- **Files:** `value_estimatio.py`  
- **Tests:** `test_value_estimation.py`

---

### **Step 4 – Visualization & Analytics**
- **Goal:** Provide analytical insights and visual validation.  
- **Features:**  
  - Side-by-side comparison (`Original` vs `Estimated`)  
  - Delta and % change tables per metric  
  - Distribution plots by categorical dimensions  
- **Files:** `visualization.py`  
- **Output:** `.png` plots, comparison CSVs

---

## 3. Testing Strategy

- **Framework:** `pytest`
- **Coverage:**  
  - Step 1 – Data merge & JSON parsing  
  - Step 2 – Cartesian domain completeness  
  - Step 3 – Estimation correctness & sum preservation  
  - Optional – Visualization data integrity  
- **Approach:** Synthetic datasets for deterministic checks

---

## 4. Performance & Scalability

- **Challenges:** Categorical explosion in Cartesian product.  
- **Mitigations:**  
  - Remove date fields from domain  
  - Process per `ad_id` to limit memory footprint  
  - Treat `NA` as wildcard in missing detection  
- **Future Scaling:**  
  - Distributed execution via Dask/Spark  
  - Parquet storage for efficient I/O  
  - Sparse encoding for event tables

---

## 5. Code Quality & Reliability

- Modular structure, type hints, and clear function boundaries  
- Defensive parsing for all external data  
- Logging at every pipeline stage  
- Final assertions on total preservation

---

## 6. How to Run

```bash
pip install -r requirements.txt
python main.py \
  --core-csv data/test_raw_data_age_gender.csv \
  --second-csv data/test_raw_data_publisher_platform.csv \
  --out-dir artifacts \
  --strategy neighbors \
  --neighbors-k 5 \
  --viz-metric impressions