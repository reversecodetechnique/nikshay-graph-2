# NIKSHAY-GRAPH

## Overview
**NIKSHAY-GRAPH** is an AI-powered decision-support system designed for India's frontline health workers, the ASHA (Accredited Social Health Activist). Every year, India loses approximately 48,600 TB patients to Loss to Follow-Up (LTFU), which risks drug resistance, household transmission, and harder-to-treat re-entries. 

The core insight of Nikshay-Graph is that **TB dropout is a network problem, not a records problem**. By modeling patients as nodes in a typed, weighted knowledge graph, the system uses a Temporal Graph Network (TGN) to predict dropout before it happens and provides overloaded ASHAs with a prioritized, structured daily visit list.

Check out the dashboard here: https://nikshay-graph-dashboard-cxcvdfdpeddwgmdy.centralindia-01.azurewebsites.net/

---

## What It Does
A five-stage AI pipeline runs nightly via Azure Functions. Every morning, ASHAs receive a ranked priority visit list—translated into their preferred language and read aloud via neural Text-To-Speech (TTS)—explaining which patients need a visit and why.

Alerts escalate automatically through the clinical hierarchy: ASHA red flag → CHO → Medical Officer (MO) → District TB Officer (DTO).

---

## AI Pipeline Architecture

### Stage 0: Data Ingestion & Schema Normalisation
Merges three NTEP record formats (Static profile, monthly CHO observations, weekly ASHA updates) into a flat patient dictionary.

### Stage 1: NLP, Graph Construction & Silence Detection
* **Contact Extraction:** Azure AI Language NER extracts person names and relationship terms from ASHA free-text notes.
* **Graph Construction:** Maintains Patient, ASHA, CHO, PHC, WelfareScheme, Village, and Contact nodes in Azure Cosmos DB using an UPSERT pattern.
* **Silence Detection:** Applies phase-adaptive thresholds to detect consecutive days uncontactable.
* **Alert Routing:** Routes severe red flags directly to clinical levels.

### Stage 2: Temporal Graph Network (TGN)
Encodes events into a 20-dimensional message vector. Uses a GRUCell to update a per-patient memory vector and a GATConv to propagate signals over the graph. Produces a dropout probability trained to handle the 15:1 class imbalance.

### Stage 3: Composite Scoring Pipeline
Calculates the final risk score using the formula:
`composite = min(tgn_weight * tgn_score + bbn_weight * bbn_score + 0.05 * asha_load_score, 0.97)`
* Handover seamlessly shifts from a Bayesian Belief Network (BBN) driven by literature odds ratios on Day 1 to the TGN as evidence accumulates.
* Features velocity overrides for sudden deterioration and an equity floor adjusting for ASHA overload.

### Stage 4: Explainability
Generates template-based explanations keyed to the primary risk factor (e.g., adverse reactions, nutritional deterioration). Validated by Azure AI Foundry content safety to block diagnostic claims.

### Stage 5: Voice & Multilingual Briefings
Briefings are translated into 22 Indian scheduled languages via Azure AI Translator and rendered as MP3s via Azure AI Speech, playable directly in the ASHA's Streamlit dashboard.

---

## Azure Tech Stack
* **Azure Cosmos DB (Gremlin API):** Stores the patient knowledge graph and persists TGN memory vectors natively.
* **Azure Event Hubs:** Receives real-time dose, visit, and silence events without a polling loop.
* **Azure AI Language:** Performs NER on clinical/field notes mixing English and Indic languages.
* **Azure AI Translator:** Translates morning briefings natively.
* **Azure AI Speech:** Neural TTS for producing high-quality localized audio.
* **Azure Functions:** Serverless orchestration for the overnight batch job and real-time triggers.
* **Azure ML:** Production serving of trained TGN weights.

---

## System Evaluation
Nikshay-Graph was validated across 8 automated and manual evaluations using a 500-patient dataset reflecting Tondiarpet block population constants:
* **Score Sensitivity:** Correctly escalates scores for missed doses and compounds multiple risk factors accurately.
* **Priority List Face Validity:** Demonstrated a 0.811 score gap between top and bottom deciles based on clinical risk.
* **Stability:** The BBN is fully deterministic with zero variance across runs.
* **ASHA Caseload Validation:** Additive load floor produces mathematically exact equity adjustments (+0.035 delta at max load variance).
* **Treatment Phase Thresholds:** Successfully shifts alert sensitivity based on the patient's treatment phase.
* **Contact Propagation:** Graph edges carry real clinical weight; symptomatic household contacts raise patient scores significantly more than asymptomatic ones.
