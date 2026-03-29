"""
stage2_tgn.py
=============
Stage 2: Temporal Risk Modelling — Temporal Graph Network (TGN)

Implements the full TGN architecture from the pipeline document:
  - Memory module: 64-dim GRU-updated vector per patient node
  - Graph Attention (GATConv): attention weights extracted for explainability
  - Silence events update memory (disengagement before dose gaps)
  - Patient memory persisted in Cosmos DB between inference sessions
  - Deployed as Azure ML managed endpoint in production

For hackathon prototype:
  - TGN runs locally using PyTorch Geometric
  - Azure ML endpoint call is stubbed with clear TODO markers
  - Memory vectors saved locally + pushed to Cosmos

Architecture:
  Input events (dose, visit, silence, contact symptom)
    → GRU memory update per node
    → GATConv aggregation over neighbours
    → 64-dim embedding
    → 2-layer MLP prediction head
    → dropout probability (0-1)

Usage:
    from stage2_tgn import TGNRiskModel, run_tgn_inference
    embeddings, attention_weights = run_tgn_inference(patients, graph)
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH GEOMETRIC TGN — full implementation
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch Geometric not installed. TGN will run in simulation mode.")
    print("Install with: pip install torch torch-geometric")

# Module-level cache — model loaded once per process, not on every dashboard update
_CACHED_TGN_MODEL = None
_CACHED_TGN_MTIME = None


if TORCH_AVAILABLE:

    class GRUMemoryModule(nn.Module):
        """
        Memory module from Temporal Graph Networks (Rossi et al. 2020).
        Each node carries a 64-dim memory vector updated by a GRU cell.
        The GRU learns which events should cause large memory updates
        and which should cause small ones — from training data, not rules.

        message_dim = 20 per §5.2 encode_event() specification.
        """
        def __init__(self, memory_dim: int = 64, message_dim: int = 20):
            super().__init__()
            self.memory_dim  = memory_dim
            self.message_dim = message_dim
            self.gru         = nn.GRUCell(message_dim, memory_dim)

        def forward(self, current_memory: torch.Tensor,
                    message: torch.Tensor) -> torch.Tensor:
            """
            Update memory given a new event message.
            current_memory: (N, 64)
            message:        (N, 20)
            returns:        (N, 64) updated memory
            """
            return self.gru(message, current_memory)


    def encode_event(event_type: str, features: dict, delta_t_days: float) -> torch.Tensor:
        """
        20-dim event message vector — v8 §5.2.
        Every dimension has a stated mechanism linking it to dropout risk.
        Clinical severity fields (SpO2, BP, RR, pulse, temperature) are NOT
        encoded here — they predict acute severity, not LTFU disengagement.

        Tier 1 (dims 0-5):  Engagement trajectory — strongest dropout predictors
        Tier 2 (dims 6-9):  Clinical trajectory
        Tier 3 (dims 10-11):Programme engagement — leading indicators (4-6wks ahead)
        Static (dims 12-19):Baseline Record A features repeated every event for GRU context
        """
        vec = torch.zeros(20)

        # TIER 1: ENGAGEMENT TRAJECTORY
        vec[0]  = float(delta_t_days) / 30.0                           # Δt — PRIMARY temporal signal (Rossi eq.1)
        vec[1]  = 1.0 if event_type == "DOSE_MISSED" else 0.0
        vec[2]  = min(features.get("silence_days", 0) / 14.0, 1.0)
        vec[3]  = 1.0 if features.get("unable_to_visit_reason") == "patient_refused" else 0.0
        vec[4]  = 1.0 if features.get("unable_to_visit_reason") == "patient_absent"  else 0.0
        vec[5]  = 1.0 if features.get("expressed_reluctance") else 0.0

        # TIER 2: CLINICAL TRAJECTORY
        vec[6]  = max(-1.0, min(1.0, features.get("weight_delta_kg", 0.0) / 5.0))  # RATIONS trial
        vec[7]  = features.get("adr_grade", 0) / 4.0                  # Grade 2+: direct pharmacological dropout path
        vec[8]  = 1.0 if event_type == "PHASE_TRANSITION" else 0.0
        # dim 9: Management outcome signal
        #   -0.5  = referred to higher centre → patient is now in the system (protective)
        #   +1.0  = adr_symptoms present AND not assessed by MO (unmanaged risk)
        #    0.0  = normal / managed
        # Severe non-ADR red flags (haemoptysis, altered consciousness) predict MORTALITY not LTFU
        _referred_up = features.get("management_decision") in (
            "referral_to_higher_centre", "referral_for_hospitalisation"
        )
        if _referred_up:
            vec[9] = -0.5
        elif features.get("adr_symptoms") and not features.get("mo_assessment_done"):
            vec[9] = 1.0
        else:
            vec[9] = 0.0

        # TIER 3: PROGRAMME ENGAGEMENT (inverted: 1.0 = NOT engaged)
        vec[10] = 0.0 if features.get("nikshay_divas_attended", True) else 1.0
        vec[11] = 0.0 if features.get("npy_benefit_received",   True) else 1.0

        # STATIC BASELINE FROM RECORD A (repeated every event for GRU context)
        vec[12] = features.get("treatment_week", 0) / 26.0
        vec[13] = 1.0 if features.get("prior_lfu_history") else 0.0
        vec[14] = 1.0 if features.get("regimen") in (
                    "BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR", "DR_TB") else 0.0
        vec[15] = 1.0 if features.get("distance_to_phc_km", 0) > 10 else 0.0
        vec[16] = 1.0 if features.get("alcohol_use") else 0.0
        vec[17] = 1.0 if features.get("bmi_at_diagnosis", 20) < 18.5 else 0.0
        vec[18] = 1.0 if features.get("marital_status") in ("Divorced", "Separated") else 0.0
        vec[19] = 1.0 if 20 <= features.get("age", 40) <= 39 else 0.0

        return vec.unsqueeze(0)   # (1, 20)


    class RawMessageStore:
        """
        Stores raw messages per node between batches — required for Rossi et al.
        Algorithm 1 (Appendix A.2) correctness.

        The s_i stored here is DETACHED from the current batch's computation graph.
        It is used in the NEXT batch's Step 1 memory update where it IS in the
        computation graph, ensuring GRU memory parameters receive gradients.

        For full-sequence replay architectures the gradient flow issue is less
        severe than in the original online setting, but Algorithm 1 is implemented
        for correctness regardless (v8 §6.2).
        """
        def __init__(self):
            self.store = {}

        def update(self, node_id: str, s_i: torch.Tensor, s_j: torch.Tensor,
                   delta_t: float, features: dict):
            # s_i detached — does not participate in current batch's computation graph
            self.store[node_id] = (s_i.detach(), s_j, delta_t, features)

        def get(self, node_id: str):
            return self.store.get(node_id, None)

        def items(self):
            return self.store.items()

        def clear(self):
            self.store = {}


    class TGNRiskModel(nn.Module):
        """
        Full Temporal Graph Network for TB dropout risk.
        Components:
          1. GRUMemoryModule — updates node memory on each event (20-dim messages)
          2. GATConv — graph attention over SUPERVISED_BY/REGISTERED_AT edges ONLY
          3. MLP prediction head — (64 mem + 64 GATConv = 128-dim) → dropout probability

        v8 Error 4 fix: MLP head receives concatenation of memory + GATConv output (128-dim).
          Previously only GATConv output (64-dim) was passed, losing individual trajectory.
        v8 Error 5 fix: forward() accepts dropout_edge_index — contact edges excluded.
          Passing contact edges forced GATConv to learn dropout prediction and TB contact
          screening simultaneously, degrading both tasks.
        """
        def __init__(self, memory_dim: int = 64, hidden_dim: int = 64,
                     n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.memory_dim  = memory_dim
            self.memory      = GRUMemoryModule(memory_dim, message_dim=20)
            self.attention   = GATConv(memory_dim, hidden_dim // n_heads,
                                       heads=n_heads, dropout=0.1, concat=True)
            self.norm        = nn.LayerNorm(hidden_dim)
            # MLP input = 128 (64 memory + 64 GATConv) — v8 §6.1 hyperparameter table
            self.head        = nn.Sequential(
                nn.Linear(memory_dim + hidden_dim, 64),   # 64+64=128 → 64
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self._attention_weights = None

        def forward(self, memory_vectors: torch.Tensor,
                    dropout_edge_index: torch.Tensor) -> tuple:
            """
            Forward pass.
            memory_vectors:      (N, 64) — current memory state of all nodes
            dropout_edge_index:  (2, E) — SUPERVISED_BY + REGISTERED_AT edges ONLY
                                          contact edges must never be passed here

            Returns: (risk_scores, gat_edge_index, attn_weights)
            """
            # GATConv on dropout-prediction edges only (Error 5 fix)
            h, (gat_edge_index, attn_weights) = self.attention(
                memory_vectors, dropout_edge_index, return_attention_weights=True
            )
            h = self.norm(h)
            self._attention_weights = (gat_edge_index, attn_weights.detach())

            # Concatenate memory + GATConv output: preserve individual trajectory
            # alongside systemic context (Error 4 fix — 128-dim input to MLP)
            combined = torch.cat([memory_vectors, h], dim=-1)   # (N, 128)
            risk     = self.head(combined)                       # sigmoid: P(dropout) in [0,1]
            return risk.squeeze(-1), gat_edge_index, attn_weights

        def extract_top_attention_factors(self, node_idx: int,
                                          gat_edge_index: torch.Tensor,
                                          attn_weights: torch.Tensor,
                                          node_labels: list) -> list:
            """
            For a given patient node, return the top 3 neighbours by attention
            weight — these are the explainability factors shown to the officer.

            gat_edge_index: edge index returned by GATConv (may differ in size
                            from the original edge_index passed to forward()).
                            Shape: (2, E_gat).
            attn_weights:   attention weights from GATConv. Shape: (E_gat, n_heads).
            """
            # Find all edges pointing TO this node in GATConv's edge index
            target_mask   = (gat_edge_index[1] == node_idx)
            source_nodes  = gat_edge_index[0][target_mask]
            weights       = attn_weights[target_mask].mean(dim=-1)

            if weights.numel() == 0:
                return []

            top_k  = min(3, weights.numel())
            top_idx = weights.argsort(descending=True)[:top_k]
            return [
                {
                    "neighbour": (node_labels[source_nodes[i].item()]
                                  if node_labels and source_nodes[i].item() < len(node_labels)
                                  else f"node_{source_nodes[i].item()}"),
                    "attention_weight": round(weights[i].item(), 4)
                }
                for i in top_idx
            ]


    def batch_events(events_sorted: list, batch_size: int = 200):
        """Yield successive batches from a chronologically sorted event list."""
        for i in range(0, len(events_sorted), batch_size):
            yield events_sorted[i: i + batch_size]


    def train_one_epoch(model, event_stream: list, labels: dict, optimizer,
                        loss_fn, node_id_map: dict,
                        dropout_edge_index: torch.Tensor,
                        pos_weight: float = 15.0) -> float:
        """
        Training loop — GRU gradient fix (Rossi et al. 2020 Algorithm 1).

        ROOT CAUSE OF ORIGINAL BUG: every memory tensor was detached before
        entering the loss computation. backward() hit a detached leaf and
        stopped — the GRU cell received exactly zero gradient every epoch.
        Only MLP head and GATConv weights ever updated. The GRU is the entire
        temporal component, so the model never learned event trajectories.

        FIX: GRU forward runs inside the current batch's computation graph.
        live_mem[pid] is an in-graph tensor. It feeds mem_mat → GATConv →
        MLP → loss → backward(). Gradient flows all the way to GRU params.
        Detach happens AFTER backward(), only when handing off to next batch.
        """
        model.train()
        detached_memory: dict = {}   # prev-batch outputs — NOT in current graph
        zeros64     = torch.zeros(64)
        total_loss  = 0.0
        n_batches   = 0
        sorted_nids = sorted(node_id_map, key=node_id_map.get)

        events_sorted = sorted(event_stream, key=lambda e: e["timestamp"])
        pw_tensor     = torch.tensor(pos_weight)

        for batch in batch_events(events_sorted, batch_size=200):
            optimizer.zero_grad()

            # STEP 1 — Base memory from detached prev-batch store (no grad, correct)
            base_mem = torch.stack([
                detached_memory.get(nid, zeros64) for nid in sorted_nids
            ])  # (N, 64)

            # STEP 2 — GRU forward IN-GRAPH for every event in this batch.
            # Multiple events per patient in one batch: chain GRU updates so
            # each event's output is the next event's hidden state. The chain
            # stays live in the graph — backward() reaches GRU params. ✓
            live_mem: dict = {}
            for ev in batch:
                pid = ev.get("patient_id", "")
                if pid not in node_id_map:
                    continue
                h_prev = live_mem.get(pid, base_mem[node_id_map[pid]].unsqueeze(0))
                if h_prev.dim() == 1:
                    h_prev = h_prev.unsqueeze(0)
                msg           = encode_event(ev.get("event_type", ""),
                                             ev.get("features", {}),
                                             ev.get("delta_t", 0.0))
                live_mem[pid] = model.memory(h_prev, msg)   # in-graph ✓

            # STEP 3 — Assemble memory matrix.
            # Updated nodes: in-graph tensor → GRU gets gradient.
            # Untouched nodes: detached base → no spurious gradient.
            mem_rows = []
            for nid in sorted_nids:
                if nid in live_mem:
                    mem_rows.append(live_mem[nid].squeeze(0))
                else:
                    mem_rows.append(base_mem[node_id_map[nid]])
            mem_mat = torch.stack(mem_rows)   # (N, 64)

            # STEP 4-5 — GATConv → norm → concat → MLP → sigmoid
            h, (gat_ei, attn) = model.attention(
                mem_mat, dropout_edge_index, return_attention_weights=True
            )
            h        = model.norm(h)
            combined = torch.cat([mem_mat, h], dim=-1)   # (N, 128)
            all_risk = model.head(combined).squeeze(-1)

            # STEP 6 — Weighted BCE, deduplicated per patient per batch
            seen, unique_pids = set(), []
            for ev in batch:
                pid = ev.get("patient_id", "")
                if pid in labels and pid in node_id_map and pid not in seen:
                    seen.add(pid)
                    unique_pids.append(pid)

            if unique_pids:
                preds_t  = torch.stack([all_risk[node_id_map[p]] for p in unique_pids])
                labels_t = torch.tensor([float(labels[p]) for p in unique_pids])
                weights  = torch.where(labels_t == 1, pw_tensor, torch.ones_like(labels_t))
                loss     = (loss_fn(preds_t, labels_t) * weights).mean()

                # STEP 7 — backward: loss → MLP → GATConv → mem_mat → live_mem → GRU ✓
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            # STEP 8 — Detach AFTER backward and store for next batch
            for nid, h_live in live_mem.items():
                detached_memory[nid] = h_live.squeeze(0).detach()

        return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION MODE — when PyTorch not available
# ─────────────────────────────────────────────────────────────────────────────


def _enrich_event_features_from_record_b(features: dict, patient: dict) -> dict:
    """
    Fix 3: copy adr_symptoms and mo_assessment_done from the latest Record B
    red_flags dict into the TGN event features dict so that encode_event()
    dim 9 (ADR unmanaged) is correctly set.

    Called before every encode_event() invocation that processes a Record B
    or Record C event. Without this, features.get("adr_symptoms") is always
    None and dim 9 of the 20-dim message vector is permanently 0.0.
    """
    records_b = patient.get("records_b", [])
    if not records_b:
        return features

    # Use the most recent Record B with ADR data
    latest_rf = {}
    for rb in sorted(records_b, key=lambda x: x.get("month", 0), reverse=True):
        rf = rb.get("red_flags", {})
        if rf:
            latest_rf = rf
            break

    enriched = dict(features)
    if "adr_symptoms" not in enriched:
        enriched["adr_symptoms"] = latest_rf.get("adr_symptoms", False)
    if "mo_assessment_done" not in enriched:
        enriched["mo_assessment_done"] = latest_rf.get("mo_assessment_done", False)

    # Pull weight_delta, nikshay_divas_attended, npy_benefit_received from
    # most recent Record B (dims 6, 10, 11 of the TGN event vector)
    if records_b:
        latest_b = sorted(records_b, key=lambda x: x.get("month", 0))[-1]
        if "weight_delta_kg" not in enriched:
            enriched["weight_delta_kg"] = latest_b.get("vitals", {}).get("weight_delta_kg", 0.0)
        if "nikshay_divas_attended" not in enriched:
            enriched["nikshay_divas_attended"] = latest_b.get("programme", {}).get("nikshay_divas_attended", True)
        if "npy_benefit_received" not in enriched:
            enriched["npy_benefit_received"] = latest_b.get("programme", {}).get("npy_benefit_received", True)

    return enriched

def simulate_tgn_output(patients: list) -> tuple:
    """
    Simulate TGN output without PyTorch.
    Returns (risk_scores_dict, attention_weights_dict) with realistic values
    derived from the clinical knowledge-based formula.
    Used for prototype demo when GPU/PyTorch not available.
    """
    # Import the knowledge-based formula from Stage 3 as a stand-in
    from stage3_score import compute_risk_score

    risk_scores = {}
    attention_weights = {}

    for p in patients:
        result = compute_risk_score(p)
        pid    = p["patient_id"]
        risk_scores[pid] = result["risk_score"]

        # Simulate attention weights using factor magnitudes
        factors = result.get("all_factors", {})
        total   = sum(abs(np.log(v)) for v in factors.values() if v > 0) or 1
        attention_weights[pid] = {
            k: round(abs(np.log(v)) / total, 4)
            for k, v in list(factors.items())[:3]
        }

    return risk_scores, attention_weights


# ─────────────────────────────────────────────────────────────────────────────
# BUILD PYTORCH GRAPH from patient records
# ─────────────────────────────────────────────────────────────────────────────

def build_pytorch_graph(patients: list) -> tuple:
    """
    Build dropout_edge_index, contact_edge_index, memory_vectors, node_id_map.

    Graph structure (per clinical hierarchy):
      Patient → SUPERVISED_BY → ASHA  (weekly temporal data from Record C)
      Patient → ASSESSED_BY   → CHO   (monthly temporal data from Record B)
      Patient → REGISTERED_AT → PHC   (static facility anchor)
      Patient → HOUSEHOLD_CONTACT → Contact (contact network — PageRank only)

    ANM nodes removed — ANM has no direct patient interaction and no temporal data.
    CHO nodes added — monthly Record B assessments are genuine temporal events.
    PHC node added — single district anchor providing structural context.

    dropout_edge_index: SUPERVISED_BY + ASSESSED_BY + REGISTERED_AT → GATConv
    contact_edge_index: HOUSEHOLD_CONTACT → PageRank only, never GATConv
    """
    if not TORCH_AVAILABLE:
        return None, None, None, {}

    node_id_map    = {}
    memory_list    = []
    dropout_edges  = []
    contact_edges  = []

    # ── PHC anchor node (single node for the district facility) ──────────────
    PHC_NODE_ID = "PHC_DISTRICT"
    node_id_map[PHC_NODE_ID] = len(node_id_map)
    phc_mem = torch.zeros(64)
    phc_mem[0] = float(len(patients)) / 500.0   # relative load
    memory_list.append(phc_mem)

    # ── Index patient nodes ───────────────────────────────────────────────────
    for p in patients:
        idx = len(node_id_map)
        node_id_map[p["patient_id"]] = idx

        mem = torch.zeros(64)
        treatment_week = p.get("treatment_week") or (
            (p.get("clinical") or {}).get("total_treatment_days", 0) // 7
        )
        mem[0] = float(min(treatment_week, 26)) / 26.0

        adh = p.get("adherence") or {}
        mem[1] = float(adh.get("days_since_last_dose", 0)) / 14.0
        mem[2] = float(adh.get("adherence_rate_30d", 1.0))
        dist   = adh.get("distance_to_center_km") or (
            (p.get("baseline_clinical") or {}).get("distance_to_phc_km", 0)
        ) or 0
        mem[3] = float(dist) / 20.0

        base_cl       = p.get("baseline_clinical") or p.get("clinical") or {}
        comorbidities = base_cl.get("comorbidities", {})
        mem[4] = 1.0 if (comorbidities.get("hiv") or base_cl.get("hiv")) else 0.0
        mem[5] = 1.0 if (adh.get("prior_lfu_history") or base_cl.get("prior_lfu_history")) else 0.0
        mem[6] = float(min(treatment_week, 26)) / 26.0

        # Record B CHO data — weight delta and ADR from last monthly assessment
        records_b = p.get("records_b", [])
        if records_b:
            last_b = sorted(records_b, key=lambda x: x.get("month", 0))[-1]
            mem[7] = float(last_b.get("vitals", {}).get("weight_delta_kg", 0.0)) / 5.0
            mem[8] = float(last_b.get("adr", {}).get("grade", 0)) / 4.0
            mem[9] = 1.0 if last_b.get("red_flags", {}).get("any_red_flag_positive") else 0.0

        memory_list.append(mem)

    # ── Index ASHA nodes and build SUPERVISED_BY edges ────────────────────────
    for p in patients:
        pid     = p["patient_id"]
        op      = p.get("operational") or {}
        asha_id = op.get("asha_id")
        if asha_id and asha_id not in node_id_map:
            node_id_map[asha_id] = len(node_id_map)
            mem = torch.zeros(64)
            mem[0] = float(op.get("asha_load_score", p.get("asha_load_score", 0.3)))
            memory_list.append(mem)

        if asha_id and asha_id in node_id_map:
            dropout_edges.append([node_id_map[pid], node_id_map[asha_id]])
            dropout_edges.append([node_id_map[asha_id], node_id_map[pid]])

    # ── Index CHO nodes and build ASSESSED_BY edges ───────────────────────────
    for p in patients:
        pid    = p["patient_id"]
        op     = p.get("operational") or {}
        cho_id = op.get("cho_id")
        if not cho_id:
            continue
        if cho_id not in node_id_map:
            node_id_map[cho_id] = len(node_id_map)
            mem = torch.zeros(64)
            # CHO memory seeded with caseload proxy and last assessment recency
            cho_patients = [q for q in patients
                            if (q.get("operational") or {}).get("cho_id") == cho_id]
            mem[0] = min(float(len(cho_patients)) / 45.0, 1.0)  # 45 = typical CHO caseload
            records_b_counts = [len(q.get("records_b", [])) for q in cho_patients]
            mem[1] = float(sum(records_b_counts)) / max(len(cho_patients) * 6, 1)
            memory_list.append(mem)

        if cho_id in node_id_map:
            # Weight by recency of last Record B — more recent = stronger edge
            records_b = p.get("records_b", [])
            last_month = max((rb.get("month", 0) for rb in records_b), default=0)
            edge_weight_dim = min(float(last_month) / 6.0, 1.0)
            dropout_edges.append([node_id_map[pid], node_id_map[cho_id]])
            dropout_edges.append([node_id_map[cho_id], node_id_map[pid]])

    # ── Build REGISTERED_AT edges (patient → PHC) ────────────────────────────
    phc_idx = node_id_map[PHC_NODE_ID]
    for p in patients:
        pid = p["patient_id"]
        if pid in node_id_map:
            dropout_edges.append([node_id_map[pid], phc_idx])
            dropout_edges.append([phc_idx, node_id_map[pid]])

    # ── Index contact nodes (contact_edge_index only — never to GATConv) ─────
    for p in patients:
        pid = p["patient_id"]
        contacts = p.get("contact_network") or []
        for c in contacts:
            cid = f"CONTACT_{c['name'].replace(' ', '_').replace('.', '')}"
            if cid not in node_id_map:
                node_id_map[cid] = len(node_id_map)
                mem = torch.zeros(64)
                mem[0] = float(c.get("vulnerability_score", 1.0)) / 2.0
                rel = c.get("rel") or c.get("relationship_type", "Household")
                mem[1] = 1.0 if rel == "Household" else 0.5
                memory_list.append(mem)
            contact_edges.append([node_id_map[pid], node_id_map[cid]])
            contact_edges.append([node_id_map[cid], node_id_map[pid]])

    memory_vectors     = torch.stack(memory_list)
    dropout_edge_index = (
        torch.tensor(dropout_edges, dtype=torch.long).t().contiguous()
        if dropout_edges else torch.zeros(2, 0, dtype=torch.long)
    )
    contact_edge_index = (
        torch.tensor(contact_edges, dtype=torch.long).t().contiguous()
        if contact_edges else torch.zeros(2, 0, dtype=torch.long)
    )

    return dropout_edge_index, contact_edge_index, memory_vectors, node_id_map


def build_event_stream(patients: list) -> list:
    """
    Build a chronologically sorted event stream from Record B and Record C data.
    This is what train_one_epoch() consumes — without this the training loop
    receives an empty list and the GATConv weights stay random.

    Each event has:
        patient_id, timestamp, event_type, delta_t (days since prev event), features
    """
    from datetime import datetime, timezone, timedelta

    events = []
    base_date = datetime(2025, 3, 1, tzinfo=timezone.utc)

    for p in patients:
        pid        = p["patient_id"]
        start_str  = p.get("treatment_start_date", "")
        op         = p.get("operational") or {}
        adh        = p.get("adherence") or {}
        base_cl    = p.get("baseline_clinical") or p.get("clinical") or {}
        soc        = p.get("social") or {}
        diag       = p.get("diagnosis") or p.get("clinical") or {}
        demo       = p.get("demographics") or {}

        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00")) \
                       if start_str else base_date
        except Exception:
            start_dt = base_date

        static_features = {
            "treatment_week":    p.get("treatment_week", 1),
            "prior_lfu_history": adh.get("prior_lfu_history") or base_cl.get("prior_lfu_history", False),
            "regimen":           diag.get("regimen", "DS-TB"),
            "distance_to_phc_km": soc.get("distance_to_phc_km") or adh.get("distance_to_center_km", 0),
            "alcohol_use":       soc.get("alcohol_use", False),
            "bmi_at_diagnosis":  base_cl.get("bmi", 20),
            "marital_status":    demo.get("marital_status", ""),
            "age":               demo.get("age", 35),
        }

        last_event_date = start_dt

        # ── Events from Record C (weekly ASHA visits) ─────────────────────────
        records_c = p.get("records_c", [])
        for rc in sorted(records_c, key=lambda x: x.get("week", 0)):
            week = rc.get("week", 1)
            event_date = start_dt + timedelta(weeks=week)
            delta_t = max(0.0, (event_date - last_event_date).total_seconds() / 86400)
            last_event_date = event_date

            dose_status = rc.get("dose_status", "confirmed")
            event_type  = "DOSE_MISSED" if dose_status == "missed" else "DOSE_CONFIRMED"

            rf = rc.get("red_flags", {})
            pf = rc.get("patient_flags", {})

            features = {
                **static_features,
                "silence_days":           rc.get("silence_days", 0),
                "unable_to_visit_reason": rc.get("unable_to_visit_reason"),
                "expressed_reluctance":   pf.get("expressed_reluctance", False),
                "adr_symptoms":           rf.get("adr_symptoms", False),
                "mo_assessment_done":     rf.get("mo_assessment_done", False),
                "weight_delta_kg":        0.0,
                "adr_grade":              0,
                "nikshay_divas_attended": True,
                "npy_benefit_received":   True,
            }

            events.append({
                "patient_id":    pid,
                "counterparty_id": op.get("asha_id", ""),
                "timestamp":     event_date.isoformat(),
                "event_type":    event_type,
                "delta_t":       delta_t,
                "features":      features,
            })

        # ── Events from Record B (monthly CHO assessments) ────────────────────
        records_b = p.get("records_b", [])
        for rb in sorted(records_b, key=lambda x: x.get("month", 0)):
            month = rb.get("month", 1)
            event_date = start_dt + timedelta(days=month * 30)
            delta_t = max(0.0, (event_date - last_event_date).total_seconds() / 86400)
            last_event_date = event_date

            vitals   = rb.get("vitals", {})
            adr      = rb.get("adr", {})
            prog     = rb.get("programme", {})
            rf       = rb.get("red_flags", {})

            mgmt_dec = prog.get("management_decision", "")
            # If referred, MO has formally assessed the patient — mark mo_assessment_done
            mo_done = rf.get("mo_assessment_done", False) or mgmt_dec in (
                "referral_to_higher_centre", "referral_for_hospitalisation"
            )

            features = {
                **static_features,
                "silence_days":           0,
                "unable_to_visit_reason": None,
                "expressed_reluctance":   False,
                "adr_symptoms":           rf.get("adr_symptoms", False),
                "mo_assessment_done":     mo_done,
                "weight_delta_kg":        vitals.get("weight_delta_kg", 0.0),
                "adr_grade":              adr.get("grade", 0),
                "nikshay_divas_attended": prog.get("nikshay_divas_attended", True),
                "npy_benefit_received":   prog.get("npy_benefit_received", True),
                # Management decision from CHO Annexure 1 — referral is protective (dim 9)
                "management_decision":    mgmt_dec,
            }

            events.append({
                "patient_id":    pid,
                "counterparty_id": op.get("cho_id", ""),
                "timestamp":     event_date.isoformat(),
                "event_type":    "CLINICAL_ASSESSMENT",
                "delta_t":       delta_t,
                "features":      features,
            })

    events.sort(key=lambda e: e["timestamp"])
    return events


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY PERSISTENCE — Cosmos DB
# ─────────────────────────────────────────────────────────────────────────────

def save_memory_to_cosmos(gc, patient_id: str, memory_vector):
    """
    Persist updated TGN memory vector to Cosmos DB patient node.
    Memory survives across sessions — patient history is not lost.
    """
    if gc is None:
        return
    if TORCH_AVAILABLE and hasattr(memory_vector, 'tolist'):
        mem_list = memory_vector.tolist()
    else:
        mem_list = list(memory_vector)

    from datetime import datetime, timezone
    mem_str    = json.dumps(mem_list)
    updated_at = datetime.now(timezone.utc).isoformat()

    from cosmos_client import run_query, safe
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('memory_vector', '{safe(mem_str)}')"
        f".property('memory_updated_at', '{safe(updated_at)}')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AZURE ML ENDPOINT CALL (production path)
# ─────────────────────────────────────────────────────────────────────────────

def call_azure_ml_endpoint(patient_features: list) -> list:
    """
    TODO: In production, replace simulate_tgn_output() with this function.
    Calls the TGN deployed as an Azure ML managed online endpoint.

    Setup:
    1. Azure ML workspace → Models → Register your trained TGN .pt file
    2. Endpoints → Create managed online endpoint
    3. Deploy model to endpoint with GPU compute
    4. Get scoring URI from endpoint details

    Returns list of risk score floats.
    """
    import requests
    endpoint_url = os.getenv("AZURE_ML_ENDPOINT_URL")
    endpoint_key = os.getenv("AZURE_ML_ENDPOINT_KEY")

    if not endpoint_url or not endpoint_key:
        raise ValueError("AZURE_ML_ENDPOINT_URL and AZURE_ML_ENDPOINT_KEY not set in .env")

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {endpoint_key}"
    }
    payload = {"input_data": {"data": patient_features}}
    response = requests.post(endpoint_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"]


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PATIENT LIVE RESCORING (for ASHA dashboard updates)
# ─────────────────────────────────────────────────────────────────────────────

def is_tgn_trained() -> bool:
    """
    Returns True only when PyTorch is available AND data/tgn_weights.pt exists.
    Used by rescore_patient_locally() to decide whether to attempt live TGN
    re-inference or fall back to BBN-only mode.
    """
    from pathlib import Path as _Path
    return TORCH_AVAILABLE and _Path("data/tgn_weights.pt").exists()


def score_single_patient(patient: dict, all_patients: list, delta_t: float = 7.0) -> float:
    """
    Fast TGN re-score for one patient after a dashboard update.

    Does NOT call build_event_stream(all_patients) — that processes every
    patient's full history (~1700 events) and takes seconds per click.
    Instead rolls GRU through only THIS patient's own records_b + records_c,
    then applies the current update event on top.
    GATConv still attends over the full graph (all_patients needed for edges).

    Model is cached after first load — no disk I/O on repeated updates.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available — use BBN-only mode")

    from pathlib import Path as _Path
    weights_path = _Path("data/tgn_weights.pt")
    if not weights_path.exists():
        raise RuntimeError("TGN weights not found — use BBN-only mode")
    if len(all_patients) < 5:
        raise RuntimeError("Too few patients for reliable TGN graph — use BBN-only mode")

    # Load model once, re-use on subsequent calls
    global _CACHED_TGN_MODEL, _CACHED_TGN_MTIME
    try:
        mtime = weights_path.stat().st_mtime
    except OSError:
        mtime = 0
    if _CACHED_TGN_MODEL is None or _CACHED_TGN_MTIME != mtime:
        _CACHED_TGN_MODEL = TGNRiskModel()
        _CACHED_TGN_MODEL.load_state_dict(
            torch.load(str(weights_path), map_location="cpu")
        )
        _CACHED_TGN_MODEL.eval()
        _CACHED_TGN_MTIME = mtime
    model = _CACHED_TGN_MODEL

    dropout_edge_index, _contact_ei, memory_vectors, node_id_map = \
        build_pytorch_graph(all_patients)

    pid = patient["patient_id"]
    if pid not in node_id_map:
        raise RuntimeError(f"Patient {pid} not found in graph node map")
    pid_idx = node_id_map[pid]

    adh     = patient.get("adherence") or {}
    op      = patient.get("operational") or {}
    base_cl = patient.get("baseline_clinical") or patient.get("clinical") or {}
    soc     = patient.get("social") or {}
    diag    = patient.get("diagnosis") or patient.get("clinical") or {}
    demo    = patient.get("demographics") or {}

    static_features = {
        "treatment_week":     patient.get("treatment_week", 1),
        "prior_lfu_history":  adh.get("prior_lfu_history") or base_cl.get("prior_lfu_history", False),
        "regimen":            diag.get("regimen", "DS-TB"),
        "distance_to_phc_km": soc.get("distance_to_phc_km") or adh.get("distance_to_center_km", 0),
        "alcohol_use":        soc.get("alcohol_use", False),
        "bmi_at_diagnosis":   base_cl.get("bmi", 20),
        "marital_status":     demo.get("marital_status", ""),
        "age":                demo.get("age", 35),
    }

    from datetime import datetime, timezone as _tz, timedelta
    start_str = patient.get("treatment_start_date", "")
    try:
        start_dt = datetime.fromisoformat(
            start_str.replace("Z", "+00:00")) if start_str \
            else datetime(2025, 3, 1, tzinfo=_tz.utc)
    except Exception:
        start_dt = datetime(2025, 3, 1, tzinfo=_tz.utc)

    with torch.no_grad():
        h       = memory_vectors[pid_idx].unsqueeze(0)   # (1, 64) static seed
        last_dt = start_dt

        for rc in sorted(patient.get("records_c", []), key=lambda x: x.get("week", 0)):
            ev_dt    = start_dt + timedelta(weeks=rc.get("week", 1))
            dt_      = max(0.0, (ev_dt - last_dt).total_seconds() / 86400)
            last_dt  = ev_dt
            rf, pf   = rc.get("red_flags", {}), rc.get("patient_flags", {})
            ds       = rc.get("dose_status", "confirmed")
            feats    = {**static_features,
                        "silence_days":           rc.get("silence_days", 0),
                        "unable_to_visit_reason": rc.get("unable_to_visit_reason"),
                        "expressed_reluctance":   pf.get("expressed_reluctance", False),
                        "adr_symptoms":           rf.get("adr_symptoms", False),
                        "mo_assessment_done":     rf.get("mo_assessment_done", False),
                        "weight_delta_kg": 0.0, "adr_grade": 0,
                        "nikshay_divas_attended": True, "npy_benefit_received": True}
            h = model.memory(h, encode_event(
                "DOSE_MISSED" if ds == "missed" else "DOSE_CONFIRMED", feats, dt_))

        for rb in sorted(patient.get("records_b", []), key=lambda x: x.get("month", 0)):
            ev_dt   = start_dt + timedelta(days=rb.get("month", 1) * 30)
            dt_     = max(0.0, (ev_dt - last_dt).total_seconds() / 86400)
            last_dt = ev_dt
            vitals, adr_, prog, rf = (rb.get("vitals", {}), rb.get("adr", {}),
                                      rb.get("programme", {}), rb.get("red_flags", {}))
            feats   = {**static_features,
                       "silence_days": 0, "unable_to_visit_reason": None,
                       "expressed_reluctance": False,
                       "adr_symptoms":         rf.get("adr_symptoms", False),
                       "mo_assessment_done":   rf.get("mo_assessment_done", False),
                       "weight_delta_kg":      vitals.get("weight_delta_kg", 0.0),
                       "adr_grade":            adr_.get("grade", 0),
                       "nikshay_divas_attended": prog.get("nikshay_divas_attended", True),
                       "npy_benefit_received":   prog.get("npy_benefit_received", True)}
            h = model.memory(h, encode_event("CLINICAL_ASSESSMENT", feats, dt_))

        # Apply current dashboard update event on top of rolled-out history
        days_missed    = adh.get("days_since_last_dose", 0)
        visit_days_ago = op.get("last_asha_visit_days_ago", 0)
        event_type     = "DOSE_MISSED" if (days_missed > 0 or visit_days_ago > 1) else "DOSE_CONFIRMED"
        cur_feats = {**static_features,
                     "silence_days":           max(op.get("silence_days", 0),
                                                   adh.get("silence_days", 0),
                                                   days_missed, visit_days_ago),
                     "unable_to_visit_reason": op.get("unable_to_visit_reason"),
                     "expressed_reluctance":   False,
                     "adr_symptoms":           adh.get("adr_symptoms", False),
                     "mo_assessment_done":     adh.get("mo_assessment_done", False),
                     "weight_delta_kg":        adh.get("weight_delta_kg", 0.0),
                     "adr_grade":              adh.get("adr_grade", 0),
                     "nikshay_divas_attended": adh.get("nikshay_divas_attended", True),
                     "npy_benefit_received":   op.get("welfare_enrolled", True)}
        cur_feats = _enrich_event_features_from_record_b(cur_feats, patient)
        h = model.memory(h, encode_event(event_type, cur_feats, delta_t))

        updated_mem          = memory_vectors.clone()
        updated_mem[pid_idx] = h.squeeze(0)
        risk_tensor, _, _    = model(updated_mem, dropout_edge_index)

    return round(float(risk_tensor[pid_idx].item()), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tgn_inference(patients: list, gc=None) -> tuple:
    """
    Run TGN inference on patient list.
    Returns (risk_scores_dict, attention_weights_dict)

    Production path: calls Azure ML endpoint.
    Prototype path: trains locally then runs inference.
      - Loads saved weights from data/tgn_weights.pt if available
      - Otherwise trains for 20 epochs on the patient event stream
      - Saves weights after training so each run accumulates learning
    """
    azure_ml_url = os.getenv("AZURE_ML_ENDPOINT_URL")

    if azure_ml_url:
        print("  [Stage 2] Running TGN inference via Azure ML endpoint...")
        try:
            features = [
                _enrich_event_features_from_record_b(
                    {"patient_id": p["patient_id"], **(p.get("adherence") or {})},
                    p
                )
                for p in patients
            ]
            scores   = call_azure_ml_endpoint(features)
            risk_scores = {p["patient_id"]: s for p, s in zip(patients, scores)}
            return risk_scores, {}
        except Exception as e:
            print(f"  Azure ML call failed ({e}), falling back to local mode.")

    if TORCH_AVAILABLE:
        import os as _os
        from pathlib import Path as _Path

        print("  [Stage 2] Building graph...")
        dropout_edge_index, contact_edge_index, memory_vectors, node_id_map = \
            build_pytorch_graph(patients)

        weights_path = _Path("data/tgn_weights.pt")
        model = TGNRiskModel()

        if weights_path.exists():
            try:
                model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
                print("  [Stage 2] Loaded saved TGN weights from data/tgn_weights.pt")
            except Exception as e:
                print(f"  [Stage 2] Could not load weights ({e}) — retraining")
                weights_path = _Path("data/tgn_weights_INVALID")

        if not weights_path.exists() or str(weights_path).endswith("INVALID"):
            print("  [Stage 2] Training TGN for 40 epochs on patient event stream...")
            event_stream = build_event_stream(patients)
            labels = {}
            for p in patients:
                pid    = p["patient_id"]
                stored = p.get("dropout_label")
                if stored is not None and int(stored) in (0, 1):
                    labels[pid] = float(int(stored))
                else:
                    rcs      = p.get("records_c", [])
                    recent   = rcs[-6:] if len(rcs) >= 6 else rcs
                    n_missed = sum(1 for rc in recent if rc.get("dose_status") == "missed")
                    days_out = (p.get("adherence") or {}).get("days_since_last_dose", 0)
                    labels[pid] = 1.0 if (n_missed / max(len(recent), 1) >= 0.5
                                          and days_out >= 14) else 0.0

            n_pos = sum(1 for v in labels.values() if v == 1.0)
            n_neg = sum(1 for v in labels.values() if v == 0.0)

            if event_stream and n_pos > 0 and n_neg > 0:
                # pos_weight clamped to [1.5, 20.0]. Using the raw dataset ratio
                # (n_neg/n_pos ≈ 2.3 at 30% synthetic dropout) causes the model to
                # collapse — it learns to predict "completer" for everyone since that
                # minimises BCE under mild class weighting. Clamping to ≥1.5 keeps
                # minority class pressure without overcorrecting.
                dynamic_pos_weight = float(np.clip(n_neg / n_pos, 1.5, 20.0))
                print(f"  [Stage 2] Labels: {n_pos} dropout / {n_neg} non-dropout "
                      f"({100*n_pos/(n_pos+n_neg):.1f}%)  pos_weight={dynamic_pos_weight:.2f}")

                optimizer  = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
                loss_fn    = torch.nn.BCELoss(reduction="none")
                best_loss  = float("inf")
                best_state = None
                patience   = 0

                for epoch in range(40):
                    loss = train_one_epoch(
                        model, event_stream, labels, optimizer,
                        loss_fn, node_id_map, dropout_edge_index,
                        pos_weight=dynamic_pos_weight,
                    )
                    if epoch % 5 == 0 or epoch == 39:
                        print(f"    Epoch {epoch:02d}  loss={loss:.4f}")
                    if loss < best_loss:
                        best_loss  = loss
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        patience   = 0
                    else:
                        patience  += 1
                    if patience >= 8:
                        print(f"    Early stop at epoch {epoch}")
                        break

                if best_state:
                    model.load_state_dict(best_state)
                _Path("data").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "data/tgn_weights.pt")
                print(f"  [Stage 2] Training complete. Best loss={best_loss:.4f}. "
                      f"Weights saved to data/tgn_weights.pt")
            else:
                print(f"  [Stage 2] Insufficient labelled data "
                      f"(n_pos={n_pos}, n_neg={n_neg}) — using untrained model")

        # ── Inference-time GRU rollout ────────────────────────────────────────
        # Apply trained GRU over full event stream so each patient's memory
        # reflects their actual temporal trajectory, not just the static seed
        # from build_pytorch_graph(). Without this, TGN scores ≈ BBN scores
        # (nonlinear function of the same static features, no temporal signal).
        model.eval()
        event_stream_inf = build_event_stream(patients)
        inf_memory: dict = {}

        with torch.no_grad():
            for ev in event_stream_inf:
                pid = ev.get("patient_id", "")
                if pid not in node_id_map:
                    continue
                pid_idx = node_id_map[pid]
                h_prev  = inf_memory.get(pid, memory_vectors[pid_idx].unsqueeze(0))
                if h_prev.dim() == 1:
                    h_prev = h_prev.unsqueeze(0)
                msg             = encode_event(ev.get("event_type", ""),
                                               ev.get("features", {}),
                                               ev.get("delta_t", 0.0))
                inf_memory[pid] = model.memory(h_prev, msg).squeeze(0)

            updated_mem = memory_vectors.clone()
            for nid, h in inf_memory.items():
                if nid in node_id_map:
                    updated_mem[node_id_map[nid]] = h

            risk_tensor, gat_edge_index, attn = model(updated_mem, dropout_edge_index)

        risk_scores       = {}
        attention_weights = {}
        node_labels       = list(node_id_map.keys())

        for p in patients:
            pid = p["patient_id"]
            idx = node_id_map.get(pid, 0)
            risk_scores[pid] = round(float(risk_tensor[idx].item()), 4)

            top_factors = model.extract_top_attention_factors(
                idx, gat_edge_index, attn, node_labels
            )
            attention_weights[pid] = top_factors

            if gc:
                save_memory_to_cosmos(gc, pid, memory_vectors[idx])

        print(f"  [Stage 2] TGN inference complete. {len(risk_scores)} patients scored.")
        return risk_scores, attention_weights

    else:
        print("  [Stage 2] Running TGN in simulation mode (PyTorch not available).")
        return simulate_tgn_output(patients)


if __name__ == "__main__":
    with open("nikshay_grounded_dataset.json") as f:
        patients = json.load(f)[:100]

    risk_scores, attn = run_tgn_inference(patients)
    print(f"\nSample outputs (first 5 patients):")
    for pid, score in list(risk_scores.items())[:5]:
        factors = attn.get(pid, [])
        print(f"  {pid}: risk={score}  attention={factors}")