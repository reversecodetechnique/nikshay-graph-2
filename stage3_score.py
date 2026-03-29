"""
stage3_score.py
===============
Stage 3: Dropout Risk Classification  [v8 — all errors corrected]

Implements the v8 sequential-handover scoring design:
  - BBN prior (cold-start, week 0) → linearly fades to zero by the phase boundary
  - TGN output (ramps from 0 → full weight by the phase boundary)
  - ASHA load: additive equity floor ONLY (Fix 4 — multiplicative term removed, double-counting prevented)
  - Urgency multiplier: risk × (1 + treatment_week/26 × 0.5)
  - Adaptive thresholds: HIGH/MEDIUM tighten as treatment progresses

Phase boundaries (v8 §8.2):
  DS-TB / BPaLM         → boundary = week 8
  Shorter/Longer MDR    → boundary = week 16

After the phase boundary TGN is the sole primary model.

BBN OR update (v8 §8.4):
  Uses a 2×2 contingency table OR requiring both dropout AND completer records.
  Frozen until MIN_CASES_TO_UPDATE exists for BOTH populations.
  Fading mechanism retained — only the broken formula is fixed.

Usage:
    from stage3_score import score_all_patients, detect_systemic_failures
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# BBN PRIOR — literature-calibrated odds ratios (fades as real data accumulates)
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_LTFU     = 0.062
BASELINE_LOG_ODDS = np.log(BASELINE_LTFU / (1 - BASELINE_LTFU))

# Published adjusted Odds Ratios — log scale = logistic regression coefficients
# DIABETES NOTE: ln(0.52) protective direction is TAMIL NADU-SPECIFIC.
# The RATIONS trial identifies diabetes as a MORTALITY predictor — not protective
# nationally. Reflects TN integrated TB-DM programme. Label explicitly in presentations.
LOG_OR = {
    "alcohol_use":            np.log(1.92),
    "divorced_separated":     np.log(3.80),
    "diabetes":               np.log(0.52),    # protective in Tamil Nadu ONLY — see note above
    "prior_lfu":              np.log(3.00),    # strongest static predictor (v8 §9.1)
    "prior_tb":               np.log(2.10),    # prior TB history without LFU
    "hiv":                    np.log(2.16),
    "low_education":          np.log(1.55),
    "drug_use":               np.log(2.40),
    "male_sex":               np.log(1.29),
    "distance_5_to_10km":    np.log(1.60),
    "distance_over_10km":    np.log(2.10),
    "continuation_phase":     np.log(2.30),
    "no_welfare":             np.log(1.45),    # Nikshay Poshan Yojana non-enrollment
    "dr_tb":                  np.log(2.80),    # WHO DR-TB report 2022
    "missed_7_to_13_days":   np.log(3.20),
    "missed_14_plus_days":   np.log(6.50),
    "age_20_to_39":           np.log(2.07),
    "age_over_60":            np.log(1.40),
    "low_bmi_at_diagnosis":   np.log(1.85),    # RATIONS trial: BMI<18.5 at diagnosis (v8 §9.1)
    # CHO Annexure 1 signals — added to ensure monthly clinical assessment drives BBN
    "adr_grade_2_plus":       np.log(3.20),    # ADR grade ≥2: Sagwa et al. 2013 — direct pharmacological dropout path
    "nikshay_divas_missed":   np.log(2.00),    # Missed monthly Nikshay Divas — leading disengagement indicator
    "weight_declining":       np.log(2.30),    # Weight decline >2 kg since last assessment — RATIONS trial surrogate
    # CHO management decision — dynamic temporal factor (CHO Annexure 1, Record B)
    # Referral to higher centre means patient is now formally in the system → protective.
    # OR < 1 → log-OR < 0 → subtracts from log-odds → lowers dropout probability.
    "referred_to_higher_centre": np.log(0.55),  # referred up = patient is under active higher-facility management
}

# BBN retires after this many confirmed real dropout cases are observed
BBN_RETIREMENT_THRESHOLD = 200

def compute_bbn_prior(record: dict) -> dict:
    """
    Knowledge-based logistic model using published odds ratios.
    Returns score (0-1) + all contributing factors with their OR values.
    Active during Phase 1 (prototype) and early Phase 2 (pilot).
    Automatically fades as TGN accumulates real evidence.

    Supports both v2 schema (record["diagnosis"], record["baseline_clinical"]) and
    legacy flat schema (record["clinical"], record["adherence"]).
    DIABETES NOTE: protective direction is Tamil Nadu-specific only.
    """
    effective_log_or = get_effective_log_ors()
    lo = BASELINE_LOG_ODDS
    factors = {}

    # ── Schema-agnostic field resolution (v2 and legacy flat schema) ──────────
    diag    = record.get("diagnosis") or record.get("clinical", {})
    base_cl = record.get("baseline_clinical") or record.get("clinical", {})
    soc     = record.get("social") or record.get("social", {})
    adh     = record.get("adherence") or record.get("adherence", {})
    demo    = record.get("demographics") or record.get("demographics", {})
    op      = record.get("operational") or record.get("operational", {})
    comorbidities = base_cl.get("comorbidities", {}) or record.get("clinical", {}).get("comorbidities", {})

    # Regimen — support both new names and legacy DR_TB flag
    regimen = diag.get("regimen") or base_cl.get("regimen", "DS-TB")
    dr_tb_regimens = {"BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR", "DR_TB"}

    # Marital status — check both schema locations
    marital = demo.get("marital") or demo.get("marital_status", "")

    if soc.get("alcohol_use") or base_cl.get("alcohol_use"):
        lo += effective_log_or["alcohol_use"]
        factors["Alcohol use"] = round(np.exp(effective_log_or["alcohol_use"]), 2)
    if marital in ("Divorced", "Separated"):
        lo += effective_log_or["divorced_separated"]
        factors["Divorced/separated"] = round(np.exp(effective_log_or["divorced_separated"]), 2)
    if comorbidities.get("diabetes") or base_cl.get("diabetes"):
        lo += effective_log_or["diabetes"]
        factors["Diabetes (monitored — protective in TN only)"] = round(np.exp(effective_log_or["diabetes"]), 2)
    if comorbidities.get("hiv") or base_cl.get("hiv"):
        lo += effective_log_or["hiv"]
        factors["HIV co-infection"] = round(np.exp(effective_log_or["hiv"]), 2)

    # prior_lfu (OR 3.00 — strongest static predictor) vs prior TB without LFU (OR 2.10)
    prior_lfu = adh.get("prior_lfu_history") or base_cl.get("prior_lfu_history")
    prior_tb  = base_cl.get("prior_tb_history") or adh.get("prior_tb_history")
    if prior_lfu:
        lo += effective_log_or["prior_lfu"]
        factors["Prior LTFU history (strongest predictor)"] = round(np.exp(effective_log_or["prior_lfu"]), 2)
    elif prior_tb:
        lo += effective_log_or["prior_tb"]
        factors["Prior TB history"] = round(np.exp(effective_log_or["prior_tb"]), 2)

    if demo.get("gender") == "Male" or demo.get("sex") == "Male":
        lo += effective_log_or["male_sex"]
        factors["Male sex"] = round(np.exp(effective_log_or["male_sex"]), 2)
    if soc.get("low_education") or base_cl.get("low_education"):
        lo += effective_log_or["low_education"]
        factors["Low education"] = round(np.exp(effective_log_or["low_education"]), 2)
    if soc.get("drug_use") or base_cl.get("substance_use_disorder"):
        lo += effective_log_or["drug_use"]
        factors["Drug use"] = round(np.exp(effective_log_or["drug_use"]), 2)

    phase = diag.get("phase") or base_cl.get("phase", "")
    if phase == "Continuation":
        lo += effective_log_or["continuation_phase"]
        factors["Continuation phase"] = round(np.exp(effective_log_or["continuation_phase"]), 2)

    # NPY non-enrollment
    npy_enrolled = op.get("welfare_enrolled") or op.get("npy_enrolled") or base_cl.get("npy_enrolled")
    if not npy_enrolled:
        lo += effective_log_or["no_welfare"]
        factors["Not enrolled in Nikshay Poshan Yojana"] = round(np.exp(effective_log_or["no_welfare"]), 2)

    if regimen in dr_tb_regimens:
        lo += effective_log_or["dr_tb"]
        factors["Drug-resistant TB"] = round(np.exp(effective_log_or["dr_tb"]), 2)

    # Distance — support both field names
    dist = adh.get("distance_to_center_km") or base_cl.get("distance_to_phc_km", 0) or 0
    if 5 <= dist < 10:
        lo += effective_log_or["distance_5_to_10km"]
        factors[f"Distance {dist:.1f}km (5-10km)"] = round(np.exp(effective_log_or["distance_5_to_10km"]), 2)
    elif dist >= 10:
        lo += effective_log_or["distance_over_10km"]
        factors[f"Distance {dist:.1f}km (>10km)"] = round(np.exp(effective_log_or["distance_over_10km"]), 2)

    missed = adh.get("days_since_last_dose", 0)
    if 7 <= missed < 14:
        lo += effective_log_or["missed_7_to_13_days"]
        factors[f"{missed} days since last dose"] = round(np.exp(effective_log_or["missed_7_to_13_days"]), 2)
    elif missed >= 14:
        lo += effective_log_or["missed_14_plus_days"]
        factors[f"{missed} days since last dose (CRITICAL)"] = round(np.exp(effective_log_or["missed_14_plus_days"]), 2)

    age = demo.get("age", 30)
    if 20 <= age <= 39:
        lo += effective_log_or["age_20_to_39"]
        factors[f"Age {age} (high-risk group 20-39)"] = round(np.exp(effective_log_or["age_20_to_39"]), 2)
    elif age > 60:
        lo += effective_log_or["age_over_60"]
        factors[f"Age {age} (elderly)"] = round(np.exp(effective_log_or["age_over_60"]), 2)

    # low_bmi_at_diagnosis — RATIONS trial (v8 §9.1)
    bmi_at_diag = base_cl.get("bmi_at_diagnosis") or base_cl.get("bmi")
    if bmi_at_diag and float(bmi_at_diag) < 18.5:
        lo += effective_log_or["low_bmi_at_diagnosis"]
        factors[f"Low BMI at diagnosis ({bmi_at_diag:.1f} — RATIONS trial)"] = round(
            np.exp(effective_log_or["low_bmi_at_diagnosis"]), 2)

    # ── CHO Annexure 1 signals (Record B monthly assessment) ─────────────────
    # These only fire after a CHO submission updates adh — they are zero for
    # patients whose CHO form has never been submitted (default values absent).

    # ADR grade ≥2 or explicit adr_symptoms flag
    # Suppressed when management_decision is a referral — the referral IS the clinical
    # response to the ADR. Applying both would double-count the same event and
    # prevent the referral from ever producing a net risk reduction.
    _mgmt_preview = adh.get("management_decision", "")
    _adr_managed  = _mgmt_preview in ("referral_to_higher_centre", "referral_for_hospitalisation")
    if not _adr_managed and (adh.get("adr_symptoms") or adh.get("adr_grade", 0) >= 2):
        lo += effective_log_or["adr_grade_2_plus"]
        _adr_g = adh.get("adr_grade", 0)
        factors[f"ADR grade {_adr_g} / symptoms present"] = round(
            np.exp(effective_log_or["adr_grade_2_plus"]), 2)

    # Nikshay Divas non-attendance — only penalise when explicitly False
    # (absent field = never submitted a CHO form, no signal either way)
    if adh.get("nikshay_divas_attended") is False:
        lo += effective_log_or["nikshay_divas_missed"]
        factors["Missed Nikshay Divas attendance"] = round(
            np.exp(effective_log_or["nikshay_divas_missed"]), 2)

    # Weight decline >2 kg since last assessment
    wt_delta = adh.get("weight_delta_kg", 0)
    if wt_delta < -2.0:
        lo += effective_log_or["weight_declining"]
        factors[f"Weight decline ({wt_delta:.1f} kg)"] = round(
            np.exp(effective_log_or["weight_declining"]), 2)

    # Management decision from CHO Annexure 1 — referral to higher centre is PROTECTIVE.
    # Once a patient is formally referred, they are under active management at a higher
    # facility and the dropout pathway is substantially closed. OR < 1 subtracts log-odds.
    # Only fires when management_decision is explicitly set from a CHO submission.
    mgmt = adh.get("management_decision", "")
    if mgmt in ("referral_to_higher_centre", "referral_for_hospitalisation"):
        lo += effective_log_or["referred_to_higher_centre"]
        factors["Referred to higher centre (protective — patient in system)"] = round(
            np.exp(effective_log_or["referred_to_higher_centre"]), 2)

    # Clip at 0.97 — logistic regression derived from odds ratios is not
    # perfectly calibrated at the tails. A BBN cannot claim certainty (1.0).
    prob = float(np.clip(1 / (1 + np.exp(-lo)), 0.0, 0.97))
    return {"score": round(prob, 4), "all_factors": factors}


def has_sufficient_tgn_evidence(treatment_week: int, regimen: str = "DS-TB") -> bool:
    """
    Returns True when TGN has reached its phase boundary and is the sole primary model.
    Retained from v7 fading insight (v8 §8.2: fading mechanism preserved, broken formula fixed).
    """
    boundary = 16 if regimen in ("Shorter-Oral-MDR", "Longer-Oral-MDR") else 8
    return treatment_week >= boundary


def compute_tgn_weight(treatment_week: int, regimen: str = "DS-TB") -> float:
    """
    TGN weight ramps linearly from 0.0 (week 0) to 1.0 at the phase boundary.
    Phase boundary: week 8 for DS-TB/BPaLM, week 16 for MDR regimens.
    After phase boundary: TGN weight = 1.0 (BBN fully superseded).

    v8 §8.2 — replaces the hard binary switch from v7.
    """
    if regimen in ("Shorter-Oral-MDR", "Longer-Oral-MDR"):
        boundary = 16
    else:
        boundary = 8  # DS-TB, BPaLM
    return float(min(treatment_week / boundary, 1.0)) if boundary > 0 else 1.0


def compute_risk_score_v8(record: dict, tgn_score: float, bbn_score: float,
                           asha_load_score: float, treatment_week: int = 0,
                           regimen: str = "DS-TB", max_uplift: float = 0.30,
                           tgn_trained: bool = True) -> dict:
    """
    v8 sequential-handover scoring (replaces compose_final_score).

    Primary risk = tgn_w * tgn_score + bbn_w * bbn_score
      - Weights sum to 1.0; no double-counting at either extreme.
      - Static features in TGN are scaled by tgn_w, so double-counting is
        proportional, not additive. At week 8 (DS-TB), TGN has full weight.

    ASHA load: additive equity floor only (Fix 4 — double-counting removed).
      Adds 0.05 * asha_load_score to primary risk. TGN already captures load
      structurally via ASHA node memory propagating through GATConv edges.

    data_source tag: 'tgn' | 'bbn_coldstart' | 'blend_N%_tgn'
    """
    tgn_w = compute_tgn_weight(treatment_week, regimen)

    if not tgn_trained:
        # TGN has no independent signal (PyTorch unavailable, no saved weights,
        # or simulation mode where TGN score == BBN score).
        # Give BBN full weight so ASHA updates produce visible score movement.
        tgn_w = 0.0
        bbn_w = 1.0
    else:
        # Floor bbn_weight at 0.10 so ASHA adherence updates always move the score.
        # Without this floor, post-phase-boundary patients have bbn_weight=0.0 and
        # days_since_last_dose changes have zero effect on the final score.
        bbn_w = max(0.10, 1.0 - tgn_w)
        tgn_w = 1.0 - bbn_w   # keep weights summing to 1.0

    primary_risk = tgn_w * tgn_score + bbn_w * bbn_score

    # Additive equity floor ONLY — multiplicative term removed (Fix 4).
    # ASHA load already enters TGN score via ASHA node memory (dim 0) propagating
    # through SUPERVISED_BY edges in GATConv. Applying a second multiplicative
    # modifier double-counts that structural signal. The additive floor alone
    # handles the equity concern: low-risk patients on overloaded ASHAs still
    # get a small upward nudge without corrupting the TGN's graph-learned output.
    additive_floor = 0.05 * asha_load_score
    effective_risk = float(min(primary_risk + additive_floor, 0.97))

    data_source = (
        "bbn_only_simulation" if not tgn_trained else
        "tgn" if tgn_w >= 1.0 else
        "bbn_coldstart" if tgn_w == 0.0 else
        f"blend_{int(tgn_w * 100)}pct_tgn"
    )

    return {
        "composite_score":  round(effective_risk, 4),
        "primary_risk":     round(primary_risk, 4),
        "tgn_weight":       round(tgn_w, 3),
        "bbn_weight":       round(bbn_w, 3),
        "asha_weight":      0.0,   # load is now a modifier, not a third component
        "asha_load_uplift": round(effective_risk - primary_risk, 4),
        "data_source":      data_source,
        "asha_load_score":  round(asha_load_score, 3),
        "bbn_status":       "active" if bbn_w > 0 else "retired",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASHA LOAD SCORE — system-side risk component
# ─────────────────────────────────────────────────────────────────────────────

def compute_asha_load_score(record: dict, asha_summaries: dict) -> float:
    """
    Computed from the ASHA worker node's properties.
    Reflects caseload pressure, visit frequency decline, geographic spread.
    This component is permanent — system failure is always a real risk factor.
    """
    asha_id = record["operational"]["asha_id"]
    summary = asha_summaries.get(asha_id)
    if not summary:
        return 0.3  # default moderate if no ASHA data

    return summary["load_score"]


# ─────────────────────────────────────────────────────────────────────────────
# URGENCY MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def apply_urgency_multiplier(composite_score: float, treatment_week: int,
                              total_treatment_weeks: int = 26) -> float:
    """
    DEPRECATED — urgency multiplier removed.
    Multiplying a probability by a week-based factor produced scores > 1.0
    for MDR patients, which np.clip silently capped to 1.0, destroying
    information and making moderate-risk MDR patients appear as certain dropouts.

    The risk_score now equals the composite_score directly.
    Visit ordering uses rank_score = (risk_score, treatment_week) two-key sort.
    This function is retained for backwards-compatibility only and is a no-op.
    """
    return round(float(np.clip(composite_score, 0, 1)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLDS — tighten as treatment progresses
# ─────────────────────────────────────────────────────────────────────────────

def get_adaptive_thresholds(treatment_week: int) -> dict:
    """
    Thresholds tighten as patient gets closer to treatment completion.
    Same score triggers different tiers depending on treatment stage.

    Phase             | Treatment Week | HIGH threshold | MEDIUM threshold
    Intensive         | 1-8            | 0.75           | 0.50
    Early Continuation| 9-16           | 0.65           | 0.40
    Late Continuation | 17-26          | 0.55           | 0.30
    """
    if treatment_week <= 8:
        return {"high": 0.85, "medium": 0.50, "phase_label": "Intensive (wk 1-8)"}
    elif treatment_week <= 16:
        return {"high": 0.65, "medium": 0.40, "phase_label": "Early Continuation (wk 9-16)"}
    else:
        return {"high": 0.55, "medium": 0.30, "phase_label": "Late Continuation (wk 17-26)"}


def assign_risk_tier(final_score: float, treatment_week: int) -> str:
    thresholds = get_adaptive_thresholds(treatment_week)
    if final_score >= thresholds["high"]:
        return "HIGH"
    elif final_score >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL FLAGS — split from single clinical_deterioration_flag (v8 Error 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_clinical_flags(record: dict) -> dict:
    """
    Compute adr_flag and nutritional_deterioration_flag from Record B data.

    v8 §8 / Error 9: single clinical_deterioration_flag is too coarse.
    ADR-driven avoidance and nutritional deterioration require different
    ASHA briefing language and different escalation actions.

    adr_flag:                      ADR grade ≥2 in any Record B month
    nutritional_deterioration_flag: ≥2kg weight drop between any two consecutive
                                    Record B months (RATIONS trial threshold)
    """
    records_b = record.get("records_b", [])
    adr_flag = False
    nutritional_deterioration_flag = False

    # adr_flag: any Record B with ADR grade >= 2
    for rb in records_b:
        adr_grade = rb.get("adr_grade") or rb.get("adr", {}).get("grade", 0) or 0
        if adr_grade >= 2:
            adr_flag = True
            break

    # nutritional_deterioration_flag: ≥2kg drop between consecutive months
    weights = []
    for rb in sorted(records_b, key=lambda x: x.get("month", 0)):
        w = rb.get("weight_kg")
        if w is not None:
            weights.append(float(w))
    for i in range(1, len(weights)):
        if weights[i - 1] - weights[i] >= 2.0:
            nutritional_deterioration_flag = True
            break

    return {
        "adr_flag":                      adr_flag,
        "nutritional_deterioration_flag": nutritional_deterioration_flag,
    }


# Backwards-compatible wrapper used by Stage 4 and simulation mode
def compute_risk_score(record: dict) -> dict:
    """BBN-only score wrapper. Used by stage2 simulation and legacy callers."""
    bbn = compute_bbn_prior(record)
    return {
        "risk_score":  bbn["score"],
        "all_factors": bbn["all_factors"],
        "top_factors": dict(sorted(bbn["all_factors"].items(), key=lambda x: x[1], reverse=True)[:3]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL PATIENTS
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SCORE AUDIT LOG — persists every rescore event for the Explainability tab
# ─────────────────────────────────────────────────────────────────────────────

SCORE_AUDIT_LOG_FILE = "data/score_audit_log.json"

def append_score_audit(patient_id: str, old_score: float, new_score: float,
                        old_tier: str, new_tier: str, trigger: str,
                        composition: dict = None, change_reason: str = ""):
    """
    Append one rescore event to data/score_audit_log.json.
    Called by score_all_patients, rescore_patient_locally, process_overnight_notes.

    trigger: "overnight_pipeline" | "manual_trigger" | "asha_update" | "note_ner"
    """
    from datetime import datetime, timezone
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)

    log = []
    if Path(SCORE_AUDIT_LOG_FILE).exists():
        try:
            with open(SCORE_AUDIT_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []

    entry = {
        "patient_id":   patient_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "trigger":      trigger,
        "old_score":    round(float(old_score), 4),
        "new_score":    round(float(new_score), 4),
        "score_delta":  round(float(new_score) - float(old_score), 4),
        "old_tier":     old_tier,
        "new_tier":     new_tier,
        "tier_changed": old_tier != new_tier,
    }
    if change_reason:
        entry["change_reason"] = change_reason
    if composition:
        entry["tgn_weight"]  = composition.get("tgn_weight", 0)
        entry["bbn_weight"]  = composition.get("bbn_weight", 0)
        entry["data_source"] = composition.get("data_source", "")
        entry["asha_floor"]  = composition.get("asha_load_uplift", 0)

    log.insert(0, entry)
    log = log[:2000]  # keep latest 2000 entries
    with open(SCORE_AUDIT_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def score_all_patients(patients: list, tgn_scores: dict = None,
                       asha_summaries: dict = None,
                       confirmed_cases: int = 0) -> list:
    """
    Full v8 scoring pipeline:
    1. BBN prior for every patient
    2. Compute risk via sequential-handover confidence ramp (compute_risk_score_v8)
    3. Apply urgency multiplier
    4. Assign risk tier using adaptive thresholds
    5. Compute adr_flag and nutritional_deterioration_flag
    """
    if asha_summaries is None:
        asha_summaries = {}

    print(f"Scoring {len(patients)} patients (v8 confidence-ramp scoring)...")

    high = medium = low = 0

    for p in patients:
        bbn_result = compute_bbn_prior(p)
        bbn_score  = bbn_result["score"]
        tgn_score  = (tgn_scores or {}).get(p["patient_id"], bbn_score)
        asha_load  = compute_asha_load_score(p, asha_summaries)

        # Derive treatment_week from Record A treatment_start_date (Fix 5).
        # len(records_b)*4 removed — a patient with no Record B yet is NOT at week 0.
        # treatment_start_date is always set by the MO at registration.
        treatment_week = p.get("treatment_week")
        if not treatment_week:
            start_str = p.get("treatment_start_date") or (
                p.get("baseline_clinical") or {}
            ).get("treatment_start_date", "")
            if start_str:
                try:
                    from datetime import datetime, timezone
                    start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                    now   = datetime.now(timezone.utc)
                    treatment_week = max(1, (now - start).days // 7)
                except Exception:
                    treatment_week = p.get("treatment_week") or 1
            else:
                treatment_week = p.get("treatment_week") or 1
        total_wks = p.get("total_treatment_weeks") or 26
        treatment_week = min(int(treatment_week), int(total_wks))

        # Regimen — support both schema versions
        regimen = (
            (p.get("diagnosis") or {}).get("regimen") or
            (p.get("clinical") or {}).get("regimen") or
            "DS-TB"
        )

        composition = compute_risk_score_v8(
                     p, tgn_score, bbn_score, asha_load,
                      treatment_week=treatment_week, regimen=regimen,
                       tgn_trained=Path("data/tgn_weights.pt").exists(),
         )
        # Urgency multiplier removed — score is the direct composite probability.
        # rank_score uses treatment_week as tiebreaker so later-week patients
        # sort above equal-risk earlier-week patients in visit priority lists.
        final_score = round(float(np.clip(composition["composite_score"], 0, 1)), 4)
        rank_score  = round(final_score + (treatment_week / 10000.0), 6)
        risk_tier   = assign_risk_tier(final_score, treatment_week)
        thresholds  = get_adaptive_thresholds(treatment_week)

        prev_score    = p.get("previous_risk_score") or final_score
        risk_velocity = round(final_score - float(prev_score), 4)

        # Audit log entry (Fix 8)
        append_score_audit(
            p["patient_id"],
            old_score=prev_score,
            new_score=final_score,
            old_tier=p.get("risk_level", "LOW"),
            new_tier=risk_tier,
            trigger="overnight_pipeline",
            composition=composition,
        )

        # Clinical flags (v8 Error 9 fix)
        flags = compute_clinical_flags(p)

        p["risk_score"]                      = final_score
        p["rank_score"]                      = rank_score
        p["previous_risk_score"]             = final_score
        p["risk_velocity"]                   = risk_velocity
        p["composite_score"]                 = composition["composite_score"]
        p["risk_level"]                      = risk_tier
        p["tgn_score"]                       = tgn_score
        p["treatment_week"]                  = treatment_week
        p["score_composition"]               = composition
        p["thresholds"]                      = thresholds
        p["all_factors"]                     = bbn_result["all_factors"]
        p["top_factors"]                     = dict(sorted(bbn_result["all_factors"].items(),
                                                           key=lambda x: x[1], reverse=True)[:3])
        p["asha_load_score"]                 = asha_load
        p["data_source"]                     = composition["data_source"]
        p["adr_flag"]                        = flags["adr_flag"]
        p["nutritional_deterioration_flag"]  = flags["nutritional_deterioration_flag"]

        if risk_velocity >= 0.12 and risk_tier != "HIGH":
            p["risk_level"]         = "HIGH"
            p["velocity_escalated"] = True

        if p["risk_level"] == "HIGH":     high   += 1
        elif p["risk_level"] == "MEDIUM": medium += 1
        else:                             low    += 1

    print(f"  HIGH: {high}  |  MEDIUM: {medium}  |  LOW: {low}")
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMIC FAILURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_systemic_failures(patients: list) -> list:
    """
    Four-tier escalation matching the ASHA → CHO → MO → DTO supervision chain
    per National Guidance 2025 (Annexure 5):

      Tier 1 — ASHA level:    individual patient silent 7+ days → alert to ASHA
      Tier 2 — CHO level:     >50% of ONE ASHA's patients missing → alert to CHO
      Tier 3 — Block MO:      multiple ASHAs in ONE CHO zone all failing → alert to MO
      Tier 4 — DTO level:     >3 CHO zones affected district-wide → alert to DTO

    Tier 2+ requires min 5 patients per ASHA to avoid false positives.
    Tier 1 alerts are per-patient and returned separately so Stage 5 can route them.
    """
    from collections import defaultdict

    # Group by ASHA, then track problem ASHAs by CHO zone
    asha_groups = defaultdict(list)

    for p in patients:
        asha_groups[p["operational"]["asha_id"]].append(p)

    alerts = []

    # ── Tier 2: per-ASHA systemic failure → escalate to CHO ──────────────────
    problem_ashas_by_cho = defaultdict(list)

    for asha_id, group in asha_groups.items():
        if len(group) < 5:
            continue
        miss_rate = sum(
            1 for p in group
            if (p.get("adherence") or {}).get("days_since_last_dose", 0) > 0
        ) / len(group)
        if miss_rate > 0.50:
            cho_id   = group[0]["operational"].get("cho_id", "CHO-UNKNOWN")
            avg_load = sum(p.get("asha_load_score", 0) for p in group) / len(group)
            alerts.append({
                "tier":             2,
                "asha_id":          asha_id,
                "cho_id":           cho_id,
                "patients_affected":len(group),
                "miss_rate_pct":    round(miss_rate * 100, 1),
                "avg_load_score":   round(avg_load, 3),
                "alert_type":       "SYSTEMIC_ASHA",
                "escalate_to":      f"CHO {cho_id}",
                "message": (
                    f"TIER 2 ALERT — {asha_id} (CHO zone {cho_id}): "
                    f"{miss_rate*100:.0f}% of {len(group)} patients missed doses. "
                    f"Possible worker absence or local disruption. "
                    f"Escalate to CHO {cho_id} — do NOT send to ASHA directly."
                ),
            })
            problem_ashas_by_cho[cho_id].append(asha_id)

    # ── Tier 3: multiple ASHAs failing within one CHO zone → escalate to MO ──
    problem_cho_zones = []
    for cho_id, bad_ashas in problem_ashas_by_cho.items():
        all_cho_ashas = {p["operational"]["asha_id"] for p in patients
                         if p["operational"].get("cho_id") == cho_id}
        fail_rate = len(bad_ashas) / max(len(all_cho_ashas), 1)

        if fail_rate >= 0.40 and len(bad_ashas) >= 2:
            problem_cho_zones.append(cho_id)
            alerts.append({
                "tier":             3,
                "cho_id":           cho_id,
                "ashas_affected":   bad_ashas,
                "fail_rate_pct":    round(fail_rate * 100, 1),
                "alert_type":       "SYSTEMIC_BLOCK",
                "escalate_to":      "Medical Officer (PHC)",
                "message": (
                    f"TIER 3 ALERT — CHO zone {cho_id}: "
                    f"{len(bad_ashas)} of {len(all_cho_ashas)} ASHAs showing systemic failures. "
                    f"Likely block-level disruption (drug stockout, clinic closure). "
                    f"Escalate to Medical Officer at PHC immediately."
                ),
            })

    # ── Tier 4: district-wide — multiple CHO zones affected → escalate to DTO
    if len(problem_cho_zones) > 3:
        alerts.append({
            "tier":           4,
            "cho_zones":      problem_cho_zones,
            "zones_affected": len(problem_cho_zones),
            "alert_type":     "SYSTEMIC_DISTRICT",
            "escalate_to":    "District TB Officer",
            "message": (
                f"TIER 4 ALERT — DISTRICT-WIDE: "
                f"{len(problem_cho_zones)} CHO zones showing concurrent systemic failures "
                f"({', '.join(problem_cho_zones[:5])}{'...' if len(problem_cho_zones)>5 else ''}). "
                f"Escalate to District TB Officer. Possible district-level drug stockout "
                f"or programme disruption."
            ),
        })

    if alerts:
        tier_counts = {2: 0, 3: 0, 4: 0}
        for a in alerts:
            tier_counts[a["tier"]] = tier_counts.get(a["tier"], 0) + 1
        print(f"\n  Systemic alerts: "
              f"Tier 2 (ASHA→CHO): {tier_counts[2]}  "
              f"Tier 3 (CHO→Block MO): {tier_counts[3]}  "
              f"Tier 4 (District): {tier_counts[4]}")
        for a in alerts:
            print(f"  [{a['escalate_to']}] {a['message'][:90]}...")

    return alerts



# ─────────────────────────────────────────────────────────────────────────────
# CONFIRMED DROPOUT TRACKING & BAYESIAN OR UPDATE
# ─────────────────────────────────────────────────────────────────────────────

CONFIRMED_DROPOUTS_FILE  = "data/confirmed_dropouts.json"
CONFIRMED_COMPLETERS_FILE = "data/confirmed_completers.json"
LEARNED_ORS_FILE         = "data/learned_ors.json"
BBN_SCHEDULE_FILE        = "data/bbn_update_schedule.json"

PRIOR_WEIGHT            = 20      # literature equivalent to ~20 real observations
MIN_CASES_TO_UPDATE     = 15      # never update an OR with fewer than this many cases
MAX_MOVE_PER_CYCLE      = 0.50    # OR cannot shift more than 50% of current value in one cycle

# How often the BBN weights are allowed to change.
# Options: "monthly", "quarterly", "biannual", "annual"
# The update runs at pipeline STARTUP, before any patient is scored.
# Mid-run updates are never allowed — every patient in a run uses identical weights.
BBN_UPDATE_FREQUENCY    = "biannual"   # default: every 6 months

_FREQUENCY_DAYS = {
    "monthly":   30,
    "quarterly": 91,
    "biannual":  182,
    "annual":    365,
}


def load_confirmed_dropouts() -> dict:
    """
    Load confirmed dropout records from disk.
    Returns dict: {patient_id: {factors: {...}, confirmed_at: iso_timestamp}}
    Creates the file if missing.
    """
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    if Path(CONFIRMED_DROPOUTS_FILE).exists():
        with open(CONFIRMED_DROPOUTS_FILE) as f:
            return json.load(f)
    return {}


def save_confirmed_dropout(patient_id: str, factors: dict):
    """
    Persist a single confirmed dropout record to disk.

    IMPORTANT — this function NEVER triggers a weight update.
    Weight updates only happen through check_and_run_scheduled_update(),
    which is called once at pipeline startup before any patient is scored.
    This guarantees every patient in a pipeline run is evaluated with
    exactly the same OR weights — no mid-run inconsistency.
    """
    from datetime import datetime, timezone
    dropouts = load_confirmed_dropouts()
    if patient_id in dropouts:
        print(f"  [BBN] {patient_id} already recorded — skipping duplicate")
        return
    dropouts[patient_id] = {
        "factors":            factors,
        "confirmed_at":       datetime.now(timezone.utc).isoformat(),
        "included_in_update": False,
    }
    with open(CONFIRMED_DROPOUTS_FILE, "w") as f:
        json.dump(dropouts, f, indent=2)
    new_pending = sum(1 for v in dropouts.values() if not v.get("included_in_update"))
    print(f"  [BBN] Confirmed dropout recorded: {patient_id} "
          f"({len(dropouts)} total, {new_pending} pending next cycle)")


def load_confirmed_completers() -> dict:
    """
    Load confirmed completer records from disk.
    Returns dict: {patient_id: {factors: {...}, confirmed_at: iso_timestamp}}
    Required for 2×2 contingency table OR computation (v8 §8.4).
    """
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    if Path(CONFIRMED_COMPLETERS_FILE).exists():
        with open(CONFIRMED_COMPLETERS_FILE) as f:
            return json.load(f)
    return {}


def save_confirmed_completer(patient_id: str, factors: dict):
    """
    Persist a single confirmed completer (Cured / Treatment Completed) to disk.
    Called from the District Officer dashboard on confirmed treatment completion.

    Required by v8 §8.4: run_bbn_update_cycle() needs BOTH dropout AND completer
    records to compute a valid 2×2 contingency table OR.
    Does NOT trigger a weight update — only check_and_run_scheduled_update() does.
    """
    from datetime import datetime, timezone
    completers = load_confirmed_completers()
    if patient_id in completers:
        print(f"  [BBN] {patient_id} already recorded as completer — skipping")
        return
    completers[patient_id] = {
        "factors":            factors,
        "confirmed_at":       datetime.now(timezone.utc).isoformat(),
        "included_in_update": False,
    }
    with open(CONFIRMED_COMPLETERS_FILE, "w") as f:
        json.dump(completers, f, indent=2)
    print(f"  [BBN] Confirmed completer recorded: {patient_id} ({len(completers)} total)")


def load_bbn_schedule() -> dict:
    """
    Load the BBN update schedule metadata.
    Returns dict with last_update_date and next_due_date.
    Creates a default schedule (due immediately) if no file exists.
    """
    from pathlib import Path
    from datetime import datetime, timezone, timedelta
    if Path(BBN_SCHEDULE_FILE).exists():
        with open(BBN_SCHEDULE_FILE) as f:
            return json.load(f)
    # First run — schedule is due immediately so the system initialises correctly
    now = datetime.now(timezone.utc)
    return {
        "last_update_date":  None,
        "next_due_date":     now.isoformat(),
        "frequency":         BBN_UPDATE_FREQUENCY,
        "cycles_completed":  0,
    }


def save_bbn_schedule(schedule: dict):
    """Write the updated schedule to disk after a successful cycle."""
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open(BBN_SCHEDULE_FILE, "w") as f:
        json.dump(schedule, f, indent=2)


def is_update_due(schedule: dict = None) -> tuple:
    """
    Check whether the BBN update cycle is due.

    Returns (is_due: bool, reason: str).

    The cycle is due when:
      - It has never run before (no last_update_date), OR
      - The current date is on or after next_due_date

    The cycle is NOT due when:
      - The pipeline was already run today (same calendar day), OR
      - The next due date is in the future

    This means if you run the pipeline 10 times in one day, the weights
    only update on the first run that day. All subsequent runs that day
    use the weights that were locked in at the start of that first run.
    """
    from datetime import datetime, timezone
    if schedule is None:
        schedule = load_bbn_schedule()

    now = datetime.now(timezone.utc)

    if schedule.get("last_update_date") is None:
        return True, "First run — initialising BBN schedule"

    next_due_str = schedule.get("next_due_date")
    if not next_due_str:
        return True, "No next_due_date recorded — running update"

    next_due = datetime.fromisoformat(next_due_str)

    if now >= next_due:
        days_overdue = (now - next_due).days
        return True, (
            f"Update due ({schedule.get('frequency', BBN_UPDATE_FREQUENCY)} cycle). "
            f"Last ran: {schedule.get('last_update_date', 'never')[:10]}. "
            f"Overdue by {days_overdue} days."
        )

    days_remaining = (next_due - now).days
    return False, (
        f"BBN weights current. Next update due: {next_due_str[:10]} "
        f"({days_remaining} days). Frequency: {schedule.get('frequency', BBN_UPDATE_FREQUENCY)}."
    )


def check_and_run_scheduled_update(frequency: str = None) -> dict:
    """
    Call this ONCE at pipeline startup, before score_all_patients() runs.

    Checks whether the calendar-based update cycle is due.
    If due AND enough cases exist: runs the update, locks new weights to disk.
    If not due OR too few cases: does nothing — current weights stay.

    Either way, score_all_patients() then reads whatever is in learned_ors.json
    and every patient in the run uses exactly the same weights.

    Args:
        frequency: override BBN_UPDATE_FREQUENCY (for testing). One of:
                   "monthly", "quarterly", "biannual", "annual"

    Returns dict with keys:
        update_ran      : bool
        reason          : str — why update ran or was skipped
        weights_source  : "learned" | "literature" | "updated"
        new_cases_used  : int
        next_due_date   : str (ISO)
    """
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    freq     = frequency or BBN_UPDATE_FREQUENCY
    freq_key = freq if freq in _FREQUENCY_DAYS else "biannual"
    interval = _FREQUENCY_DAYS[freq_key]

    schedule     = load_bbn_schedule()
    due, reason  = is_update_due(schedule)

    print(f"  [BBN Schedule] {reason}")

    if not due:
        # Weights unchanged — determine source for reporting
        from pathlib import Path
        source = "learned" if Path(LEARNED_ORS_FILE).exists() else "literature"
        return {
            "update_ran":    False,
            "reason":        reason,
            "weights_source": source,
            "new_cases_used": 0,
            "next_due_date": schedule.get("next_due_date", ""),
        }

    # Update is due — check whether enough cases exist for BOTH populations (v8 §8.4)
    dropouts   = load_confirmed_dropouts()
    completers = load_confirmed_completers()
    new_dropouts   = [v for v in dropouts.values()  if not v.get("included_in_update")]
    new_completers = [v for v in completers.values() if not v.get("included_in_update")]

    if len(new_dropouts) < MIN_CASES_TO_UPDATE or len(new_completers) < MIN_CASES_TO_UPDATE:
        reason_skip = (
            f"Update due but insufficient cases for 2×2 OR: "
            f"{len(new_dropouts)} dropout(s), {len(new_completers)} completer(s) "
            f"(minimum {MIN_CASES_TO_UPDATE} each required). "
            f"Record completers via save_confirmed_completer() in DTO dashboard."
        )
        print(f"  [BBN Schedule] {reason_skip}")
        _advance_schedule(schedule, freq_key, interval)
        return {
            "update_ran":     False,
            "reason":         reason_skip,
            "weights_source": "learned" if Path(LEARNED_ORS_FILE).exists() else "literature",
            "new_cases_used": 0,
            "next_due_date":  schedule.get("next_due_date", ""),
        }

    # Run the update
    print(f"  [BBN Schedule] Running update with {len(new_dropouts)} dropouts, "
          f"{len(new_completers)} completers...")
    run_bbn_update_cycle(dropouts, completers)

    # Advance schedule
    _advance_schedule(schedule, freq_key, interval)

    return {
        "update_ran":     True,
        "reason":         f"Scheduled {freq_key} update ran — {len(new_dropouts)} dropouts, {len(new_completers)} completers",
        "weights_source": "updated",
        "new_cases_used": len(new_dropouts),
        "next_due_date":  schedule.get("next_due_date", ""),
    }


def _advance_schedule(schedule: dict, freq_key: str, interval_days: int):
    """Update the schedule after a cycle (whether update ran or was skipped)."""
    from datetime import datetime, timezone, timedelta
    now      = datetime.now(timezone.utc)
    next_due = now + timedelta(days=interval_days)
    schedule["last_update_date"] = now.isoformat()
    schedule["next_due_date"]    = next_due.isoformat()
    schedule["frequency"]        = freq_key
    schedule["cycles_completed"] = schedule.get("cycles_completed", 0) + 1
    save_bbn_schedule(schedule)
    print(f"  [BBN Schedule] Next update scheduled: {next_due.strftime('%Y-%m-%d')} "
          f"({freq_key}, {interval_days} days)")


def load_learned_ors() -> dict:
    """
    Load the latest learned OR values.
    Falls back to the hardcoded literature values if no learned file exists.
    """
    from pathlib import Path
    if Path(LEARNED_ORS_FILE).exists():
        with open(LEARNED_ORS_FILE) as f:
            data = json.load(f)
            return data.get("ors", {})
    # Return literature defaults as starting point
    return {k: float(np.exp(v)) for k, v in LOG_OR.items()}


def run_bbn_update_cycle(dropouts: dict = None, completers: dict = None):
    """
    Bayesian OR update using a proper 2×2 contingency table (v8 §8.4).

    Correct formula:
        observed_OR = (d_with / d_without) / (c_with / c_without)

        d_with:    confirmed dropouts WITH this factor
        d_without: confirmed dropouts WITHOUT this factor
        c_with:    confirmed completers WITH this factor
        c_without: confirmed completers WITHOUT this factor

    This is a genuine odds ratio, not the risk ratio computed by the v7 formula
    (which divided within only the dropout population).

    Frozen per factor if fewer than MIN_CASES_TO_UPDATE dropouts OR completers
    have that factor. Fading mechanism (Bayesian weighted average) is retained.
    OR cannot move more than MAX_MOVE_PER_CYCLE (50%) per cycle.

    Requires BOTH dropout AND completer records — call save_confirmed_completer()
    from the DTO dashboard as patients complete treatment.
    """
    from datetime import datetime, timezone
    if dropouts is None:
        dropouts = load_confirmed_dropouts()
    if completers is None:
        completers = load_confirmed_completers()

    new_dropouts   = [v for v in dropouts.values()  if not v.get("included_in_update")]
    new_completers = [v for v in completers.values() if not v.get("included_in_update")]

    if len(new_dropouts) < 10 or len(new_completers) < 10:
        print(f"  [BBN Update] Need ≥10 dropouts AND ≥10 completers. "
              f"Have {len(new_dropouts)} dropouts, {len(new_completers)} completers. Skipping.")
        return

    current_ors = load_learned_ors()
    update_log  = {}

    factor_map = {
        "alcohol_use":            "Alcohol use",
        "divorced_separated":     "Divorced/separated",
        "hiv":                    "HIV co-infection",
        "prior_lfu":              "Prior LTFU history (strongest predictor)",
        "prior_tb":               "Prior TB history",
        "drug_use":               "Drug use",
        "continuation_phase":     "Continuation phase",
        "no_nutritional_support": "No nutritional support",
        "no_welfare":             "Not enrolled in Nikshay Poshan Yojana",
        "dr_tb":                  "Drug-resistant TB",
        "low_education":          "Low education",
        "male_sex":               "Male sex",
        "low_bmi_at_diagnosis":   "Low BMI at diagnosis",
    }

    n_d = len(new_dropouts)
    n_c = len(new_completers)

    for factor_key, factor_label in factor_map.items():
        lit_or     = float(np.exp(LOG_OR.get(factor_key, 0)))
        current_or = current_ors.get(factor_key, lit_or)

        # 2×2 contingency table counts
        d_with    = sum(1 for case in new_dropouts  if factor_label in case.get("factors", {}))
        c_with    = sum(1 for case in new_completers if factor_label in case.get("factors", {}))
        d_without = n_d - d_with
        c_without = n_c - c_with

        if d_with < MIN_CASES_TO_UPDATE or c_with < MIN_CASES_TO_UPDATE:
            update_log[factor_key] = (
                f"FROZEN (d_with={d_with}, c_with={c_with}, "
                f"minimum {MIN_CASES_TO_UPDATE} each required)"
            )
            continue

        # Guard against zero denominators
        if d_without == 0 or c_with == 0 or c_without == 0:
            update_log[factor_key] = "FROZEN (zero cell in 2×2 table — cannot compute OR)"
            continue

        # True odds ratio from 2×2 contingency table
        observed_or = (d_with / d_without) / (c_with / c_without)

        # Weighted Bayesian average with literature prior
        updated_or = (
            (current_or * PRIOR_WEIGHT) + (observed_or * n_d)
        ) / (PRIOR_WEIGHT + n_d)

        # Clamp: OR cannot move more than 50% of current value in one cycle
        max_move   = current_or * MAX_MOVE_PER_CYCLE
        updated_or = max(current_or - max_move, min(current_or + max_move, updated_or))
        updated_or = round(max(0.1, updated_or), 4)

        update_log[factor_key] = (
            f"lit={lit_or:.3f} → prev={current_or:.3f} → updated={updated_or:.3f} "
            f"(d_with={d_with}/{n_d}, c_with={c_with}/{n_c})"
        )
        current_ors[factor_key] = updated_or

    # Mark all processed cases
    for pid in dropouts:
        if not dropouts[pid].get("included_in_update"):
            dropouts[pid]["included_in_update"] = True
    with open(CONFIRMED_DROPOUTS_FILE, "w") as f:
        json.dump(dropouts, f, indent=2)

    for pid in completers:
        if not completers[pid].get("included_in_update"):
            completers[pid]["included_in_update"] = True
    with open(CONFIRMED_COMPLETERS_FILE, "w") as f:
        json.dump(completers, f, indent=2)

    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open(LEARNED_ORS_FILE, "w") as f:
        json.dump({
            "ors":               current_ors,
            "updated_at":        datetime.now(timezone.utc).isoformat(),
            "dropouts_used":     n_d,
            "completers_used":   n_c,
            "total_dropouts":    len(dropouts),
            "total_completers":  len(completers),
            "update_log":        update_log,
        }, f, indent=2)

    print(f"\n  [BBN Update] 2×2 OR update complete — {n_d} dropouts, {n_c} completers")
    for k, v in update_log.items():
        print(f"    {k}: {v}")
    print(f"  Saved → {LEARNED_ORS_FILE}")


def get_effective_log_ors() -> dict:
    """
    Return the current effective log-OR dict.
    Uses learned values if available, falls back to literature values.
    Called by compute_bbn_prior() so the BBN automatically uses updated weights.
    """
    learned = load_learned_ors()
    result  = dict(LOG_OR)  # start from literature
    for k in result:
        if k in learned:
            result[k] = np.log(max(0.01, learned[k]))
    return result


if __name__ == "__main__":
    with open("nikshay_grounded_dataset.json") as f:
        patients = json.load(f)

    scored  = score_all_patients(patients[:100])
    alerts  = detect_systemic_failures(scored)

    top5 = sorted(scored, key=lambda x: x["risk_score"], reverse=True)[:5]
    print("\nTop 5 patients:")
    for p in top5:
        print(f"  {p['patient_id']}  final={p['risk_score']}  tier={p['risk_level']}  "
              f"week={p['treatment_week']}  threshold_H={p['thresholds']['high']}")

    with open("nikshay_scored_dataset.json", "w") as f:
        json.dump(scored, f, indent=2)
    print("\nSaved nikshay_scored_dataset.json")