"""
dataset_gen_v2.py
=================
Generates synthetic NTEP-grounded patient data for Nikshay-Graph training.

Produces three separate output files:
  data/records_a.json          — 500 static patient profiles (Record A)
  data/records_b.json          — monthly CHO observations (Record B)
  data/records_c.json          — weekly ASHA field updates (Record C)
  data/trajectory_flat.csv     — denormalised table for XGBoost baseline

WHY THE OLD dataset_gen.py CANNOT BE USED WITH THE NEW ARCHITECTURE:

  32 of 47 fields had issues. Critical ones:
  1. No Record B or Record C — only a flat snapshot. The TGN needs event
     sequences, not snapshots.
  2. dose_history_30d is a 30-element binary array with no timestamps
     and no Δt. Useless for the temporal model.
  3. risk_score stored as a patient field. This leaks the target variable
     into the model input (Error 1 in the architecture doc).
  4. Phase randomly assigned (35% Intensive, 65% Continuation).
     Must be derived deterministically from treatment_week and regimen.
  5. Regimen names are Cat_I/Cat_II/DR_TB — not DS-TB/BPaLM/MDR per NTEP.
  6. prior_lfu_history conflated with prior_tb. These are different things.
  7. contact_network has vulnerability_score and has_comorbidity — neither
     is on any NTEP form. Contact screening outcomes on the contact node
     rather than in Record B (CHO duty).
  8. Dropout labels derived from BBN score threshold — circular.
     Correct: Weibull survival draw independent of the BBN.
  9. N=1000 patients. New architecture uses N=500 (Tondiarpet realistic).
 10. free_text_note, risk_velocity, previous_risk_score — computed outputs
     stored as inputs.

KEY DESIGN DECISIONS:

  - Dropout label: Weibull survival draw with shape=1.5, scale derived
    from the BBN LTFU probability. Independent of the BBN score — no
    circular training.
  - Weight trajectory: completers gain 0.2–0.8 kg/month (RATIONS trial).
    Dropouts lose 1.0–3.0 kg/month in the 4 weeks before dropout_week.
  - ADR events: ~15% of DR-TB/BPaLM patients, ~5% of DS-TB get grade ≥2
    ADR onset 2–4 weeks before dropout_week in a subset of dropouts.
  - Silence days: completers 0–2 days/week average. Dropouts show growing
    silence starting ~6 weeks before dropout_week.
  - Phase derived from treatment_week + regimen. Never randomly assigned.
  - triage_positive_at_diagnosis: ~12% of patients. Only these patients
    get haemoptysis_history, muac_cm, undernutrition_status (Annexure 2).

OUTPUTS ARE CONSISTENT: Record B weight_delta is computed from Record A
baseline_weight. Record C silence_days escalates toward dropout_week.
The event stream built from these records will show learnable patterns.

Usage:
    python dataset_gen_v2.py
    python dataset_gen_v2.py --n 200 --seed 123
"""

import json
import csv
import argparse
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from faker import Faker
    fake = Faker("en_IN")
    def random_name(): return fake.name()
except ImportError:
    _NAMES = [
        "Murugan Selvam", "Anitha Rajan", "Karthik Kumar", "Priya Devi",
        "Rajan Muthu", "Savitri Amma", "Suresh Babu", "Meena Devi",
        "Venkatesh Pillai", "Lakshmi N", "Senthil Kumar", "Kavitha Raj",
        "Balaji Sundaram", "Deepa Krishnan", "Manoj Prabhu", "Saranya V",
        "Gopal Iyer", "Usha Krishnan", "Dinesh Raj", "Ponni Muthu",
    ]
    import random as _r
    def random_name(): return _r.choice(_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DISTRICT = "Chennai"
BLOCK    = "Tondiarpet"
STATE    = "Tamil Nadu"
LANGUAGE = "Tamil"
PHC_ID   = "PHC_Tondiarpet"

N_PATIENTS    = 500
MAX_PER_ASHA  = 15     # realistic NTEP max caseload per ASHA
N_ASHAS       = -(-N_PATIENTS // MAX_PER_ASHA)  # ceiling division → 34
ASHA_IDS    = [f"ASHA-TN-{i+1:03d}" for i in range(N_ASHAS)]
# One CHO per 3 ASHAs (AAM catchment area covers ~3 ASHA zones)
N_CHOS      = max(1, N_ASHAS // 3)
CHO_IDS     = [f"CHO-TN-{i+1:02d}"  for i in range(N_CHOS)]
ASHA_TO_CHO = {a: CHO_IDS[min(i // 3, N_CHOS - 1)]
               for i, a in enumerate(ASHA_IDS)}

# Regimen names per DTC Guidelines (not Cat_I/Cat_II)
# ~80% DS-TB, ~10% BPaLM, ~6% Shorter-Oral-MDR, ~4% Longer-Oral-MDR
REGIMENS = (["DS-TB"] * 16 + ["BPaLM"] * 2 +
            ["Shorter-Oral-MDR"] * 1 + ["Longer-Oral-MDR"] * 1)

# Phase boundary: week at which Intensive phase ends
PHASE_BOUNDARY = {
    "DS-TB":           8,
    "BPaLM":           8,
    "Shorter-Oral-MDR": 16,
    "Longer-Oral-MDR":  32,
}

# Total treatment weeks per regimen
TREATMENT_WEEKS_TOTAL = {
    "DS-TB":           26,
    "BPaLM":           26,
    "Shorter-Oral-MDR": 48,
    "Longer-Oral-MDR":  88,
}

# Population prevalences
PREV = {
    "diabetes":             0.12,
    "hiv":                  0.03,
    "alcohol_use":          0.22,
    "tobacco_smoking":      0.30,
    "substance_use":        0.08,
    "low_education":        0.45,
    "bpl_status":           0.55,
    "npy_enrolled":         0.65,
    "npy_bank_tagged":      0.55,
    "triage_positive":      0.12,
}

# BBN log-ORs for LTFU probability (same as stage3_score — not used as label directly)
BASELINE_LTFU     = 0.062
BASELINE_LOG_ODDS = np.log(BASELINE_LTFU / (1 - BASELINE_LTFU))
LOG_OR = {
    "alcohol_use":        np.log(1.92),
    "divorced_separated": np.log(3.80),
    "diabetes":           np.log(0.52),
    "hiv":                np.log(2.16),
    "prior_lfu":          np.log(3.00),
    "low_education":      np.log(1.73),
    "substance_use":      np.log(2.40),
    "dr_tb":              np.log(2.80),
    "distance_over_10km": np.log(2.10),
    "distance_5_to_10km": np.log(1.60),
    "no_welfare_npy":     np.log(1.45),
    "continuation_phase": np.log(2.30),
    "age_20_to_39":       np.log(2.07),
    "age_over_60":        np.log(1.40),
    "low_bmi":            np.log(1.85),
}

RELATIONSHIP_TYPES = [
    "Household", "Household", "Household", "Household",
    "Household", "Household",
    "Workplace", "Workplace",
    "Social",
]

MARITAL_STATUSES = ["Married", "Married", "Married", "Single",
                    "Single", "Divorced", "Widowed"]

OCCUPATIONS = [
    "Wage earner", "Wage earner", "Wage earner",
    "Daily labourer", "Daily labourer",
    "Self-employed", "Unemployed", "Student",
]

EDUCATION_LEVELS = [
    "No formal education", "Primary", "Primary",
    "Secondary", "Secondary", "Higher secondary", "Graduate",
]

CASE_TYPES = [
    "Microbiologically Confirmed", "Microbiologically Confirmed",
    "Microbiologically Confirmed", "Clinically Diagnosed",
]

SITES = ["Pulmonary", "Pulmonary", "Pulmonary", "Extra-pulmonary"]

TREATMENT_HISTORY_TYPES = [
    "New", "New", "New", "New", "New",       # ~75% new
    "Retreatment-Recurrent",                   # ~10%
    "After-Failure",                           # ~8%
    "After-LFU",                               # ~7%
]

ADR_TYPES = [
    "peripheral_neuropathy", "nausea_vomiting", "hepatotoxicity",
    "skin_rash", "visual_disturbance", "joint_pain",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _asha(uid): return ASHA_IDS[min(uid // MAX_PER_ASHA, N_ASHAS - 1)]
def _cho(aid):  return ASHA_TO_CHO.get(aid, CHO_IDS[0])


def get_phase(regimen: str, treatment_week: int) -> str:
    """Phase derived deterministically from treatment_week and regimen."""
    boundary = PHASE_BOUNDARY.get(regimen, 8)
    return "Intensive" if treatment_week <= boundary else "Continuation"


def compute_ltfu_prob(p: dict) -> float:
    """
    Compute LTFU probability from static risk factors.
    Used ONLY for Weibull survival draw — never stored as risk_score input.
    """
    lo = BASELINE_LOG_ODDS
    if p["social"]["alcohol_use"]:                         lo += LOG_OR["alcohol_use"]
    if p["demographics"]["marital_status"] == "Divorced":  lo += LOG_OR["divorced_separated"]
    if p["baseline_clinical"]["diabetes"]:                 lo += LOG_OR["diabetes"]
    if p["baseline_clinical"]["hiv"]:                      lo += LOG_OR["hiv"]
    if p["prior_lfu_history"]:                             lo += LOG_OR["prior_lfu"]
    if p["social"]["low_education"]:                       lo += LOG_OR["low_education"]
    if p["social"]["substance_use_disorder"]:              lo += LOG_OR["substance_use"]
    if p["diagnosis"]["regimen"] in ("BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR"):
        lo += LOG_OR["dr_tb"]
    dist = p["social"]["distance_to_phc_km"]
    if dist >= 10:   lo += LOG_OR["distance_over_10km"]
    elif dist >= 5:  lo += LOG_OR["distance_5_to_10km"]
    if not p["welfare"]["npy_enrolled"]:                   lo += LOG_OR["no_welfare_npy"]
    age = p["demographics"]["age"]
    if 20 <= age <= 39: lo += LOG_OR["age_20_to_39"]
    elif age > 60:      lo += LOG_OR["age_over_60"]
    bmi = p["baseline_clinical"]["bmi"]
    if 0 < bmi < 18.5:  lo += LOG_OR["low_bmi"]
    return float(np.clip(1 / (1 + np.exp(-lo)), 0.0, 1.0))


def weibull_dropout_draw(ltfu_prob: float, total_weeks: int,
                          rng: np.random.Generator) -> tuple:
    """
    Draw a binary dropout label and dropout week from a Weibull survival model.

    The Weibull shape=1.5 produces an increasing hazard — risk of dropout
    accelerates over time, which matches clinical reality (continuation phase
    has higher LTFU than intensive phase).

    Scale parameter is calibrated to the patient's LTFU probability so that
    high-risk patients are more likely to drop out AND tend to drop out earlier.

    Returns (dropout_label: int, dropout_week: int or None)
    """
    scale = max(0.5, -total_weeks / np.log(max(1 - ltfu_prob, 1e-6)))
    shape = 0.8  # reduced from 1.2; targets ~8-12% observed dropout rate at N=500

    dropout_label = 0
    dropout_week  = None

    for week in range(1, total_weeks + 1):
        # Weibull hazard: h(t) = (shape/scale) * (t/scale)^(shape-1)
        hazard = (shape / scale) * (week / scale) ** (shape - 1)
        prob_this_week = 1 - np.exp(-hazard)
        if rng.random() < prob_this_week:
            dropout_label = 1
            dropout_week  = week
            break

    return dropout_label, dropout_week


def generate_weight_trajectory(baseline_weight: float, n_months: int,
                                dropout_label: int, dropout_week: int,
                                rng: np.random.Generator) -> list:
    """
    Generate monthly weight values consistent with dropout label.

    Per RATIONS trial (DTC §1.2.1(d)):
      Completers: gain 0.2–0.8 kg/month throughout treatment
      Dropouts:   gain weight early, then lose 1.0–3.0 kg/month in the
                  4 weeks before dropout_week

    Returns list of n_months weight values (kg), not including baseline.
    """
    weights  = []
    current  = baseline_weight
    dropout_month = (dropout_week // 4) if dropout_week else None

    for month in range(1, n_months + 1):
        if (dropout_label == 1 and dropout_month is not None and
                month >= dropout_month - 1):
            # Weight drop preceding dropout
            delta = rng.uniform(-3.0, -1.0)
        else:
            # Normal treatment: gaining weight
            delta = rng.uniform(0.2, 0.8)
        current = round(max(30.0, current + delta), 1)
        weights.append(current)

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# RECORD A GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_record_a(uid: int, rng: np.random.Generator,
                      base_date: datetime) -> dict:
    """
    Generate one Record A — static patient profile.
    Collected by Medical Officer at diagnosis. Never changes.
    """
    asha_id = _asha(uid)
    cho_id  = _cho(asha_id)

    age        = int(rng.integers(18, 75))
    sex        = str(rng.choice(["Male", "Female"], p=[0.65, 0.35]))
    marital    = str(rng.choice(MARITAL_STATUSES))
    regimen    = str(rng.choice(REGIMENS))
    treat_hist = str(rng.choice(TREATMENT_HISTORY_TYPES))
    total_wks  = int(TREATMENT_WEEKS_TOTAL[regimen])

    # Treatment week: where in the course is the patient right now
    treatment_week = int(rng.integers(1, total_wks + 1))

    # Diagnosis date: treatment_week weeks before base_date
    diag_date  = base_date - timedelta(weeks=treatment_week)
    start_date = diag_date + timedelta(days=int(rng.integers(1, 5)))

    # Demographics
    diabetes  = bool(rng.random() < PREV["diabetes"])
    hiv       = bool(rng.random() < PREV["hiv"])
    alcohol   = bool(rng.random() < PREV["alcohol_use"])
    tobacco   = bool(rng.random() < PREV["tobacco_smoking"])
    substance = bool(rng.random() < PREV["substance_use"])
    low_edu   = bool(rng.random() < PREV["low_education"])
    bpl       = bool(rng.random() < PREV["bpl_status"])
    npy       = bool(rng.random() < PREV["npy_enrolled"])
    npy_bank  = bool(npy and rng.random() < 0.85)
    dist      = float(round(float(rng.gamma(2, 2.5)), 1))

    # prior_lfu is only True when treatment_history_type is After-LFU
    prior_lfu = (treat_hist == "After-LFU")
    # prior_tb is True for any retreatment type
    prior_tb  = treat_hist != "New"

    # Baseline clinical
    height_cm   = float(round(rng.normal(162, 8), 1))
    weight_kg   = float(round(rng.normal(52, 9), 1))
    weight_kg   = max(30.0, weight_kg)
    bmi         = round(weight_kg / ((height_cm / 100) ** 2), 1)
    rbs_mgdl    = int(rng.integers(80, 200)) if diabetes else int(rng.integers(70, 110))
    hb_gdl      = round(float(rng.normal(10.5, 2.0)), 1)
    gen_cond    = str(rng.choice(["ambulatory", "ambulatory", "ambulatory", "bedridden"]))

    # Triage
    triage_pos  = bool(rng.random() < PREV["triage_positive"])

    # Contacts — identity only (screening outcomes in Record B)
    n_contacts = int(rng.integers(1, 5))
    contacts   = []
    for _ in range(n_contacts):
        c_age = int(rng.integers(5, 75))
        rel   = str(rng.choice(RELATIONSHIP_TYPES))
        contacts.append({
            "name":              random_name(),
            "age":               c_age,
            "relationship_type": rel,
        })

    record_a = {
        "nikshay_id":           f"NIK-{100001 + uid}",
        "district":             DISTRICT,
        "phc_id":               PHC_ID,
        "treatment_start_date": start_date.strftime("%Y-%m-%d"),
        "date_of_diagnosis":    diag_date.strftime("%Y-%m-%d"),
        "treatment_week":       treatment_week,
        "total_treatment_weeks": total_wks,

        "demographics": {
            "age":            age,
            "sex":            sex,
            "marital_status": marital,
            "education_level": str(rng.choice(EDUCATION_LEVELS)),
            "occupation":     str(rng.choice(OCCUPATIONS)),
            "bpl_status":     bpl,
        },

        "diagnosis": {
            "case_type":             str(rng.choice(CASE_TYPES)),
            "site":                  str(rng.choice(SITES)),
            "treatment_history_type": treat_hist,
            "regimen":               regimen,
        },

        "baseline_clinical": {
            "weight_kg":        weight_kg,   # CRITICAL: baseline for weight delta
            "height_cm":        height_cm,
            "bmi":              bmi,
            "hiv":              hiv,
            "hiv_art_status":   str(rng.choice(["on_art", "not_on_art"])) if hiv else None,
            "diabetes":         diabetes,
            "rbs_mgdl":         rbs_mgdl,
            "hb_gdl":           max(6.0, round(hb_gdl, 1)),
            "general_condition": gen_cond,
        },

        "social": {
            "alcohol_use":            alcohol,
            "tobacco_smoking":        tobacco,
            "substance_use_disorder": substance,
            "distance_to_phc_km":     dist,
            "low_education":          low_edu,
            "prior_tb_history":       prior_tb,
        },

        "welfare": {
            "npy_enrolled":            npy,
            "npy_bank_account_tagged": npy_bank,
        },

        # NTEP-grounded flags
        "prior_lfu_history":           prior_lfu,
        "triage_positive_at_diagnosis": triage_pos,

        # Contact identity only — screening outcomes live in Record B
        "contact_network": contacts,

        # Operational
        "operational": {
            "asha_id":   asha_id,
            "cho_id":    cho_id,
            "language":  LANGUAGE,
            "phone_type": str(rng.choice(["Smartphone", "Basic", "None"],
                                          p=[0.55, 0.38, 0.07])),
        },

        # Triage-positive only fields (Annexure 2, nodal physician)
        # present only when triage_positive_at_diagnosis = true
        **({"triage_positive_fields": {
            "haemoptysis_history":  bool(rng.random() < 0.25),
            "muac_cm":              float(round(rng.normal(22, 3), 1)) if gen_cond == "bedridden" else None,
            "undernutrition_status": str(rng.choice(["None", "Severe", "Very Severe"],
                                                     p=[0.50, 0.35, 0.15])),
        }} if triage_pos else {}),
    }

    return record_a


# ─────────────────────────────────────────────────────────────────────────────
# RECORD B GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_records_b(record_a: dict, dropout_label: int,
                        dropout_week: int, weight_trajectory: list,
                        rng: np.random.Generator) -> list:
    """
    Generate monthly CHO observations (Record B).
    One per month from month 1 up to current treatment_week // 4,
    or up to the month of dropout for dropouts.
    """
    treatment_week = record_a["treatment_week"]
    regimen        = record_a["diagnosis"]["regimen"]
    baseline_wt    = record_a["baseline_clinical"]["weight_kg"]
    baseline_bmi   = record_a["baseline_clinical"]["bmi"]
    height_cm      = record_a["baseline_clinical"]["height_cm"]
    start_date_str = record_a["treatment_start_date"]
    start_date     = datetime.strptime(start_date_str, "%Y-%m-%d")
    contacts       = record_a["contact_network"]

    dropout_month  = (dropout_week // 4) if (dropout_label and dropout_week) else None
    n_months       = treatment_week // 4
    if dropout_month:
        n_months = min(n_months, dropout_month)
    n_months = max(1, min(n_months, 6))  # Record B: up to 6 months

    # ADR: assign to a subset of patients
    dr_tb_patient = regimen in ("BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR")
    adr_prob      = 0.15 if dr_tb_patient else 0.05
    has_adr       = bool(rng.random() < adr_prob)
    adr_onset_month = int(rng.integers(1, n_months + 1)) if has_adr else None

    gen_cond    = record_a["baseline_clinical"]["general_condition"]
    records_b   = []
    prev_weight = baseline_wt

    for month in range(1, n_months + 1):
        assess_date = start_date + timedelta(days=month * 30)
        phase = get_phase(regimen, month * 4)

        # Weight
        current_weight = weight_trajectory[month - 1] if month - 1 < len(weight_trajectory) \
                         else prev_weight
        weight_delta   = round(current_weight - prev_weight, 1)
        bmi_now        = round(current_weight / ((height_cm / 100) ** 2), 1)
        prev_weight    = current_weight

        # SpO2 — lower for triage-positive or DR-TB
        is_triage_pos = record_a["triage_positive_at_diagnosis"]
        spo2 = int(rng.integers(88, 94)) if (is_triage_pos and rng.random() < 0.3) \
               else int(rng.integers(95, 100))

        # Vitals
        rr   = int(rng.integers(18, 28))
        bp_s = int(rng.integers(100, 140))
        bp_d = int(rng.integers(60, 90))
        puls = int(rng.integers(60, 110))
        temp = round(float(rng.normal(37.0, 0.4)), 1)

        # Red flags — CHO formal assessment
        # Deteriorating patients more likely to have red flags
        deteriorating = (dropout_label == 1 and dropout_month and month >= dropout_month - 1)
        rf_base_prob  = 0.15 if deteriorating else 0.03
        red_flags = {
            "confined_to_bed":      bool(rng.random() < rf_base_prob * 0.3),
            "breathlessness":       bool(rng.random() < rf_base_prob),
            "severe_pain_chest_or_abdomen": bool(rng.random() < rf_base_prob * 0.5),
            "altered_consciousness": bool(rng.random() < rf_base_prob * 0.2),
            "haemoptysis_one_cup":  bool(rng.random() < rf_base_prob * 0.2),
            "recurrent_vomiting_diarrhoea": bool(rng.random() < rf_base_prob * 0.4),
            "adr_symptoms":         bool(has_adr and adr_onset_month and month >= adr_onset_month),
        }
        any_rf       = any(red_flags.values())
        mo_assessed  = bool(any_rf and rng.random() < 0.65)
        red_flags["any_red_flag_positive"] = any_rf
        red_flags["mo_assessment_done"]    = mo_assessed

        # ADR detail
        adr_this_month = has_adr and adr_onset_month and month >= adr_onset_month
        adr = {}
        if adr_this_month:
            grade = int(rng.choice([2, 3, 4], p=[0.60, 0.30, 0.10]))
            adr = {
                "reported": True,
                "type":     str(rng.choice(ADR_TYPES)),
                "grade":    grade,
                "resolved": bool(rng.random() < (0.7 if grade == 2 else 0.3)),
                "action":   str(rng.choice(["counselling", "dose_modification",
                                             "referral_to_mo", "regimen_change"],
                                           p=[0.4, 0.3, 0.2, 0.1])),
            }
        else:
            adr = {"reported": False, "type": None, "grade": 0, "resolved": None, "action": None}


        # Fix 6: deterministic severity_classification from Box 4.2 thresholds
        _jaund = bool(adr.get("type") == "hepatotoxicity")
        severity_classification = "severe" if (
            spo2 < 94 or rr > 24 or
            bp_s < 90 or bp_s >= 140 or
            bp_d < 60 or bp_d >= 90 or
            puls > 120 or puls < 60 or
            bmi_now < 14 or _jaund
        ) else "non_severe"
        _mgmt = (
            "referral_to_hospital" if severity_classification == "severe"
            else "referral_to_phc" if any_rf
            else "ambulatory_care"
        )

        # Programme
        dropout_leading = (dropout_label == 1 and dropout_month and month >= dropout_month - 2)
        nikshay_divas   = bool(rng.random() < (0.30 if dropout_leading else 0.75))
        npy_received    = bool(record_a["welfare"]["npy_enrolled"] and rng.random() < 0.80)

        # Contact screening (CHO duty — not ASHA duty)
        contact_screening = []
        for c in contacts:
            if rng.random() < 0.35:  # ~35% of contacts screened each month
                sym_pos = bool(rng.random() < 0.12)
                contact_screening.append({
                    "contact_name":        c["name"],
                    "screened_this_month": True,
                    "screening_method":    str(rng.choice(["symptom_only",
                                                           "symptom_and_cxr",
                                                           "symptom_and_cbnaat"],
                                                          p=[0.6, 0.25, 0.15])),
                    "symptom_positive":    sym_pos,
                    "tpt_eligible":        bool(not sym_pos and rng.random() < 0.40),
                    "referred_for_diagnosis": bool(sym_pos and rng.random() < 0.80),
                })

        record_b = {
            "nikshay_id":      record_a["nikshay_id"],
            "month":           month,
            "assessment_date": assess_date.strftime("%Y-%m-%d"),
            "treatment_week":  month * 4,
            "phase":           phase,

            "red_flags": red_flags,

            "vitals": {
                "weight_kg":          current_weight,
                "bmi":                bmi_now,
                "weight_delta_kg":    weight_delta,
                "spo2_percent":       spo2,
                "spo2_flag":          spo2 < 94,
                "respiratory_rate":   rr,
                "respiratory_rate_flag": rr > 24,
                "bp_systolic":        bp_s,
                "bp_diastolic":       bp_d,
                "bp_flag":            (bp_s < 90 or bp_s >= 140 or
                                       bp_d < 60 or bp_d >= 90),
                "pulse_rate":         puls,
                "pulse_flag":         (puls > 120 or puls < 60),
                "temperature_celsius": temp,
            },

            "clinical_signs": {
                "unable_to_stand_without_support": bool(rng.random() < 0.02),
                "pedal_oedema":  bool(rng.random() < 0.04),
                "icterus":       bool(adr.get("type") == "hepatotoxicity"),
                "haemoptysis":   red_flags["haemoptysis_one_cup"],
                "general_condition": gen_cond if (deteriorating and
                                     rng.random() < 0.3) else "ambulatory",
            },

            "investigations": {
                "hb_gdl":         None,
                "rbs_mgdl":       None,
                "chest_xray_done": bool(month == 2 or (any_rf and rng.random() < 0.5)),
                "lft_done":       bool(adr_this_month and rng.random() < 0.7),
                "kft_done":       bool(adr_this_month and rng.random() < 0.3),
            },

            "adr": adr,

            "programme": {
                "nikshay_divas_attended": nikshay_divas,
                "npy_benefit_received":   npy_received,
                "management_decision":     _mgmt,
                "severity_classification": severity_classification,
            },

            "contact_screening": contact_screening,
        }

        records_b.append(record_b)

    return records_b


# ─────────────────────────────────────────────────────────────────────────────
# RECORD C GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_records_c(record_a: dict, dropout_label: int,
                        dropout_week: int, rng: np.random.Generator,
                        records_b: list = None) -> list:
    """
    Generate weekly ASHA field updates (Record C).
    One per week from week 1 up to current treatment_week,
    or up to dropout_week for dropouts.
    """
    treatment_week = record_a["treatment_week"]
    regimen        = record_a["diagnosis"]["regimen"]
    asha_id        = record_a["operational"]["asha_id"]
    start_date_str = record_a["treatment_start_date"]
    start_date     = datetime.strptime(start_date_str, "%Y-%m-%d")

    n_weeks = treatment_week
    if dropout_label and dropout_week:
        n_weeks = min(n_weeks, dropout_week)
    n_weeks = max(1, n_weeks)

    b_by_month = {}
    for rb in (records_b or []):
        b_by_month[rb.get("month", 0)] = rb

    records_c = []
    cumulative_silence = 0

    for week in range(1, n_weeks + 1):
        visit_date = start_date + timedelta(weeks=week)
        phase      = get_phase(regimen, week)

        # Dropout trajectory: silence builds in the 6 weeks before dropout
        weeks_to_dropout = (dropout_week - week) if (dropout_label and dropout_week) else 999
        dropout_imminent = (dropout_label == 1 and weeks_to_dropout <= 6)

        # Dose status
        if dropout_imminent:
            dose_status = str(rng.choice(
                ["missed", "missed", "confirmed", "not_a_dose_day"],
                p=[0.55, 0.0, 0.30, 0.15]   # weights adjusted below
            ))
            # Recalculate with proper probabilities
            r = rng.random()
            if r < 0.55:   dose_status = "missed"
            elif r < 0.85: dose_status = "not_a_dose_day"
            else:          dose_status = "confirmed"
        else:
            r = rng.random()
            if r < 0.85:   dose_status = "confirmed"
            elif r < 0.93: dose_status = "not_a_dose_day"
            else:          dose_status = "missed"

        # Silence days: completers average 0–2/week; dropouts escalate
        if dropout_imminent:
            base_silence = 6 - weeks_to_dropout  # 0→6 as countdown goes 6→0
            silence_days = int(min(7, max(0, base_silence + int(rng.integers(0, 3)))))
        else:
            silence_days = int(rng.choice([0, 0, 0, 1, 2, 3], p=[0.50, 0.20, 0.15, 0.07, 0.05, 0.03]))

        cumulative_silence = silence_days

        # Visit happened or not
        visited = silence_days == 0 and dose_status != "not_a_dose_day"

        # Unable to visit reason
        unable_reason = None
        if not visited:
            if silence_days > 0:
                if dropout_imminent:
                    probs = [0.40, 0.35, 0.15, 0.10]
                else:
                    probs = [0.50, 0.15, 0.25, 0.10]
                unable_reason = str(rng.choice(
                    ["patient_absent", "patient_refused", "asha_unavailable", "other"],
                    p=probs
                ))

                # Fix 7: Record C red flags correlated with corresponding Record B month.
        # If CHO recorded a flag in Record B for the same time window,
        # ASHA has 0.80 base probability of reporting the same (not independent).
        rf_base = 0.10 if dropout_imminent else 0.02
        corresponding_month = max(1, week // 4)
        rb_flags = b_by_month.get(corresponding_month, {}).get("red_flags", {})

        def _cf(key, mult=1.0):
            return bool(rng.random() < (0.80 if rb_flags.get(key, False) else rf_base * mult))

        red_flags = {
            "confined_to_bed":              _cf("confined_to_bed", 0.3),
            "breathlessness":               _cf("breathlessness", 1.0),
            "severe_pain_chest_or_abdomen": _cf("severe_pain_chest_or_abdomen", 0.5),
            "altered_consciousness":        _cf("altered_consciousness", 0.2),
            "haemoptysis_one_cup":          _cf("haemoptysis_one_cup", 0.2),
            "recurrent_vomiting_diarrhoea": _cf("recurrent_vomiting_diarrhoea", 0.4),
            "adr_symptoms":                 bool(rng.random() < (0.08 if dropout_imminent else 0.01)),
        }
        any_rf      = any(red_flags.values())
        mo_assessed = bool(any_rf and rng.random() < 0.55)
        red_flags["any_red_flag_identified"] = any_rf
        red_flags["mo_assessment_done"]      = mo_assessed

        # Patient flags
        expressed_reluctance = bool(
            dropout_imminent and rng.random() < 0.15
        )
        welfare_issue = bool(
            not record_a["welfare"]["npy_enrolled"] and rng.random() < 0.05
        )

        record_c = {
            "nikshay_id":            record_a["nikshay_id"],
            "week":                  week,
            "visit_date":            visit_date.strftime("%Y-%m-%d"),
            "asha_id":               asha_id,
            "dose_status":           dose_status,
            "visited":               visited,
            "silence_days":          silence_days,
            "unable_to_visit_reason": unable_reason,
            "red_flags":             red_flags,
            "patient_flags": {
                "expressed_reluctance": expressed_reluctance,
                "welfare_issue_raised": welfare_issue,
            },
        }

        records_c.append(record_c)

    return records_c


# ─────────────────────────────────────────────────────────────────────────────
# FLAT CSV FOR XGBOOST BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def flatten_to_csv_row(record_a: dict, records_b: list,
                        records_c: list) -> dict:
    """Flatten all three records into a single row for XGBoost baseline."""
    last_b = records_b[-1] if records_b else {}
    last_c = records_c[-1] if records_c else {}

    # Aggregate Record C
    n_missed    = sum(1 for r in records_c if r["dose_status"] == "missed")
    n_confirmed = sum(1 for r in records_c if r["dose_status"] == "confirmed")
    n_weeks     = len(records_c)
    adherence_rate = round(n_confirmed / max(n_weeks, 1), 3)
    max_silence = max((r["silence_days"] for r in records_c), default=0)
    any_reluctance = any(r["patient_flags"]["expressed_reluctance"] for r in records_c)

    # Aggregate Record B
    min_weight = min((r["vitals"]["weight_kg"] for r in records_b), default=None)
    max_weight_drop = 0.0
    for i in range(1, len(records_b)):
        drop = records_b[i - 1]["vitals"]["weight_kg"] - records_b[i]["vitals"]["weight_kg"]
        if drop > max_weight_drop:
            max_weight_drop = drop
    any_adr = any(r["adr"]["grade"] >= 2 for r in records_b)
    missed_nikshay_divas = sum(1 for r in records_b if not r["programme"]["nikshay_divas_attended"])

    dem = record_a["demographics"]
    soc = record_a["social"]
    bl  = record_a["baseline_clinical"]
    diag = record_a["diagnosis"]

    return {
        "nikshay_id":            record_a["nikshay_id"],
        "dropout_label":         record_a.get("dropout_label", 0),
        "dropout_week":          record_a.get("dropout_week"),
        "treatment_week":        record_a["treatment_week"],
        "age":                   dem["age"],
        "sex_male":              int(dem["sex"] == "Male"),
        "marital_divorced":      int(dem["marital_status"] in ("Divorced", "Separated")),
        "low_education":         int(soc["low_education"]),
        "bpl_status":            int(dem["bpl_status"]),
        "distance_km":           soc["distance_to_phc_km"],
        "alcohol_use":           int(soc["alcohol_use"]),
        "substance_use":         int(soc["substance_use_disorder"]),
        "hiv":                   int(bl["hiv"]),
        "diabetes":              int(bl["diabetes"]),
        "bmi_at_diagnosis":      bl["bmi"],
        "weight_kg_baseline":    bl["weight_kg"],
        "prior_lfu":             int(record_a["prior_lfu_history"]),
        "dr_tb":                 int(diag["regimen"] in ("BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR")),
        "npy_enrolled":          int(record_a["welfare"]["npy_enrolled"]),
        "triage_positive":       int(record_a["triage_positive_at_diagnosis"]),
        "adherence_rate":        adherence_rate,
        "n_missed_doses":        n_missed,
        "max_silence_days":      max_silence,
        "any_expressed_reluctance": int(any_reluctance),
        "max_weight_drop_kg":    round(max_weight_drop, 2),
        "any_adr_grade2_plus":   int(any_adr),
        "n_missed_nikshay_divas": missed_nikshay_divas,
        "last_silence_days":     last_c.get("silence_days", 0),
        "last_dose_missed":      int(last_c.get("dose_status") == "missed"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_save(n: int = N_PATIENTS, seed: int = 42,
                      output_dir: str = "data") -> dict:
    """
    Generate n patients with Records A, B, C.
    Returns dict with all three record lists and summary statistics.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 3, 1, tzinfo=timezone.utc)

    print(f"\nGenerating {n} patients — {BLOCK}, {DISTRICT}")
    print(f"  ASHAs: {N_ASHAS} (max {MAX_PER_ASHA} each)  CHOs: {N_CHOS}  PHC: {PHC_ID}")

    all_a     = []
    all_b     = []
    all_c     = []
    flat_rows = []

    dropout_total = 0
    label_dist    = {0: 0, 1: 0}

    for uid in range(n):
        # Generate Record A (static profile)
        rec_a = generate_record_a(uid, rng, base_date)

        # Compute LTFU probability for Weibull draw
        ltfu_prob = compute_ltfu_prob(rec_a)
        total_wks = rec_a["total_treatment_weeks"]

        # Draw dropout label independently of BBN score (no circular training)
        dropout_label, dropout_week = weibull_dropout_draw(
            ltfu_prob, total_wks, rng
        )

        # Add label to Record A
        rec_a["dropout_label"] = dropout_label
        rec_a["dropout_week"]  = dropout_week

        # Generate weight trajectory consistent with label
        n_months = min(rec_a["treatment_week"] // 4 + 1, 6)
        weight_traj = generate_weight_trajectory(
            rec_a["baseline_clinical"]["weight_kg"],
            n_months, dropout_label, dropout_week, rng
        )

        # Generate Records B and C
        recs_b = generate_records_b(rec_a, dropout_label, dropout_week,
                                    weight_traj, rng)
        recs_c = generate_records_c(rec_a, dropout_label, dropout_week, rng,
                                    records_b=recs_b)

        all_a.append(rec_a)
        all_b.extend(recs_b)
        all_c.extend(recs_c)

        flat_rows.append(flatten_to_csv_row(rec_a, recs_b, recs_c))
        label_dist[dropout_label] += 1
        if dropout_label:
            dropout_total += 1

    # Save JSON files
    path_a = f"{output_dir}/records_a.json"
    path_b = f"{output_dir}/records_b.json"
    path_c = f"{output_dir}/records_c.json"

    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(all_a, f, indent=2, default=str)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(all_b, f, indent=2, default=str)
    with open(path_c, "w", encoding="utf-8") as f:
        json.dump(all_c, f, indent=2, default=str)

    # Save flat CSV for XGBoost baseline
    path_csv = f"{output_dir}/trajectory_flat.csv"
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    # Summary
    from collections import Counter
    asha_counts  = Counter(p["operational"]["asha_id"] for p in all_a)
    high_dist    = sum(1 for p in all_a if p.get("dropout_label") == 1)
    n_triage_pos = sum(1 for p in all_a if p["triage_positive_at_diagnosis"])
    dr_tb        = sum(1 for p in all_a if p["diagnosis"]["regimen"] in
                       ("BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR"))
    total_b      = len(all_b)
    total_c      = len(all_c)

    print(f"\n  Records A: {len(all_a)} patients  →  {path_a}")
    print(f"  Records B: {total_b} monthly CHO observations  →  {path_b}")
    print(f"  Records C: {total_c} weekly ASHA updates  →  {path_c}")
    print(f"  Flat CSV:  {len(flat_rows)} rows  →  {path_csv}")
    print(f"\n  Dropout (label=1): {label_dist[1]} ({label_dist[1]/n*100:.1f}%)")
    print(f"  Completer (label=0): {label_dist[0]} ({label_dist[0]/n*100:.1f}%)")
    print(f"  Triage-positive: {n_triage_pos} ({n_triage_pos/n*100:.1f}%)")
    print(f"  DR-TB / MDR:     {dr_tb} ({dr_tb/n*100:.1f}%)")
    print(f"  ASHA load range: {min(asha_counts.values())}–{max(asha_counts.values())} patients")
    print(f"  Avg Record B per patient: {total_b/n:.1f}")
    print(f"  Avg Record C per patient: {total_c/n:.1f}")

    # Sanity checks
    assert max(asha_counts.values()) <= MAX_PER_ASHA, "ASHA over-capacity"
    assert label_dist[1] > 0, "No dropout labels generated"
    assert label_dist[1] / n < 0.50, "Dropout rate implausibly high (>50%)"

    print("\n  ✓ All sanity checks passed")

    return {
        "records_a":  all_a,
        "records_b":  all_b,
        "records_c":  all_c,
        "flat_rows":  flat_rows,
        "summary": {
            "n_patients":    n,
            "n_dropouts":    label_dist[1],
            "n_completers":  label_dist[0],
            "dropout_rate":  round(label_dist[1] / n, 4),
            "n_records_b":   total_b,
            "n_records_c":   total_c,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Nikshay-Graph synthetic dataset")
    parser.add_argument("--n",    type=int, default=N_PATIENTS,
                        help="Number of patients (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    args = parser.parse_args()

    result = generate_and_save(n=args.n, seed=args.seed, output_dir=args.output_dir)
