"""
schema_utils.py
===============
Schema adapter: translates the v2 three-record dataset (Records A / B / C)
produced by dataset_gen_v2.py into the enriched flat patient dict that every
downstream pipeline stage (stage1_nlp, stage2_tgn, stage3_score, stage4_explain,
app.py) expects.

WHY THIS EXISTS
---------------
dataset_gen_v2 produces a three-record structure with different field names and
nesting than the legacy flat schema the rest of the pipeline was written for.
Rather than touching every downstream file, a single normalise_record() call at
load time (main.py, function_app.py) makes the v2 records look like the flat
schema. All downstream code then works without modification.

FIELD MAPPING SUMMARY
---------------------
v2 field                          → flat schema field
───────────────────────────────────────────────────────
nikshay_id                        → patient_id
district (top-level)              → location.district
operational.asha_id block         → location.block  (via BLOCK constant)
diagnosis.regimen                 → clinical.regimen + clinical.phase
treatment_week + total_wks        → clinical.total_treatment_days
baseline_clinical.*               → clinical.comorbidities + passthrough
social.distance_to_phc_km         → adherence.distance_to_center_km
welfare.npy_enrolled              → operational.welfare_enrolled
Record C aggregation              → adherence.days_since_last_dose
                                     adherence.adherence_rate_30d
                                     operational.last_asha_visit_days_ago
                                     adherence.silence_days
contact.relationship_type         → contact.rel
contact (identity-only)           → contact + screened:False + vulnerability_score

Usage:
    from schema_utils import normalize_dataset, normalize_record

    # Load v2 data
    with open("data/records_a.json") as f: records_a = json.load(f)
    with open("data/records_b.json") as f: records_b = json.load(f)
    with open("data/records_c.json") as f: records_c = json.load(f)

    patients = normalize_dataset(records_a, records_b, records_c)
    # patients is now a list of flat dicts compatible with all pipeline stages
"""

from __future__ import annotations
from typing import Optional

# Block name from the dataset generator — all Tondiarpet patients share this.
# In production this would come from operational.block on each record.
_DEFAULT_BLOCK = "Tondiarpet"


def normalize_dataset(records_a: list, records_b: list, records_c: list) -> list:
    """
    Merge Records A, B, C for all patients into flat dicts.

    Builds lookup dicts from B and C by nikshay_id so each record_a can be
    enriched in O(1). Returns a list of normalised patient dicts ready for
    every downstream stage.
    """
    # Index records_b and records_c by nikshay_id
    b_by_id: dict[str, list] = {}
    for rb in records_b:
        nid = rb.get("nikshay_id", "")
        b_by_id.setdefault(nid, []).append(rb)

    c_by_id: dict[str, list] = {}
    for rc in records_c:
        nid = rc.get("nikshay_id", "")
        c_by_id.setdefault(nid, []).append(rc)

    patients = []
    for ra in records_a:
        nid  = ra.get("nikshay_id", "")
        rbs  = sorted(b_by_id.get(nid, []), key=lambda x: x.get("month", 0))
        rcs  = sorted(c_by_id.get(nid, []), key=lambda x: x.get("week", 0))
        patients.append(normalize_record(ra, rbs, rcs))

    return patients


def normalize_record(record_a: dict,
                     records_b: Optional[list] = None,
                     records_c: Optional[list] = None) -> dict:
    """
    Translate one Record A (+ its B and C sequences) into the flat patient dict.

    All fields set here are exactly those accessed by the downstream pipeline.
    The original v2 keys are preserved alongside the flat-schema aliases so
    any code that directly reads the v2 structure still works.
    """
    records_b = records_b or []
    records_c = records_c or []

    ra   = record_a
    op   = ra.get("operational", {})
    diag = ra.get("diagnosis", {})
    bcl  = ra.get("baseline_clinical", {})
    soc  = ra.get("social", {})
    wel  = ra.get("welfare", {})
    dem  = ra.get("demographics", {})

    # ── patient_id ──────────────────────────────────────────────────────────
    pid = ra.get("nikshay_id") or ra.get("patient_id", "")

    # ── treatment timing ────────────────────────────────────────────────────
    treatment_week  = ra.get("treatment_week", 0)
    total_wks       = ra.get("total_treatment_weeks", 26)
    total_days      = treatment_week * 7   # approximate; good enough for week derivation

    # ── phase ───────────────────────────────────────────────────────────────
    regimen         = diag.get("regimen", "DS-TB")
    phase_boundary  = 16 if regimen in ("Shorter-Oral-MDR", "Longer-Oral-MDR") else 8
    phase           = "Intensive" if treatment_week <= phase_boundary else "Continuation"

    # ── aggregated adherence from Record C ──────────────────────────────────
    n_confirmed  = sum(1 for rc in records_c if rc.get("dose_status") == "confirmed")
    n_missed     = sum(1 for rc in records_c if rc.get("dose_status") == "missed")
    n_weeks      = max(len(records_c), 1)

    # days_since_last_dose: look at most recent Record C entries to find last confirmed
    days_since_last_dose = 0
    last_visit_days_ago  = 0
    silence_days         = 0

    if records_c:
        latest_rc = records_c[-1]

        # Silence days from most recent week's record
        silence_days = latest_rc.get("silence_days", 0)

        # How many consecutive missed/unvisited weeks trail the end of the record
        consecutive_missed = 0
        for rc in reversed(records_c):
            if rc.get("dose_status") == "missed" or not rc.get("visited", False):
                consecutive_missed += 1
            else:
                break
        days_since_last_dose = consecutive_missed * 7

        # last_asha_visit_days_ago: days since last visited=True week
        last_visited_week = None
        for rc in reversed(records_c):
            if rc.get("visited", False):
                last_visited_week = rc.get("week", 0)
                break
        if last_visited_week is not None:
            last_visit_days_ago = (records_c[-1].get("week", 0) - last_visited_week) * 7
        else:
            last_visit_days_ago = len(records_c) * 7

    adherence_rate_30d = round(n_confirmed / n_weeks, 3)

    # ── contact_network: add rel alias + screening defaults ─────────────────
    contacts_raw = ra.get("contact_network", [])
    contacts_flat = []
    for c in contacts_raw:
        rel_type = c.get("relationship_type") or c.get("rel", "Household")
        contacts_flat.append({
            **c,
            "rel":               rel_type,        # alias relationship_type → rel
            "relationship_type": rel_type,        # keep original too
            "screened":          c.get("screened", False),
            "vulnerability_score": c.get("vulnerability_score", 1.0),
        })

    # ── operational flags ────────────────────────────────────────────────────
    npy_enrolled = wel.get("npy_enrolled", False)
    # nutritional_support: approximated by NPY enrollment
    nutritional_support = npy_enrolled

    # phone_type
    phone_type = op.get("phone_type", "Basic")

    # ── location ─────────────────────────────────────────────────────────────
    district = ra.get("district", "Chennai")
    block    = op.get("block", _DEFAULT_BLOCK)   # v2 doesn't store block on op;
                                                 # fall back to Tondiarpet constant

    # ── flat patient dict ────────────────────────────────────────────────────
    flat = {
        # ── Identity ──────────────────────────────────────────────────────
        "patient_id":    pid,
        "nikshay_id":    pid,            # keep v2 name too

        # ── Location (flat schema) ─────────────────────────────────────────
        "location": {
            "district": district,
            "block":    block,
            "state":    ra.get("state", "Tamil Nadu"),
        },

        # ── Clinical (flat schema) ─────────────────────────────────────────
        "clinical": {
            "phase":               phase,
            "regimen":             regimen,
            "total_treatment_days": total_days,
            "comorbidities": {
                "hiv":      bcl.get("hiv", False),
                "diabetes": bcl.get("diabetes", False),
            },
            # passthrough extras used by scoring
            "hiv":           bcl.get("hiv", False),
            "diabetes":      bcl.get("diabetes", False),
        },

        # ── Adherence (flat schema) ────────────────────────────────────────
        "adherence": {
            "days_since_last_dose":  days_since_last_dose,
            "adherence_rate_30d":    adherence_rate_30d,
            "distance_to_center_km": soc.get("distance_to_phc_km", 0.0),
            "prior_lfu_history":     ra.get("prior_lfu_history", False),
            "prior_tb_history":      soc.get("prior_tb_history", False),
            "phase_adherence_rate":  adherence_rate_30d,
            "silence_days":          silence_days,
        },

        # ── Operational (flat schema) ──────────────────────────────────────
        "operational": {
            "asha_id":              op.get("asha_id", ""),
            "cho_id":               op.get("cho_id", ""),
            "welfare_enrolled":     npy_enrolled,
            "npy_enrolled":         npy_enrolled,
            "nutritional_support":  nutritional_support,
            "last_asha_visit_days_ago": last_visit_days_ago,
            "phone_type":           phone_type,
            "language":             op.get("language", "Tamil"),
        },

        # ── Social (flat schema) ──────────────────────────────────────────
        "social": {
            "alcohol_use":            soc.get("alcohol_use", False),
            "tobacco_smoking":        soc.get("tobacco_smoking", False),
            "substance_use_disorder": soc.get("substance_use_disorder", False),
            "drug_use":               soc.get("substance_use_disorder", False),  # alias
            "low_education":          soc.get("low_education", False),
            "distance_to_phc_km":     soc.get("distance_to_phc_km", 0.0),
            "prior_tb_history":       soc.get("prior_tb_history", False),
        },

        # ── Demographics (flat schema) ─────────────────────────────────────
        "demographics": {
            "age":           dem.get("age", 30),
            "sex":           dem.get("sex", "Male"),
            "gender":        dem.get("sex", "Male"),   # alias
            "marital":       dem.get("marital_status", ""),
            "marital_status":dem.get("marital_status", ""),
            "bpl_status":    dem.get("bpl_status", False),
            "education_level":dem.get("education_level", ""),
        },

        # ── Baseline clinical (kept under its v2 key too) ──────────────────
        "baseline_clinical": {
            **bcl,
            "bmi_at_diagnosis":      bcl.get("bmi", None),
            "prior_lfu_history":     ra.get("prior_lfu_history", False),
            "prior_tb_history":      soc.get("prior_tb_history", False),
            "npy_enrolled":          npy_enrolled,
            "distance_to_phc_km":    soc.get("distance_to_phc_km", 0.0),
            "substance_use_disorder":soc.get("substance_use_disorder", False),
            "alcohol_use":           soc.get("alcohol_use", False),
            "low_education":         soc.get("low_education", False),
        },

        # ── Contact network (with rel alias) ──────────────────────────────
        "contact_network": contacts_flat,

        # ── Welfare (kept under v2 key) ────────────────────────────────────
        "welfare": {
            "npy_enrolled":            npy_enrolled,
            "npy_bank_account_tagged": wel.get("npy_bank_account_tagged", False),
        },

        # ── Treatment timeline ─────────────────────────────────────────────
        "treatment_week":        treatment_week,
        "total_treatment_weeks": total_wks,
        "treatment_start_date":  ra.get("treatment_start_date", ""),
        "date_of_diagnosis":     ra.get("date_of_diagnosis", ""),

        # ── Diagnosis (v2 key kept) ────────────────────────────────────────
        "diagnosis": diag,

        # ── NTEP flags ────────────────────────────────────────────────────
        "prior_lfu_history":            ra.get("prior_lfu_history", False),
        "triage_positive_at_diagnosis": ra.get("triage_positive_at_diagnosis", False),

        # ── v2 label fields (preserved for training) ──────────────────────
        "dropout_label": int(ra.get("dropout_label", 0)),
        "dropout_week":  ra.get("dropout_week", None),

        # ── v2 records attached for Stage 2/3 ─────────────────────────────
        # compute_clinical_flags() and encode_event() read these directly
        "records_b": records_b,
        "records_c": records_c,

        # ── Computed at scoring time (populated by stage3_score) ──────────
        "risk_score":      0.0,
        "risk_level":      "LOW",
        "risk_velocity":   0.0,
        "tgn_score":       None,
        "asha_load_score": 0.3,
        "data_source":     "bbn_coldstart",
    }

    # Carry through any triage-positive Annexure 2 fields
    if ra.get("triage_positive_at_diagnosis"):
        flat["triage_positive_fields"] = ra.get("triage_positive_fields", {})

    return flat
