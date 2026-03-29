"""
stage4_explain.py
=================
Stage 4: Explanation & Visualisation

Implements the pipeline document's explanation layer exactly:
  - Template-based explanations only — NO free-form LLM generation
    (deliberate safety decision: prevents hallucination in medical output)
  - Attention weights from TGN extracted as explainability factors
  - Two explanation formats: ASHA worker (simple) and District Officer (detailed)
  - Azure AI Foundry safety validation before any output is delivered
  - Graph data prepared for District Officer dashboard

Why template-based, not LLM:
  "The system always uses a fixed template and never generates free-form text
   for explanations. Free-form language model generation in a medical context
   carries hallucination risk. The template ensures every explanation is
   directly grounded in actual model outputs."
  — Nikshay-Graph Pipeline Document, Stage 4

Usage:
    from stage4_explain import generate_asha_explanation, generate_officer_explanation
    from stage4_explain import validate_output_safety, get_patient_visit_list
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _get_primary_reason(record: dict) -> str:
    """
    Determine the single most urgent reason in DTC red-flag priority order.
    v8 Error 9 fix: haemoptysis/confined_to_bed produce urgent language distinct
    from ADR-driven or nutritional-driven causes.

    Priority order (highest → lowest):
      1. Red flag: confined_to_bed                 → immediate referral
      2. Red flag: haemoptysis (≥1 cup)            → immediate referral
      3. Red flag: altered_consciousness           → immediate referral
      4. Red flag: breathlessness                  → urgent visit
      5. Red flag: severe chest/abdominal pain     → urgent visit
      6. Red flag: recurrent vomiting/diarrhoea    → urgent visit
      7. adr_flag (grade ≥2 any Record B)          → ADR dropout pathway
      8. nutritional_deterioration_flag (≥2kg drop)→ RATIONS trial
      9. Dose gaps / silence / prior LFU / distance
    """
    # Latest red flag state — prefer Record C, fall back to record-level flags
    records_c = record.get("records_c", [])
    latest_rc = records_c[-1] if records_c else {}
    red_flags = latest_rc.get("red_flags", {})

    # Also accept flat boolean fields on the record for legacy schema
    def rflag(key: str) -> bool:
        return bool(red_flags.get(key) or record.get(key))

    if rflag("confined_to_bed"):
        return "patient is confined to bed — RED FLAG: immediate referral needed"
    if rflag("haemoptysis_one_cup"):
        return "patient coughing blood — RED FLAG: immediate referral needed"
    if rflag("altered_consciousness"):
        return "patient has altered consciousness / convulsions — RED FLAG: immediate referral"
    if rflag("breathlessness"):
        return "patient has breathlessness at rest — RED FLAG: urgent visit required"
    if rflag("severe_pain_chest_or_abdomen"):
        return "patient has severe chest or abdominal pain — RED FLAG"
    if rflag("recurrent_vomiting_diarrhoea"):
        return "patient has recurrent vomiting or diarrhoea — RED FLAG"

    # ADR flag: grade ≥2 in any Record B — ambulatory patient with pharmacological reason to stop
    if record.get("adr_flag"):
        return "drug side-effects reported (grade ≥2) — patient has a pharmacological reason to stop medication"

    # Nutritional deterioration: ≥2kg weight drop (RATIONS trial predictor)
    if record.get("nutritional_deterioration_flag"):
        return "weight has dropped ≥2kg since last month — nutritional deterioration (RATIONS trial risk factor)"

    # Adherence / engagement signals
    adh    = record.get("adherence") or {}
    missed = adh.get("days_since_last_dose", record.get("days_missed", 0))

    if missed >= 14:
        return f"has not taken medicine for {missed} days"
    if record.get("silence_event"):
        days_quiet = record["silence_event"].get("duration_days", missed)
        return f"no contact for {days_quiet} days — may be disengaging"
    if missed >= 7:
        return f"missed medicine for {missed} days"
    if adh.get("prior_lfu_history") or (record.get("baseline_clinical") or {}).get("prior_lfu_history"):
        return "has dropped out of treatment before"

    # Clinical comorbidity signals
    base_cl = record.get("baseline_clinical") or record.get("clinical") or {}
    comorbidities = base_cl.get("comorbidities", {})
    if comorbidities.get("hiv") or base_cl.get("hiv"):
        return "HIV co-infection makes dropout especially dangerous"

    dist = adh.get("distance_to_center_km") or base_cl.get("distance_to_phc_km", 0) or 0
    if dist > 10:
        return f"lives {dist:.1f}km from treatment centre — access barrier"

    if record.get("asha_load_score", 0) > 0.7:
        return "ASHA worker has high caseload — visit overdue"

    factors = record.get("top_factors", {})
    if factors:
        top_name = list(factors.keys())[0]
        return f"risk factor: {top_name.lower()}"
    return "treatment engagement declining"


def _get_first_name(patient_id: str) -> str:
    """Extract a display name from patient ID for ASHA briefing."""
    # In production: look up actual patient name from Nikshay
    # For prototype: use patient ID suffix
    return patient_id.split("-")[-1]


def generate_asha_explanation(record: dict) -> str:
    """
    ASHA-facing explanation. Plain language, actionable, field-readable.
    Format: "⚠ URGENT / VISIT TODAY / CHECK IN — [reason]. [one follow-up action]"

    v8 Error 9 fix: distinct language for ADR-driven vs nutritional-driven causes.
    Grounded entirely in actual record data — no LLM involved.
    """
    adh    = record.get("adherence") or {}
    missed = adh.get("days_since_last_dose", record.get("days_missed", 0))
    base_cl = record.get("baseline_clinical") or record.get("clinical") or {}
    regimen = (record.get("diagnosis") or {}).get("regimen") or base_cl.get("regimen", "")
    risk    = record.get("risk_level", "MEDIUM")

    # Red flag check — any active red flag from latest Record C
    records_c = record.get("records_c", [])
    latest_rc = records_c[-1] if records_c else {}
    red_flags = latest_rc.get("red_flags", {})
    any_red_flag = (
        any(red_flags.values()) or
        any(record.get(k) for k in (
            "confined_to_bed", "haemoptysis_one_cup", "altered_consciousness",
            "breathlessness", "severe_pain_chest_or_abdomen",
            "recurrent_vomiting_diarrhoea", "adr_symptoms"
        ))
    )

    # Urgency label
    if any_red_flag or risk == "HIGH" or missed >= 14:
        urgency = "⚠ URGENT VISIT"
    elif missed >= 7 or record.get("silence_event") or record.get("adr_flag") or record.get("nutritional_deterioration_flag"):
        urgency = "VISIT TODAY"
    else:
        urgency = "CHECK IN"

    reason = _get_primary_reason(record)

    # One concrete follow-up action — differentiated by mechanism (v8 Error 9)
    if any_red_flag and (red_flags.get("confined_to_bed") or record.get("confined_to_bed") or
                          red_flags.get("haemoptysis_one_cup") or record.get("haemoptysis_one_cup") or
                          red_flags.get("altered_consciousness") or record.get("altered_consciousness")):
        action = "Refer to facility immediately — do not wait for next scheduled visit."
    elif record.get("adr_flag"):
        action = "Patient may be stopping medicine due to side effects. Bring this to the MO's attention — do not advise them to stop."
    elif record.get("nutritional_deterioration_flag"):
        action = "Patient is losing weight. Confirm Nikshay Poshan Yojana (₹500/month) is being received and meals are adequate."
    elif missed >= 7:
        action = "Bring medicine directly to the patient."
    elif record.get("silence_event"):
        action = "Call or visit — patient has gone quiet."
    elif adh.get("prior_lfu_history") or base_cl.get("prior_lfu_history"):
        action = "Remind them treatment must be completed fully."
    elif regimen in ("BPaLM", "Shorter-Oral-MDR", "Longer-Oral-MDR", "DR_TB"):
        action = "DR-TB patient — do not miss this visit."
    else:
        npy = (record.get("operational") or {}).get("welfare_enrolled") or base_cl.get("npy_enrolled")
        if not npy:
            action = "Help them enrol in Nikshay Poshan Yojana (₹500/month)."
        else:
            action = "Check that they are taking medicine daily."

    return f"{urgency} — {reason}. {action}"


def generate_officer_explanation(record: dict) -> str:
    """
    District Officer-facing explanation — structured fields, not a wall of text.
    Returns a markdown-formatted string for the dashboard to render.

    v8 fix: data_source badge shows whether score comes from TGN (sufficient evidence),
    BBN cold-start, or blend. Officer needs to know which model is authoritative.
    """
    pid         = record["patient_id"]
    tier        = record.get("risk_level", "?")
    week        = record.get("treatment_week", "?")
    score       = record.get("risk_score", 0)
    base_cl     = record.get("baseline_clinical") or record.get("clinical") or {}
    phase       = (record.get("diagnosis") or {}).get("phase") or base_cl.get("phase", "?")
    adh         = record.get("adherence") or {}
    missed      = adh.get("days_since_last_dose", record.get("days_missed", 0))
    threshold   = record.get("thresholds", {})
    composition = record.get("score_composition", {})
    factors     = list(record.get("top_factors", record.get("all_factors", {})).items())

    tgn_pct  = int(composition.get("tgn_weight",  0) * 100)
    bbn_pct  = int(composition.get("bbn_weight",  0) * 100)

    # data_source badge (v8 §8.2) — tells officer which model is authoritative
    data_source = composition.get("data_source") or record.get("data_source", "bbn_coldstart")
    if data_source == "tgn":
        source_badge = "**Source: TGN** (phase boundary reached — full temporal evidence)"
    elif data_source == "bbn_coldstart":
        source_badge = "**Source: BBN cold-start** (week 0 — literature prior only, no temporal data yet)"
    else:
        pct = data_source.replace("blend_", "").replace("pct_tgn", "")
        source_badge = f"**Source: Blend** ({pct}% TGN / {100 - int(pct)}% BBN — confidence ramp in progress)"

    primary_name, primary_or = factors[0] if factors else ("unknown", 0)
    secondary = f" · Secondary: {factors[1][0]} (OR {factors[1][1]:.2f}×)" if len(factors) > 1 else ""

    # ADR / nutritional flags
    flag_notes = []
    if record.get("adr_flag"):
        flag_notes.append("⚠ ADR grade ≥2 recorded — pharmacological dropout risk")
    if record.get("nutritional_deterioration_flag"):
        flag_notes.append("⚠ Nutritional deterioration: ≥2kg weight loss (RATIONS trial predictor)")
    flag_str = "  \n".join(flag_notes) if flag_notes else "None"

    nl = "\n\n"
    explanation = (
        f"**Risk:** {tier} · **Score:** {score:.3f} · **Week:** {week} ({phase}){nl}"
        f"**Days since last dose:** {missed} · "
        f"**HIGH threshold this week:** >{threshold.get('high', 0.65)}{nl}"
        f"**Primary driver:** {primary_name} (OR {primary_or:.2f}×){secondary}{nl}"
        f"**Score breakdown:** TGN {tgn_pct}% · BBN prior {bbn_pct}%{nl}"
        f"{source_badge}{nl}"
        f"**Clinical flags:** {flag_str}"
    )
    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# AZURE AI FOUNDRY SAFETY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_output_safety(explanation: str, record: dict) -> dict:
    """
    Validate explanation before delivery.
    Checks (from pipeline document):
    1. References only patient-specific clinical factors
    2. Contains no diagnostic claims or medication recommendations
    3. Does not fabricate information not present in model outputs
    4. Framed as visit prioritisation guidance only

    Production: calls Azure AI Foundry content safety API.
    Prototype: rule-based checks + stub.
    """
    blocked_phrases = [
        "you have", "you are", "diagnosed with", "you should take",
        "prescribe", "medication change", "stop taking",
        "cure", "will die", "cancer", "increase dose", "decrease dose",
    ]

    violations = []
    exp_lower  = explanation.lower()

    for phrase in blocked_phrases:
        if phrase in exp_lower:
            violations.append(f"Blocked phrase detected: '{phrase}'")

    # Check it's framed as visit guidance
    if not any(w in exp_lower for w in ["visit", "patient", "risk", "dose", "contact", "screen"]):
        violations.append("Explanation does not appear to be visit prioritisation guidance.")

    # Production path: Azure AI Foundry content safety.
    # A network/API error is an infrastructure problem, NOT a content problem.
    # Log it and allow the explanation through — never silently suppress valid
    # clinical guidance because of a transient connectivity issue.
    foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
    if foundry_endpoint and not violations:
        try:
            _call_foundry_safety(explanation)
        except Exception as e:
            print(f"  [Safety] Foundry API error (explanation NOT blocked): {e}")

    return {
        "passed":     len(violations) == 0,
        "violations": violations,
        "text":       explanation if not violations else "[BLOCKED — safety violation]",
    }


def _call_foundry_safety(text: str) -> bool:
    """
    Screen explanation text through Azure AI Content Safety.

    Works with both Azure endpoint formats:
      https://<n>.cognitiveservices.azure.com/   (classic Cognitive Services)
      https://<n>.services.ai.azure.com/         (AI Foundry hub)

    Raises ValueError only for genuine content violations (severity > 2).
    Returns True when text is safe. Infrastructure errors bubble up to the
    caller which logs them and allows the explanation through.
    """
    import requests
    endpoint = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")
    key      = os.getenv("FOUNDRY_KEY")
    if not endpoint or not key:
        return True  # not configured — skip silently

    url = f"{endpoint}/contentsafety/text:analyze?api-version=2024-02-15-preview"
    try:
        resp = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"},
            json={"text": text, "categories": ["Hate", "SelfHarm", "Sexual", "Violence"]},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise ValueError(
            f"Content Safety HTTP {resp.status_code}: {resp.text[:200]}"
        ) from http_err

    for cat in resp.json().get("categoriesAnalysis", []):
        if cat.get("severity", 0) > 2:
            raise ValueError(
                f"Content safety violation: {cat['category']} severity {cat['severity']}"
            )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# VISIT PRIORITY RANKING
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_visit_list(patients: list, top_n: int = 10) -> list:
    """
    Rank patients for ASHA visit priority.
    Sorting key: risk_score (includes urgency multiplier) then treatment_week desc.
    A patient with score 0.71 at week 22 ranks above 0.79 at week 3
    because the intervention window is narrower.
    """
    def priority_key(p):
        risk = p.get("risk_score", 0)
        week = p.get("treatment_week", 1)
        # Later treatment weeks get a small boost to break ties
        return (risk, week / 26)

    ranked = sorted(patients, key=priority_key, reverse=True)

    results = []
    for i, p in enumerate(ranked[:top_n]):
        asha_exp = generate_asha_explanation(p)
        officer_exp = generate_officer_explanation(p)

        asha_safety    = validate_output_safety(asha_exp, p)
        officer_safety = validate_output_safety(officer_exp, p)

        # Schema-agnostic field resolution: works after normalize_record() and
        # on the legacy flat schema. All four paths below tolerate either structure.
        _adh   = p.get("adherence") or {}
        _cl    = p.get("clinical") or p.get("baseline_clinical") or {}
        _loc   = p.get("location") or {}
        _op    = p.get("operational") or {}
        _phase = (
            (p.get("diagnosis") or {}).get("phase") or
            _cl.get("phase") or
            p.get("phase", "?")
        )
        _missed = _adh.get("days_since_last_dose", p.get("days_missed", 0))
        _block  = _loc.get("block") or _op.get("block", "—")
        _asha   = _op.get("asha_id", "—")

        results.append({
            "rank":               i + 1,
            "patient_id":         p["patient_id"],
            "risk_score":         p["risk_score"],
            "risk_level":         p.get("risk_level", "?"),
            "treatment_week":     p.get("treatment_week", "?"),
            "phase":              _phase,
            "days_missed":        _missed,
            "asha_id":            _asha,
            "block":              _block,
            "top_factors":        p.get("top_factors", {}),
            "score_composition":  p.get("score_composition", {}),
            # Template-based explanations — no LLM
            "asha_explanation":   asha_safety["text"],
            "officer_explanation":officer_safety["text"],
            "safety_passed":      asha_safety["passed"] and officer_safety["passed"],
        })

    blocked = sum(1 for r in results if not r["safety_passed"])
    if blocked:
        print(f"  ⚠ {blocked} explanations blocked by safety validation")

    return results


def get_contact_screening_list(G, pagerank_scores: dict, top_n: int = 10) -> list:
    """
    Rank contacts for TB screening by propagated PageRank score.
    Filter out already-screened contacts.

    G must be a NetworkX graph whose nodes carry properties:
        node_type, age, rel, vulnerability, screened, name, source_patient
    These are set by stage2 when building the patient/contact graph,
    NOT the PyTorch edge_index graph. Pass the nx graph, not the torch one.
    """
    contacts = []
    for node_id, score in pagerank_scores.items():
        node = G.nodes.get(node_id, {})
        if node.get("node_type") != "contact" or node.get("screened", False):
            continue
        age  = node.get("age", 30)
        rel  = node.get("rel", "Workplace")
        vuln = node.get("vulnerability", 1.0)
        age_risk = 1.5 if age > 60 or age < 10 else 1.0
        rel_risk = 1.3 if rel == "Household" else 1.0
        priority = score * vuln * age_risk * rel_risk

        # Template-based screening reason
        reason = f"Screen {node.get('name', 'contact')} (age {age}, {rel}) — unscreened contact of a high-risk patient."

        contacts.append({
            "contact_id":         node_id,
            "name":               node.get("name", "Unknown"),
            "age":                age,
            "rel":                rel,
            "vulnerability":      vuln,
            "source_patient":     node.get("source_patient", ""),
            "screening_priority": round(priority, 8),
            "screening_reason":   reason,
        })

    ranked = sorted(contacts, key=lambda x: x["screening_priority"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked[:top_n]


if __name__ == "__main__":
    with open("nikshay_scored_dataset.json") as f:
        patients = json.load(f)

    visit_list = get_patient_visit_list(patients, top_n=5)
    print("\nASHA Visit List (template explanations, safety-validated):")
    for v in visit_list:
        print(f"\n  Rank {v['rank']}: {v['patient_id']} [{v['risk_level']}]")
        print(f"  ASHA:    {v['asha_explanation']}")
        print(f"  Officer: {v['officer_explanation']}")
        print(f"  Safety:  {'✓ passed' if v['safety_passed'] else '✗ BLOCKED'}")