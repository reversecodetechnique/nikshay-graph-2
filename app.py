"""
app.py — Nikshay-Graph Dashboard

Six tabs matching the care hierarchy:
  1. ASHA Portal          — visit priority list, red flag checklist (Box 4.1)
  2. CHO Clinical Desk    — Annexure 1 input form, pending ASHA alerts, contact screening
  3. MO Assessment        — Annexure 2 input form, pending CHO referral alerts
  4. DTO Command View     — severe MO-assessed cases only, district stats
  5. Explainability       — score audit log, patient explanations, pipeline trigger
  6. TGN Graph            — network visualisation with CHO/PHC nodes

Alert routing (matches clinical hierarchy):
  ASHA red flag  →  CHO tab  (CHO is first clinical responder per Annexure 5)
  CHO referral   →  MO tab   (MO at PHC/CHC receives referred patients)
  MO severe      →  DTO tab  (DTO only sees MO-confirmed severe cases)
"""

import os
import json
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Nikshay-Graph",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "pref_language" not in st.session_state:
    st.session_state.pref_language = "English"
if "graph_activity" not in st.session_state:
    st.session_state.graph_activity = []
if "asha_updates" not in st.session_state:
    st.session_state.asha_updates = {}
if "screened_contacts" not in st.session_state:
    st.session_state.screened_contacts = set()
if "translation_cache" not in st.session_state:
    st.session_state.translation_cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def utc_to_local(iso_str: str, fmt: str = "%d %b %H:%M") -> str:
    """Convert UTC ISO timestamp (from JSON files) to local system time string."""
    from datetime import datetime, timezone
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        dt_local = dt_utc.astimezone()          # converts to local system timezone
        return dt_local.strftime(fmt)
    except Exception:
        return iso_str[:16]


def log_graph_activity(action: str, patient_id: str, detail: str):
    from datetime import datetime
    st.session_state.graph_activity.insert(0, {
        "time":       datetime.now().strftime("%H:%M:%S"),
        "action":     action,
        "patient_id": patient_id,
        "detail":     detail,
    })
    st.session_state.graph_activity = st.session_state.graph_activity[:20]


def translate_ui(text: str, language: str) -> str:
    """Translate a UI string at render time. Caches per session. Falls back to English."""
    if language == "English":
        return text
    cache_key = f"{language}::{text[:80]}"
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
    try:
        from stage5_voice import translate_text
        result = translate_text(text, language)
        st.session_state.translation_cache[cache_key] = result
        return result
    except Exception:
        return text


def rescore_patient_locally(patient_id: str, action: str, note: str = "",
                             record_b: dict = None,
                             active_flags: list = None) -> dict:
    """
    Re-score a patient after any worker submits an update.
    Accepts an optional record_b dict from CHO submission — this is appended
    to the patient's records_b list so both BBN and the TGN event stream
    see the new clinical data (weight delta, ADR grade, management decision).
    active_flags: list of red flag key strings set by ASHA (e.g. ["adr_symptoms"]).
      These are applied to the patient dict before BBN recomputation so that
      the BBN log-odds terms pick them up naturally (no hardcoded delta).
    Returns {old_score, new_score, old_tier, new_tier, changed}.
    """
    from stage3_score import (compute_bbn_prior, compute_risk_score_v8,
                               assign_risk_tier, get_adaptive_thresholds,
                               append_score_audit)

    json_path = Path("nikshay_scored_dataset.json")
    if not json_path.exists():
        json_path = Path("data/nikshay_scored_dataset.json")
    if not json_path.exists():
        return {}

    with open(json_path, encoding="utf-8") as f:
        all_patients = json.load(f)

    patient = next((p for p in all_patients if p["patient_id"] == patient_id), None)
    if not patient:
        return {}

    old_score = patient.get("risk_score", 0)
    old_tier  = patient.get("risk_level", "?")

    adh = patient.get("adherence") or {}
    op  = patient.get("operational") or {}

    if action == "done":
        adh["days_since_last_dose"] = 0
        op["last_asha_visit_days_ago"] = 0
        patient.pop("silence_event", None)
    elif action == "could_not_visit":
        op["last_asha_visit_days_ago"] = op.get("last_asha_visit_days_ago", 0) + 1
        # Under DOT, if ASHA cannot visit the dose is not observed/confirmed.
        # Increment days_since_last_dose so the BBN missed-dose OR terms fire
        # and the TGN encodes a DOSE_MISSED event instead of DOSE_CONFIRMED.
        adh["days_since_last_dose"] = adh.get("days_since_last_dose", 0) + 1
    elif action == "free_text" and note:
        patient["free_text_note"] = (patient.get("free_text_note", "") + " " + note).strip()

    # Append new Record B and propagate clinical data into adherence for BBN
    if record_b:
        existing_rbs = patient.get("records_b", [])
        existing_rbs.append(record_b)
        patient["records_b"] = existing_rbs
        vitals = record_b.get("vitals", {})
        if vitals.get("weight_delta_kg") is not None:
            adh["weight_delta_kg"] = vitals["weight_delta_kg"]
        adr = record_b.get("adr", {})
        adr_grade_val = adr.get("grade", 0)
        adh["adr_grade"] = adr_grade_val          # always overwrite — CHO may report improvement
        if adr_grade_val >= 2:
            adh["adr_symptoms"] = True
        elif adr_grade_val == 0:
            adh.pop("adr_symptoms", None)          # grade 0 = fully resolved, remove flag
        prog = record_b.get("programme", {})
        if prog.get("nikshay_divas_attended") is not None:
            adh["nikshay_divas_attended"] = prog["nikshay_divas_attended"]
        if prog.get("npy_benefit_received") is not None:
            op["welfare_enrolled"] = prog["npy_benefit_received"]
        mgmt = prog.get("management_decision", "")
        if mgmt:
            adh["management_decision"] = mgmt
            # Any referral decision implies MO formally assessed the patient
            if mgmt in ("referral_to_higher_centre", "referral_for_hospitalisation"):
                adh["mo_assessment_done"] = True

    # Apply red flags to the patient dict before BBN recomputation so that
    # the BBN's log-odds terms (e.g. adr_symptoms) pick them up naturally.
    if active_flags:
        existing_rf = set(patient.get("active_red_flags", []))
        existing_rf.update(active_flags)
        patient["active_red_flags"] = list(existing_rf)
        if any("adr" in f.lower() for f in active_flags):
            adh["adr_symptoms"] = True
        for flag in active_flags:
            patient[flag] = True

    patient["adherence"]   = adh
    patient["operational"] = op

    bbn_result = compute_bbn_prior(patient)
    bbn_score  = bbn_result["score"]
    asha_load  = patient.get("asha_load_score", 0.3)

    week      = patient.get("treatment_week", 1)
    total_wks = patient.get("total_treatment_weeks") or 26
    week      = min(int(week), int(total_wks))

    regimen = (
        (patient.get("diagnosis") or {}).get("regimen") or
        (patient.get("clinical") or {}).get("regimen") or
        "DS-TB"
    )

    # Attempt live TGN re-inference so ASHA updates produce real score movement.
    # score_single_patient() loads the trained GRU+GATConv model, encodes the
    # patient's current event state (with the updated adherence fields), and
    # returns a fresh dropout probability — independent of the stale cached score.
    tgn_trained = False
    try:
        from stage2_tgn import score_single_patient, is_tgn_trained
        if is_tgn_trained():
            # CHO Annexure 1 submissions are monthly events (delta_t=30.0).
            # All other updates (ASHA dose confirm/miss) are weekly (delta_t=7.0).
            _delta_t    = 30.0 if action == "cho_assessment" else 7.0
            tgn_score   = score_single_patient(patient, all_patients, delta_t=_delta_t)
            tgn_trained = True
        else:
            # No trained weights — fall through; tgn_trained=False causes
            # compute_risk_score_v8 to use bbn_w=1.0 so changes are fully visible.
            tgn_score = patient.get("tgn_score", bbn_score)
    except Exception:
        tgn_score = patient.get("tgn_score", bbn_score)

    composition = compute_risk_score_v8(
        patient, tgn_score, bbn_score, asha_load,
        treatment_week=week, regimen=regimen,
        tgn_trained=tgn_trained,
    )
    new_score  = round(float(np.clip(composition["composite_score"], 0, 1)), 4)

    # Guarantee: referral to a higher centre must always reduce the risk score.
    # The TGN's stale weights can otherwise overwhelm the BBN's protective signal,
    # especially before retraining on the new management_decision encoding.
    if record_b:
        _rb_mgmt = record_b.get("programme", {}).get("management_decision", "")
        if _rb_mgmt in ("referral_to_higher_centre", "referral_for_hospitalisation"):
            new_score = round(min(new_score, max(old_score - 0.05, 0.01)), 4)

    rank_score = round(new_score + (week / 10000.0), 6)
    new_tier   = assign_risk_tier(new_score, week)
    thresholds = get_adaptive_thresholds(week)

    patient["previous_risk_score"] = old_score
    patient["risk_score"]          = new_score
    patient["rank_score"]          = rank_score
    patient["risk_level"]          = new_tier
    patient["risk_velocity"]       = round(new_score - old_score, 4)
    patient["score_composition"]   = composition
    patient["data_source"]         = composition["data_source"]
    patient["thresholds"]          = thresholds
    patient["all_factors"]         = bbn_result["all_factors"]
    patient["top_factors"]         = dict(
        sorted(bbn_result["all_factors"].items(), key=lambda x: x[1], reverse=True)[:3]
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_patients, f, indent=2)

    # Build change_reason: explains which feature update drove the score change.
    # For CHO form submissions, name the specific clinical signals that fired.
    change_reason = ""
    if record_b:
        _reasons = []
        _adr_g = record_b.get("adr", {}).get("grade", 0)
        _wt    = record_b.get("vitals", {}).get("weight_delta_kg", 0)
        _mgmt  = record_b.get("programme", {}).get("management_decision", "")
        if _adr_g >= 2:
            _reasons.append(f"ADR Grade {_adr_g} (strongest dropout predictor)")
        if _wt < -2.0:
            _reasons.append(f"Weight decline {_wt:.1f} kg (RATIONS trial signal)")
        if _mgmt in ("referral_to_higher_centre", "referral_for_hospitalisation"):
            _reasons.append("Referred to higher centre → risk reduced (patient in system)")
        elif _mgmt == "ambulatory_care":
            _reasons.append("Continued ambulatory care")
        if not _reasons:
            _reasons.append("CHO Annexure 1 — no high-risk signals found")
        change_reason = "; ".join(_reasons)
    elif action == "done":
        change_reason = "ASHA confirmed dose administered"
    elif action == "could_not_visit":
        change_reason = "ASHA could not visit — dose unconfirmed"
    elif action == "free_text":
        change_reason = "Free-text note submitted"
    elif active_flags:
        change_reason = f"Red flags set: {', '.join(active_flags)}"

    # Only audit when the score actually moved — zero-delta entries (e.g. red
    # flag submissions that don't affect dropout risk) just add noise.
    if abs(new_score - old_score) > 0.001:
        try:
            append_score_audit(
                patient_id, old_score=old_score, new_score=new_score,
                old_tier=old_tier, new_tier=new_tier,
                trigger="asha_update" if not record_b else "cho_assessment",
                composition=composition,
                change_reason=change_reason,
            )
        except Exception:
            pass

    return {
        "old_score": round(old_score, 3),
        "new_score": round(new_score, 3),
        "old_tier":  old_tier,
        "new_tier":  new_tier,
        "changed":   old_tier != new_tier,
        "velocity":  round(new_score - old_score, 4),
    }


def _reply_event(asha_id: str, patient_id: str, action: str,
                 free_text: str = "", contact_name: str = ""):
    try:
        from stage5_voice import process_asha_dashboard_reply
        from stage1_nlp import get_eventhub_producer
        producer = get_eventhub_producer()
        gc = None
        try:
            from cosmos_client import get_client, health_check
            if health_check():
                gc = get_client()
        except Exception:
            pass
        process_asha_dashboard_reply(
            gc=gc, producer=producer, action=action,
            patient_id=patient_id, asha_id=asha_id,
            free_text=free_text, contact_name=contact_name,
        )
        rescore = rescore_patient_locally(patient_id, action, free_text)
        st.cache_data.clear()
        # Only log and toast when the score moved — zero-delta updates are silent.
        if rescore and abs(rescore.get("velocity", 0)) > 0.001:
            detail = f"Action: {action}"
            if rescore.get("changed"):
                detail += f" | TIER: {rescore['old_tier']} → {rescore['new_tier']}"
            log_graph_activity(action, patient_id, detail)
            if rescore.get("changed"):
                st.toast(f"⚡ {patient_id}: {rescore['old_tier']} → {rescore['new_tier']}", icon="🔄")
            else:
                st.toast(f"Score updated: {patient_id} ({rescore['old_score']:.3f} → {rescore['new_score']:.3f})", icon="📊")
        else:
            st.toast(f"Saved: {patient_id}", icon="✅")
    except Exception as e:
        log_graph_activity(action, patient_id, f"OFFLINE: {e}")
        st.toast(f"Saved locally ({e})", icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_patients():
    for p in ["nikshay_scored_dataset.json", "data/nikshay_scored_dataset.json"]:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # Cap stale scores > 0.97 from pre-fix pipeline runs
            for _p in data:
                if _p.get("risk_score", 0) > 0.97:
                    _p["risk_score"] = 0.97
            return data
    return []

@st.cache_data(ttl=60)
def load_agent3():
    for p in ["agent3_output.json", "data/agent3_output.json"]:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {"visit_list": [], "screening_list": [], "systemic_alerts": []}

@st.cache_data(ttl=60)
def load_briefings():
    for p in ["briefings_output.json", "data/briefings_output.json"]:
        if Path(p).exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {"asha_briefings": {}, "systemic_alerts": []}

def load_json_file(path: str) -> list:
    p = Path(path)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


patients       = load_scored_patients()
agent3         = load_agent3()
briefings      = load_briefings()
visit_list     = agent3.get("visit_list", [])
asha_briefings = briefings.get("asha_briefings", {})
n_patients     = max(len(patients), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫁 Nikshay-Graph")
    st.caption("TB Treatment Dropout Prevention")
    st.divider()

    try:
        from stage5_voice import ALL_INDIAN_LANGUAGES
    except ImportError:
        ALL_INDIAN_LANGUAGES = ["English", "Hindi", "Tamil", "Telugu", "Kannada",
                                 "Malayalam", "Marathi", "Gujarati", "Bengali",
                                 "Punjabi", "Odia", "Assamese", "Urdu"]

    # Always ensure English appears first in the list
    _lang_options = ["English"] + [l for l in ALL_INDIAN_LANGUAGES if l != "English"]
    _cur_lang = st.session_state.pref_language
    _lang_idx = _lang_options.index(_cur_lang) if _cur_lang in _lang_options else 0

    selected_lang = st.selectbox(
        "🌐 ASHA Portal Language",
        _lang_options,
        index=_lang_idx,
        key="lang_selector",
        help="English + all regional languages. Changes text and audio in the ASHA Portal tab.",
    )
    if selected_lang != st.session_state.pref_language:
        st.session_state.pref_language = selected_lang
        st.session_state.translation_cache = {}   # clear so all strings re-translate
        # Clear on-demand audio cache so audio regenerates in the new language
        for _k in list(st.session_state.keys()):
            if _k.startswith("audio_gen::"):
                del st.session_state[_k]
        st.rerun()

    if selected_lang != "English":
        _has_translator = bool(os.getenv("TRANSLATOR_KEY"))
        if not _has_translator:
            st.warning("⚠️ Set TRANSLATOR_KEY in .env to enable translations.", icon="🌐")

    st.divider()
    n_high   = sum(1 for p in patients if p.get("risk_level") == "HIGH")
    n_silent = sum(
        1 for p in patients
        if max((p.get("adherence") or {}).get("days_since_last_dose", 0),
               (p.get("operational") or {}).get("last_asha_visit_days_ago", 0)) >= 7
    ) if patients else 0

    st.metric("Total Patients", n_patients)
    st.metric("High Risk",  n_high,   delta=f"{n_high/n_patients*100:.1f}%",
              delta_color="inverse")
    st.metric("Silent 7d+", n_silent, delta=f"{n_silent/n_patients*100:.1f}%",
              delta_color="inverse")
    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    recent = st.session_state.graph_activity[:5]
    if recent:
        st.divider()
        st.markdown("**📡 Recent updates**")
        for ev in recent:
            icon = "🔄" if "TIER" in ev.get("detail", "") else "✓"
            st.caption(f"{icon} {ev['time']} · {ev['patient_id']}")
    if not patients:
        st.warning("No data.\nRun: `python main.py --limit 100`")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "👷 ASHA Portal",
    "🏥 CHO Clinical Desk",
    "🩺 MO Assessment",
    "🏛️ DTO Command View",
    "🔍 Explainability",
    "🕸️ TGN Graph",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASHA PORTAL
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    lang = st.session_state.pref_language
    st.header(translate_ui("👷 ASHA Visit Portal", lang))
    st.caption(translate_ui(
        "Your priority patient list for today. Tap each card to record the visit outcome.", lang
    ))

    if not patients:
        st.info(translate_ui("No data loaded yet.", lang))
    else:
        asha_ids = sorted({(p.get("operational") or {}).get("asha_id", "")
                           for p in patients if (p.get("operational") or {}).get("asha_id")})
        selected_asha = st.selectbox(
            translate_ui("Select ASHA worker", lang), asha_ids, key="asha_selector"
        )

        asha_patients = sorted(
            [p for p in patients
             if (p.get("operational") or {}).get("asha_id") == selected_asha],
            key=lambda x: x.get("rank_score", x.get("risk_score", 0)),
            reverse=True,
        )

        briefing = asha_briefings.get(selected_asha, {})
        if briefing:
            # Rebuild English text from visit_cards so it always uses patient IDs
            # (pre-stored english_text may have been generated by an older pipeline run)
            _cards = briefing.get("visit_cards", [])
            if _cards:
                _high = [c for c in _cards if c.get("risk_level") == "HIGH"][:3] or _cards[:3]
                _lines = [f"Good morning. You have {len(_high)} priority patient(s) today."]
                for _c in _high:
                    _lines.append(f"Patient {_c['patient_id']}: {_c['explanation']}")
                _lines.append("Please update each card after your visit.")
                english_text = " ".join(_lines)
            else:
                english_text = briefing.get("english_text", "")

            if english_text:
                cache_key = f"briefing::{selected_asha}::{lang}"
                if cache_key not in st.session_state.translation_cache:
                    st.session_state.translation_cache[cache_key] = translate_ui(english_text, lang)
                st.info(st.session_state.translation_cache[cache_key])


            # Audio — only auto-play pre-generated file if it matches selected language
            audio_path = briefing.get("audio_path")
            _briefing_lang = briefing.get("language", "English")
            _audio_served = False
            if audio_path and Path(audio_path).exists() and _briefing_lang == lang:
                st.audio(audio_path)
                _audio_served = True

            # On-demand audio in selected language (always available as button)
            _briefing_text_for_audio = st.session_state.translation_cache.get(
                f"briefing::{selected_asha}::{lang}", english_text
            )
            _audio_btn_label = translate_ui(f"🔊 Play in {lang}", lang)
            _gen_key = f"audio_gen::{selected_asha}::{lang}"
            if not _audio_served:
                if _gen_key in st.session_state and Path(st.session_state[_gen_key]).exists():
                    st.audio(st.session_state[_gen_key])
                elif st.button(_audio_btn_label, key=f"play_{selected_asha}_{lang}"):
                    with st.spinner(translate_ui("Generating audio briefing...", lang)):
                        try:
                            from stage5_voice import generate_voice_note
                            _gen_path = generate_voice_note(_briefing_text_for_audio, lang)
                            if _gen_path and Path(_gen_path).exists():
                                st.audio(_gen_path)
                                st.session_state[_gen_key] = _gen_path
                            else:
                                st.warning(translate_ui(
                                    "Audio generation requires SPEECH_KEY in .env", lang
                                ))
                        except Exception as _ae:
                            st.warning(f"Audio error: {_ae}")
        else:
            st.info(translate_ui(
                "No briefing available for this ASHA worker. Run the pipeline to generate today's briefing.",
                lang,
            ))

        st.divider()

        RED_FLAGS = [
            ("confined_to_bed",              translate_ui("Patient confined to bed", lang)),
            ("breathlessness",               translate_ui("Breathlessness at rest or on walking", lang)),
            ("severe_pain_chest_or_abdomen", translate_ui("Severe chest or abdominal pain", lang)),
            ("altered_consciousness",        translate_ui("Altered consciousness / convulsions / limb weakness", lang)),
            ("haemoptysis_one_cup",          translate_ui("Coughing out blood ≥ 1 cup", lang)),
            ("recurrent_vomiting_diarrhoea", translate_ui("Recurrent vomiting or diarrhoea", lang)),
            ("adr_symptoms",                 translate_ui("Symptoms of Adverse Drug Reactions", lang)),
        ]

        for p in asha_patients:
            pid       = p["patient_id"]
            tier      = p.get("risk_level", "LOW")
            score     = p.get("risk_score", 0)
            week      = p.get("treatment_week", "?")
            icon      = "🔴" if tier == "HIGH" else ("🟡" if tier == "MEDIUM" else "🟢")
            submitted = st.session_state.asha_updates.get(pid)

            with st.expander(
                f"{icon} {pid} — {translate_ui('Week', lang)} {week} — "
                f"{translate_ui('Score', lang)}: {score:.3f}"
                + (" ✓" if submitted else ""),
                expanded=(tier == "HIGH" and not submitted),
            ):
                if submitted:
                    labels = {
                        "done":            translate_ui("✅ Dose confirmed", lang),
                        "could_not_visit": translate_ui("❌ Could not visit", lang),
                        "issue":           translate_ui("⚠️ Issue flagged", lang),
                    }
                    st.success(labels.get(submitted["action"], submitted["action"]))
                    if submitted.get("red_flags"):
                        st.error(translate_ui(
                            f"Red flags reported: {len(submitted['red_flags'])}", lang
                        ))
                else:
                    factors = list(p.get("top_factors", {}).keys())
                    if factors:
                        st.caption(translate_ui(f"Main risk factor: {factors[0]}", lang))

                    st.markdown(f"**{translate_ui('Step 1 — What happened today?', lang)}**")
                    draft_key = f"draft_{pid}"
                    b1, b2, b3 = st.columns(3)
                    if b1.button(translate_ui("✅ Dose given", lang), key=f"done_{pid}"):
                        st.session_state[draft_key] = "done"; st.rerun()
                    if b2.button(translate_ui("❌ Could not visit", lang), key=f"miss_{pid}"):
                        st.session_state[draft_key] = "could_not_visit"; st.rerun()
                    if b3.button(translate_ui("⚠️ Flag for CHO", lang), key=f"issue_{pid}"):
                        st.session_state[draft_key] = "issue"; st.rerun()

                    st.markdown(f"**{translate_ui('Step 2 — Check for red flags (Box 4.1):', lang)}**")
                    _rf_cols = st.columns(2)
                    active_flags = []
                    for i, (flag_key, flag_label) in enumerate(RED_FLAGS):
                        if _rf_cols[i % 2].checkbox(flag_label, key=f"rf_{pid}_{flag_key}"):
                            active_flags.append(flag_key)
                    if active_flags:
                        st.error(translate_ui(
                            f"⚠️ {len(active_flags)} red flag(s) detected — CHO will be alerted.", lang
                        ))

                    note = st.text_area(
                        translate_ui("Optional note", lang), key=f"note_{pid}", height=60,
                        label_visibility="collapsed",
                        placeholder=translate_ui("e.g. Patient complained of side effects...", lang),
                    )

                    draft_action = st.session_state.get(draft_key)
                    if draft_action:
                        sub1, sub2 = st.columns([1, 2])
                        if sub1.button(translate_ui("📤 Submit", lang),
                                       key=f"submit_{pid}", type="primary"):
                            _reply_event(selected_asha, pid, draft_action, free_text=note.strip())
                            if active_flags:
                                try:
                                    from stage1_nlp import publish_red_flag_alert, get_eventhub_producer
                                    _producer = get_eventhub_producer()
                                    _p_rec    = next((q for q in patients if q["patient_id"] == pid), {})
                                    _cho_id   = (_p_rec.get("operational") or {}).get("cho_id", "CHO-UNKNOWN")
                                    publish_red_flag_alert(
                                        _producer, pid, selected_asha, _cho_id, active_flags, _p_rec
                                    )
                                    log_graph_activity("red_flag_alert", pid,
                                        f"🚨 Alert → CHO {_cho_id} | flags: {active_flags}")
                                    st.error(translate_ui(
                                        f"🚨 Red flag alert sent to CHO for {pid}", lang
                                    ))
                                    # Immediately rescore so the updated flags feed into
                                    # the BBN (adr_symptoms log-odds) and TGN event state.
                                    try:
                                        _rf_result = rescore_patient_locally(
                                            pid, "issue", active_flags=active_flags
                                        )
                                        if _rf_result.get("old_tier") != _rf_result.get("new_tier"):
                                            st.toast(
                                                f"⚡ {pid}: {_rf_result['old_tier']} → {_rf_result['new_tier']}",
                                                icon="🔄",
                                            )
                                    except Exception:
                                        pass
                                except Exception as _rfe:
                                    st.warning(f"Alert could not be sent: {_rfe}")
                            st.session_state.asha_updates[pid] = {
                                "action": draft_action, "note": note.strip(),
                                "red_flags": active_flags,
                            }
                            st.session_state.pop(draft_key, None)
                            st.rerun()
                        if sub2.button(translate_ui("✏️ Change", lang), key=f"clear_{pid}"):
                            st.session_state[draft_key] = None; st.rerun()
                    else:
                        st.caption(translate_ui("Select an outcome above to enable submission.", lang))

                    # ── Add Contact ───────────────────────────────────────────
                    with st.expander(translate_ui("➕ Add household/workplace contact", lang)):
                        _ct_name = st.text_input(
                            translate_ui("Contact name", lang), key=f"ct_name_{pid}"
                        )
                        _ct_rel  = st.selectbox(
                            translate_ui("Relation to patient", lang),
                            ["household", "workplace", "community"],
                            key=f"ct_rel_{pid}",
                        )
                        _ct_col1, _ct_col2 = st.columns(2)
                        _ct_age  = _ct_col1.number_input(
                            translate_ui("Age", lang), min_value=0, max_value=120,
                            value=30, key=f"ct_age_{pid}"
                        )
                        _ct_gender = _ct_col2.selectbox(
                            translate_ui("Gender", lang),
                            ["Male", "Female", "Other"],
                            key=f"ct_gender_{pid}",
                        )
                        if st.button(
                            translate_ui("💾 Save contact", lang),
                            key=f"ct_save_{pid}",
                        ):
                            if _ct_name.strip():
                                _contact_entry = {
                                    "name":              _ct_name.strip(),
                                    "rel":               _ct_rel,
                                    "relationship_type": _ct_rel,
                                    "age":               int(_ct_age),
                                    "gender":            _ct_gender,
                                    "screened":          False,
                                    "vulnerability_score": min(1.0, round(
                                        (1.0 if _ct_rel == "household" else 0.6) *
                                        (1.2 if int(_ct_age) > 60 or int(_ct_age) < 5 else 1.0),
                                        2
                                    )),
                                }
                                _json_path = Path("nikshay_scored_dataset.json")
                                if not _json_path.exists():
                                    _json_path = Path("data/nikshay_scored_dataset.json")
                                if _json_path.exists():
                                    with open(_json_path, encoding="utf-8") as _jf:
                                        _all_p = json.load(_jf)
                                    for _ep in _all_p:
                                        if _ep["patient_id"] == pid:
                                            _net = _ep.get("contact_network", [])
                                            _net.append(_contact_entry)
                                            _ep["contact_network"] = _net
                                            break
                                    with open(_json_path, "w", encoding="utf-8") as _jf:
                                        json.dump(_all_p, _jf, indent=2)
                                log_graph_activity(
                                    "contact_added", pid,
                                    f"{_ct_name.strip()} ({_ct_rel}, {_ct_age}y)",
                                )
                                st.success(translate_ui(
                                    f"Contact '{_ct_name.strip()}' added.", lang
                                ))
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.warning(translate_ui("Please enter a contact name.", lang))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHO CLINICAL DESK
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("🏥 CHO Clinical Desk")
    st.caption("Monthly clinical assessment (Annexure 1) and contact screening.")

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        rf_alerts  = load_json_file("data/red_flag_alerts.json")
        pending_rf = [a for a in rf_alerts
                      if a.get("status") == "pending" and a.get("recipient") == "CHO"]
        if pending_rf:
            st.error(f"🚨 **{len(pending_rf)} pending red flag alert(s) from ASHA workers**")
            for alert in pending_rf[:10]:
                with st.expander(
                    f"🚨 {alert['patient_id']} — ASHA: {alert['asha_id']} — "
                    f"{utc_to_local(alert['timestamp'])}", expanded=True
                ):
                    for flag in alert.get("active_flags", []):
                        st.markdown(f"  🔴 {flag.replace('_', ' ').title()}")
        else:
            st.success("✅ No pending red flag alerts.")

        st.divider()
        _cho_col1, _cho_col2 = st.columns([1, 2])
        with _cho_col1:
            tier_filter = st.selectbox(
                "Filter by risk tier", ["HIGH", "MEDIUM", "LOW"], index=0, key="cho_tier"
            )
        with _cho_col2:
            _cho_search = st.text_input(
                "Search patient", placeholder="Patient ID or name...", key="cho_search"
            )

        cho_patients = sorted(
            [p for p in patients if p.get("risk_level") == tier_filter],
            key=lambda x: x.get("rank_score", x.get("risk_score", 0)), reverse=True,
        )
        if _cho_search:
            _q = _cho_search.strip().lower()
            cho_patients = [
                p for p in cho_patients
                if _q in p.get("patient_id", "").lower()
                or _q in (p.get("name") or p.get("demographics", {}).get("name", "")).lower()
            ]
        st.info(f"**{len(cho_patients)} {tier_filter} risk patients**"
                + (f" matching '{_cho_search}'" if _cho_search else ""))

        for _cp in cho_patients[:30]:
            _cpid    = _cp["patient_id"]
            _week    = _cp.get("treatment_week", 0)
            _rbs     = _cp.get("records_b", [])
            _last_rb = _rbs[-1] if _rbs else {}
            _total_w = _cp.get("total_treatment_weeks", 26)
            _month   = max(1, _week // 4)
            _tier    = _cp.get("risk_level", "LOW")
            _icon    = "🔴" if _tier == "HIGH" else ("🟡" if _tier == "MEDIUM" else "🟢")
            _asha_id = (_cp.get("operational") or {}).get("asha_id", "—")

            with st.expander(
                f"{_icon} {_cpid} — ASHA: {_asha_id} — Week {_week}/{_total_w} — "
                f"Last: Month {_last_rb.get('month', 'None')}",
                expanded=False,
            ):
                if _last_rb:
                    _v = _last_rb.get("vitals", {})
                    _vc1, _vc2, _vc3, _vc4 = st.columns(4)
                    _vc1.metric("Weight", f"{_v.get('weight_kg','?')} kg")
                    _vc2.metric("SpO2",   f"{_v.get('spo2_percent','?')}%",
                                delta="⚠" if _v.get("spo2_flag") else None)
                    _vc3.metric("RR",     f"{_v.get('respiratory_rate','?')} /min",
                                delta="⚠" if _v.get("respiratory_rate_flag") else None)
                    _vc4.metric("Pulse",  f"{_v.get('pulse_rate','?')} /min",
                                delta="⚠" if _v.get("pulse_flag") else None)

                st.markdown("---")
                st.markdown(f"**📋 Annexure 1 — Month {_month} Assessment**")

                # Red flags
                st.markdown("**Red flag criteria:**")
                _rf1c, _rf2c = st.columns(2)
                _rf_items = [
                    ("confined_to_bed",              "Patient confined to bed"),
                    ("breathlessness",               "Breathlessness at rest / 10-15 feet walk"),
                    ("severe_pain",                  "Severe chest / abdominal pain"),
                    ("altered_consciousness",        "Altered consciousness / convulsions"),
                    ("haemoptysis_one_cup",          "Coughing blood ≥ 1 cup"),
                    ("recurrent_vomiting_diarrhoea", "Recurrent vomiting / diarrhoea"),
                    ("adr_symptoms",                 "ADR symptoms"),
                ]
                _rf_vals = {}
                for i, (k, lbl) in enumerate(_rf_items):
                    col = _rf1c if i % 2 == 0 else _rf2c
                    _rf_vals[k] = col.checkbox(lbl, key=f"cho_rf_{_cpid}_{k}")
                any_rf = any(_rf_vals.values())
                if any_rf:
                    st.warning("⚠️ Red flag identified")
                _mo_assessed = st.checkbox("Patient assessed by MO?",
                                           key=f"cho_mo_{_cpid}") if any_rf else False

                # Vitals
                st.markdown("**Vitals:**")
                _cc1, _cc2, _cc3 = st.columns(3)
                _spo2  = _cc1.number_input("SpO2 (%)",     50, 100, 97, key=f"cho_spo2_{_cpid}")
                _rr    = _cc2.number_input("RR /min",       5,  60, 18, key=f"cho_rr_{_cpid}")
                _bps   = _cc3.number_input("Systolic BP",  50, 220, 120, key=f"cho_bps_{_cpid}")
                _cc4, _cc5, _cc6 = st.columns(3)
                _bpd   = _cc4.number_input("Diastolic BP", 30, 140, 80, key=f"cho_bpd_{_cpid}")
                _pulse = _cc5.number_input("Pulse /min",   20, 200, 80, key=f"cho_pulse_{_cpid}")
                _temp  = _cc6.number_input("Temp °C", 35.0, 42.0, 37.0, step=0.1,
                                           key=f"cho_temp_{_cpid}")
                _cc7, _cc8 = st.columns(2)
                _wt = _cc7.number_input("Weight (kg)", 20.0, 150.0, 55.0, step=0.1,
                                        key=f"cho_wt_{_cpid}")
                _ht = _cc8.number_input("Height (cm)", 100.0, 220.0, 162.0, step=0.1,
                                        key=f"cho_ht_{_cpid}")
                _bmi_calc = round(_wt / ((_ht / 100) ** 2), 1) if _ht > 0 else 0
                st.caption(f"BMI: {_bmi_calc} kg/m²")

                # Clinical signs
                st.markdown("**Clinical signs:**")
                _cs1, _cs2, _cs3, _cs4 = st.columns(4)
                _cant_stand   = _cs1.checkbox("Unable to stand",   key=f"cho_stand_{_cpid}")
                _pedal_oedema = _cs2.checkbox("Pedal oedema",      key=f"cho_oed_{_cpid}")
                _icterus      = _cs3.checkbox("Icterus",           key=f"cho_ict_{_cpid}")
                _haemoptysis  = _cs4.checkbox("Haemoptysis",       key=f"cho_haem_{_cpid}")

                # Investigations
                st.markdown("**Investigations:**")
                _ic1, _ic2 = st.columns(2)
                _hb  = _ic1.number_input("Hb (g/dl)",   0.0, 20.0, 11.0, step=0.1,
                                         key=f"cho_hb_{_cpid}")
                _rbs_val = _ic2.number_input("RBS (mg/dl)", 50, 600, 100,
                                             key=f"cho_rbs_{_cpid}")
                _ic3, _ic4, _ic5, _ic6 = st.columns(4)
                _cxr_done = _ic3.checkbox("CXR done",  key=f"cho_cxr_{_cpid}")
                _lft_done = _ic4.checkbox("LFT done",  key=f"cho_lft_{_cpid}")
                _kft_done = _ic5.checkbox("KFT done",  key=f"cho_kft_{_cpid}")
                _pft_done = _ic6.checkbox("PFT done",  key=f"cho_pft_{_cpid}")

                # ADR
                st.markdown("**ADR:**")
                _adr_c1, _adr_c2 = st.columns(2)
                _adr_grade = _adr_c1.selectbox(
                    "ADR grade", [0, 1, 2, 3, 4], key=f"cho_adr_{_cpid}",
                    format_func=lambda x: f"Grade {x}" if x > 0 else "None"
                )
                _adr_type = _adr_c2.selectbox(
                    "ADR type",
                    ["None", "peripheral_neuropathy", "nausea_vomiting",
                     "hepatotoxicity", "skin_rash", "visual_disturbance", "joint_pain"],
                    key=f"cho_adr_type_{_cpid}",
                ) if _adr_grade > 0 else "None"

                # Programme
                st.markdown("**Programme:**")
                _pr1, _pr2 = st.columns(2)
                _nikshay_divas = _pr1.checkbox("Nikshay Divas attended", value=True,
                                               key=f"cho_nd_{_cpid}")
                _npy_received  = _pr2.checkbox("NPY benefit received",   value=True,
                                               key=f"cho_npy_{_cpid}")
                _comorbid_note  = st.text_input("Other comorbidities",     key=f"cho_comorbid_{_cpid}")
                _diagnosis_note = st.text_input("Provisional/final diagnosis", key=f"cho_diag_{_cpid}")
                _clin_notes     = st.text_area("Clinician's notes", key=f"cho_notes_{_cpid}",
                                               height=60)

                # Management Plan
                st.markdown("**Management Plan (tick one):**")
                _mgmt = st.radio(
                    "Management decision",
                    ["ambulatory_care", "referral_to_higher_centre", "referral_for_hospitalisation"],
                    format_func=lambda x: {
                        "ambulatory_care":              "☐ Ambulatory care",
                        "referral_to_higher_centre":    "☐ Referral to higher centre (PHC/CHC)",
                        "referral_for_hospitalisation": "☐ Referral for hospitalisation (SEVERE)",
                    }.get(x, x),
                    key=f"cho_mgmt_{_cpid}",
                )
                _bp_flag = (_bps < 90 or _bps >= 140 or _bpd < 60 or _bpd >= 90)

                if st.button("💾 Submit Assessment", key=f"cho_submit_{_cpid}", type="primary"):
                    _base_wt  = (_cp.get("baseline_clinical") or {}).get("weight_kg", _wt)
                    _wt_delta = round(_wt - _base_wt, 1)
                    _new_rb = {
                        "month": _month,
                        "assessment_date": str(pd.Timestamp.now().date()),
                        "treatment_week":  _week,
                        "phase": "Intensive" if _week <= 8 else "Continuation",
                        "red_flags": {**_rf_vals,
                                      "any_red_flag_positive": any_rf,
                                      "mo_assessment_done": _mo_assessed},
                        "vitals": {
                            "weight_kg": _wt, "bmi": _bmi_calc,
                            "weight_delta_kg": _wt_delta,
                            "spo2_percent": _spo2, "spo2_flag": _spo2 < 94,
                            "respiratory_rate": _rr, "respiratory_rate_flag": _rr > 24,
                            "bp_systolic": _bps, "bp_diastolic": _bpd, "bp_flag": _bp_flag,
                            "pulse_rate": _pulse, "pulse_flag": (_pulse > 120 or _pulse < 60),
                            "temperature_celsius": _temp,
                        },
                        "clinical_signs": {
                            "unable_to_stand_without_support": _cant_stand,
                            "pedal_oedema": _pedal_oedema,
                            "icterus": _icterus,
                            "haemoptysis": _haemoptysis,
                        },
                        "investigations": {
                            "hb_gdl": _hb, "rbs_mgdl": _rbs_val,
                            "chest_xray_done": _cxr_done,
                            "lft_done": _lft_done, "kft_done": _kft_done, "pft_done": _pft_done,
                        },
                        "adr": {
                            "reported": _adr_grade > 0,
                            "type": _adr_type if _adr_grade > 0 else None,
                            "grade": _adr_grade,
                        },
                        "programme": {
                            "nikshay_divas_attended": _nikshay_divas,
                            "npy_benefit_received": _npy_received,
                            "management_decision": _mgmt,
                        },
                        "comorbidities_note": _comorbid_note,
                        "diagnosis_note": _diagnosis_note,
                        "clinician_notes": _clin_notes,
                    }

                    _rescore = rescore_patient_locally(_cpid, "cho_assessment", record_b=_new_rb)
                    log_graph_activity("cho_assessment", _cpid,
                        f"Annexure 1 submitted | {_mgmt} | "
                        f"{_rescore.get('old_score','?')}→{_rescore.get('new_score','?')}")
                    st.cache_data.clear()

                    if _mgmt in ("referral_to_higher_centre", "referral_for_hospitalisation"):
                        try:
                            from stage1_nlp import publish_mo_alert
                            _cho_id_ref = (_cp.get("operational") or {}).get("cho_id", "CHO-UNKNOWN")
                            publish_mo_alert(_cpid, _cho_id_ref, _mgmt, clinical_summary={
                                "spo2": _spo2, "respiratory_rate": _rr,
                                "bp_systolic": _bps, "pulse_rate": _pulse,
                                "bmi": _bmi_calc, "weight_kg": _wt, "adr_grade": _adr_grade,
                            })
                            st.warning(f"📨 MO alert fired — {_cpid} referred to PHC/CHC.")
                        except Exception as _moe:
                            st.warning(f"MO alert error: {_moe}")

                    if _rescore.get("changed"):
                        st.success(
                            f"✅ Saved. Tier: {_rescore['old_tier']} → **{_rescore['new_tier']}**"
                        )
                    else:
                        st.success(
                            f"✅ Saved. Score: {_rescore.get('old_score','?')} → "
                            f"{_rescore.get('new_score','?')}"
                        )
                    st.rerun()

                # Contact screening
                _contacts   = _cp.get("contact_network", [])
                _unscreened = [c for c in _contacts if not c.get("screened", False)]
                if _unscreened:
                    st.markdown("---")
                    st.markdown(f"**Contacts due for screening ({len(_unscreened)}):**")
                    st.dataframe(pd.DataFrame([{
                        "Name":         c["name"],
                        "Age":          c.get("age", "?"),
                        "Relationship": c.get("rel") or c.get("relationship_type", "?"),
                        "Screened":     "✅" if c.get("screened") else "❌",
                    } for c in _contacts]), use_container_width=True, hide_index=True)

                    _contact_name = st.selectbox(
                        "Mark as screened",
                        ["— select —"] + [c["name"] for c in _unscreened],
                        key=f"cho_contact_{_cpid}",
                    )
                    _sym_pos = st.checkbox("Symptom positive", key=f"cho_sym_{_cpid}")
                    if _contact_name != "— select —":
                        if st.button(f"✅ Mark {_contact_name} screened",
                                     key=f"cho_screen_{_cpid}_{_contact_name.replace(' ','_')}"):
                            try:
                                from stage1_nlp import get_eventhub_producer, writeback_contact_screened
                                _prod3 = get_eventhub_producer()
                                _gc3   = None
                                try:
                                    from cosmos_client import get_client, health_check
                                    if health_check(): _gc3 = get_client()
                                except Exception:
                                    pass
                                if _gc3:
                                    writeback_contact_screened(
                                        _gc3, _prod3, _cpid, _contact_name,
                                        (_cp.get("operational") or {}).get("asha_id", "")
                                    )
                                log_graph_activity("contact_screened", _cpid,
                                    f"Contact {_contact_name} | sym_pos={_sym_pos}")
                                st.session_state.screened_contacts.add(_contact_name)
                                st.success(f"✅ {_contact_name} marked screened.")
                                st.rerun()
                            except Exception as _ce:
                                st.session_state.screened_contacts.add(_contact_name)
                                st.success(f"✅ {_contact_name} marked locally.")
                                st.rerun()
                else:
                    st.success("✅ All contacts screened.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MO ASSESSMENT (Annexure 2)
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("🩺 MO Assessment — Comprehensive DTC Assessment (Annexure 2)")
    st.caption("For triage-positive patients referred from CHO/ASHA (Annexure 2, National Guidance 2025).")

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        mo_alerts  = load_json_file("data/mo_alerts.json")
        pending_mo = [a for a in mo_alerts if a.get("status") == "pending"]

        if pending_mo:
            st.warning(f"📨 **{len(pending_mo)} patient(s) referred for MO assessment**")
        else:
            st.success("✅ No pending referrals.")

        for _ma in pending_mo[:20]:
            _mpid   = _ma["patient_id"]
            _mp_rec = next((p for p in patients if p["patient_id"] == _mpid), {})
            _m_diag = _mp_rec.get("diagnosis") or _mp_rec.get("clinical") or {}
            _m_bcl  = _mp_rec.get("baseline_clinical") or {}

            _sev_path = Path("data/severity_classifications.json")
            _existing_sev = {}
            if _sev_path.exists():
                try:
                    _sl = json.load(open(_sev_path))
                    _existing_sev = next((s for s in _sl if s["patient_id"] == _mpid), {})
                except Exception:
                    pass

            with st.expander(
                f"{'✅' if _existing_sev else '📨'} {_mpid} — CHO: {_ma.get('cho_id','?')} | "
                f"{_ma['management_decision'].replace('_',' ').title()} | {utc_to_local(_ma['timestamp'])}",
                expanded=not bool(_existing_sev),
            ):
                if _existing_sev:
                    st.success(f"✅ Classified: **{_existing_sev.get('classification','').replace('_',' ').title()}** "
                               f"({utc_to_local(_existing_sev.get('timestamp',''), fmt='%d %b %Y')})")
                    if _existing_sev.get("severe"):
                        st.error("🏥 DTO notified for hospital referral.")
                    continue

                # Part A — pre-filled
                st.markdown("**Part A — Referring facility details**")
                _pa1, _pa2, _pa3 = st.columns(3)
                _pa1.text_input("District",   value=_mp_rec.get("district", "Chennai"),
                                disabled=True, key=f"mo_dist_{_mpid}")
                _pa2.text_input("Nikshay ID", value=_mpid, disabled=True,
                                key=f"mo_nid_{_mpid}")
                _pa3.text_input("Regimen",    value=_m_diag.get("regimen", "?"),
                                disabled=True, key=f"mo_reg_{_mpid}")
                _pa4, _pa5 = st.columns(2)
                _pa4.text_input("Date of diagnosis",
                                value=_mp_rec.get("date_of_diagnosis", "?"),
                                disabled=True, key=f"mo_dod_{_mpid}")
                _pa5.text_input("Date of referral",
                                value=str(pd.Timestamp.now().date()),
                                disabled=True, key=f"mo_dor_{_mpid}")
                _nodal_name = st.text_input("Nodal physician name", key=f"mo_nodal_{_mpid}")
                _facility   = st.text_input("Referral care facility", key=f"mo_fac_{_mpid}")

                st.markdown("---")
                st.markdown("**Part B — Comprehensive assessment**")

                _mb1, _mb2 = st.columns(2)
                _haem_hist = _mb1.selectbox("History of haemoptysis", ["No", "Yes"],
                                            key=f"mo_haem_{_mpid}")
                _gen_cond  = _mb2.selectbox(
                    "General condition",
                    ["Able to walk", "Not able to walk but conscious/oriented",
                     "Conscious not oriented", "Drowsy"],
                    key=f"mo_gc_{_mpid}",
                )

                _mb3, _mb4, _mb5 = st.columns(3)
                _m_wt  = _mb3.number_input("Weight (kg)", 20.0, 150.0,
                                           float(_m_bcl.get("weight_kg", 55.0)),
                                           step=0.1, key=f"mo_wt_{_mpid}")
                _m_ht  = _mb4.number_input("Height (cm)", 100.0, 220.0,
                                           float(_m_bcl.get("height_cm", 162.0)),
                                           step=0.1, key=f"mo_ht_{_mpid}")
                _m_bmi = round(_m_wt / ((_m_ht / 100) ** 2), 1) if _m_ht > 0 else 0
                _mb5.metric("BMI", f"{_m_bmi} kg/m²")
                _muac = st.number_input("MUAC (cm)", 0.0, 40.0, 0.0, step=0.1,
                                        key=f"mo_muac_{_mpid}")

                _un1, _un2 = st.columns(2)
                _sev_undernut  = _un1.checkbox("Severe undernutrition",      key=f"mo_sun_{_mpid}")
                _vsev_undernut = _un2.checkbox("Very severe undernutrition", key=f"mo_vsun_{_mpid}")

                _addictions = st.text_area(
                    "Addiction history (CAGE / Fagerstrom)",
                    key=f"mo_add_{_mpid}", height=50
                )
                _chronic = st.text_area(
                    "Chronic conditions and occupational lung diseases",
                    key=f"mo_chronic_{_mpid}", height=50
                )

                st.markdown("**Vitals:**")
                _mv1, _mv2, _mv3, _mv4 = st.columns(4)
                _m_pulse = _mv1.number_input("Pulse /min",  20, 200, 80,  key=f"mo_pulse_{_mpid}")
                _m_temp  = _mv2.number_input("Temp °C", 35.0, 42.0, 37.0, step=0.1,
                                             key=f"mo_temp_{_mpid}")
                _m_rr   = _mv3.number_input("RR /min",   5,  60, 18,  key=f"mo_rr_{_mpid}")
                _m_spo2 = _mv4.number_input("SpO2 (%)", 50, 100, 97,  key=f"mo_spo2_{_mpid}")
                _mv5, _mv6, _mv7 = st.columns(3)
                _m_ict  = _mv5.checkbox("Icterus",      key=f"mo_ict_{_mpid}")
                _m_ped  = _mv6.checkbox("Pedal oedema", key=f"mo_ped_{_mpid}")
                _m_bpstr = _mv7.text_input("BP (mmHg)", value="120/80", key=f"mo_bp_{_mpid}")

                st.markdown("**Investigations:**")
                _mi1, _mi2, _mi3 = st.columns(3)
                _m_hb  = _mi1.number_input("Hb (g%)", 0.0, 20.0, 11.0, step=0.1,
                                           key=f"mo_hb_{_mpid}")
                _m_wbc = _mi2.number_input("WBC /mm³", 0, 100000, 8000, key=f"mo_wbc_{_mpid}")
                _m_rbs2 = _mi3.number_input("RBS mg/dl", 50, 600, 100, key=f"mo_rbs2_{_mpid}")
                _mi4, _mi5 = st.columns(2)
                _m_hiv = _mi4.selectbox("HIV status",
                                        ["Unknown", "Negative", "Positive"],
                                        key=f"mo_hiv_{_mpid}")
                _m_art = _mi5.selectbox("ART status",
                                        ["N/A", "On ART", "Not on ART"],
                                        key=f"mo_art_{_mpid}")
                _m_cxr  = st.text_area("CXR findings", key=f"mo_cxr_{_mpid}", height=40)
                _m_lft  = st.text_area("LFT",          key=f"mo_lft_{_mpid}", height=40)
                _m_rft  = st.text_area("RFT",          key=f"mo_rft_{_mpid}", height=40)
                _mi6, _mi7, _mi8 = st.columns(3)
                _m_ppps  = _mi6.number_input("PPBS mg/dl", 0, 1000, 0, key=f"mo_ppps_{_mpid}")
                _m_fps   = _mi7.number_input("FBS mg/dl",  0, 600,  0, key=f"mo_fps_{_mpid}")
                _m_hba1c = _mi8.number_input("HbA1c %", 0.0, 20.0, 0.0, step=0.1,
                                             key=f"mo_hba1c_{_mpid}")
                st.text_area("Further evaluation notes", key=f"mo_further_{_mpid}", height=40)

                st.markdown("---")
                st.markdown("**Severity classification:**")
                _confirmed_severe = st.radio(
                    "Confirmed severely ill?", ["No", "Yes"],
                    key=f"mo_sev_{_mpid}", horizontal=True,
                )
                _hdu_icu = "Not applicable"
                _severe_reasons = []

                if _confirmed_severe == "Yes":
                    _hdu_icu = st.radio(
                        "HDU/ICU required?", ["Yes", "No", "Not applicable"],
                        key=f"mo_hdu_{_mpid}", horizontal=True,
                    )
                    st.markdown("**Reasons for severe illness:**")
                    _sr1, _sr2 = st.columns(2)
                    _left_reasons = [
                        "Very severe undernutrition", "Alcohol addiction", "Other addiction(s)",
                        "Uncontrolled diabetes", "Severe Anaemia", "HIV related",
                        "Secondary Bacterial infection", "Liver complications",
                        "Renal complications", "Mental illness", "Haemoptysis",
                        "Severe COPD", "Restrictive lung diseases",
                        "Special populations (elderly/pregnancy)",
                    ]
                    _right_reasons = [
                        "Pneumothorax/Hydropneumothorax/Bilateral consolidation",
                        "Respiratory failure/ARDS/septic shock",
                        "Deep venous thrombosis and pulmonary embolism",
                        "IRIS",
                        "Complications to anti-TB treatment (ADR)",
                        "Extrapulmonary TB complications",
                        "High-grade fever",
                        "COVID-19 complications or post COVID-19 sequelae",
                        "Neurological complications",
                        "Others",
                    ]
                    for reason in _left_reasons:
                        if _sr1.checkbox(reason, key=f"mo_sr_{_mpid}_{reason[:12]}"):
                            _severe_reasons.append(reason)
                    for reason in _right_reasons:
                        if _sr2.checkbox(reason, key=f"mo_sr_{_mpid}_{reason[:12]}"):
                            _severe_reasons.append(reason)

                st.markdown("**Admission outcome:**")
                _adm1, _adm2 = st.columns(2)
                _adm_date  = _adm1.text_input("Date of admission", key=f"mo_adm_{_mpid}")
                _disc_date = _adm2.text_input("Date of discharge",  key=f"mo_disc_{_mpid}")
                _outcome = st.radio(
                    "Outcome",
                    ["Discharged for ambulatory DOT", "Death",
                     "Leave Against Medical Advice (LAMA)",
                     "Referred elsewhere and outcome unknown"],
                    key=f"mo_outcome_{_mpid}",
                )

                _final_cls = (
                    "referral_for_hospitalisation"
                    if _confirmed_severe == "Yes" and _outcome != "Discharged for ambulatory DOT"
                    else "referral_to_higher_centre"
                    if _confirmed_severe == "Yes"
                    else "ambulatory_care"
                )

                if st.button("💾 Save Annexure 2", key=f"mo_save_{_mpid}", type="primary"):
                    try:
                        _sys_bp = int(_m_bpstr.split("/")[0]) if "/" in _m_bpstr else 0
                    except Exception:
                        _sys_bp = 0

                    _clinical_params = {
                        "spo2": _m_spo2, "respiratory_rate": _m_rr,
                        "systolic_bp": _sys_bp, "pulse_rate": _m_pulse, "bmi": _m_bmi,
                        "hb": _m_hb, "hiv_status": _m_hiv, "art_status": _m_art,
                        "severe_undernutrition": _sev_undernut,
                        "very_severe_undernutrition": _vsev_undernut,
                        "hdu_icu_required": _hdu_icu,
                        "severe_reasons": _severe_reasons,
                        "general_condition": _gen_cond,
                        "haemoptysis_history": _haem_hist,
                        "rbs": _m_rbs2, "ppbs": _m_ppps, "fbs": _m_fps, "hba1c": _m_hba1c,
                        "admission_date": _adm_date, "discharge_date": _disc_date,
                        "outcome": _outcome,
                    }
                    try:
                        from stage1_nlp import save_severity_classification
                        save_severity_classification(
                            _mpid, _final_cls, _clinical_params,
                            classified_by=_nodal_name or "MO"
                        )
                        # Mark MO alert resolved
                        _mo_path = Path("data/mo_alerts.json")
                        if _mo_path.exists():
                            _mo_list = json.load(open(_mo_path))
                            for _a in _mo_list:
                                if _a["patient_id"] == _mpid and _a["status"] == "pending":
                                    _a["status"] = "assessed"
                            with open(_mo_path, "w") as _mf:
                                json.dump(_mo_list, _mf, indent=2)

                        log_graph_activity("mo_assessment", _mpid,
                            f"Annexure 2 | {_final_cls} | severe: {_confirmed_severe}")

                        if _final_cls == "referral_for_hospitalisation":
                            st.error(f"🏥 DTO notified — {_mpid} requires hospital admission.")
                        else:
                            st.success(f"✅ {_final_cls.replace('_',' ').title()}")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as _me:
                        st.error(f"Could not save: {_me}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DTO COMMAND VIEW
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("🏛️ District TB Officer — Command View")

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        import plotly.graph_objects as go

        asha_groups = defaultdict(list)
        for p in patients:
            asha_groups[(p.get("operational") or {}).get("asha_id", "?")].append(p)
        asha_loads = {}
        for aid, pts in asha_groups.items():
            n        = len(pts)
            high     = sum(1 for x in pts if x.get("risk_level") == "HIGH")
            avg_miss = sum((x.get("adherence") or {}).get("days_since_last_dose", 0)
                          for x in pts) / n
            asha_loads[aid] = round(
                min(1.0, (n / 15) * 0.4 + (avg_miss / 14) * 0.3 + (high / max(n, 1)) * 0.3), 3
            )

        overloaded       = sum(1 for v in asha_loads.values() if v > 0.60)
        unscreened_total = sum(
            1 for p in patients for c in p.get("contact_network", [])
            if not c.get("screened", False)
        )
        welfare_gap = sum(
            1 for p in patients
            if not (p.get("operational") or {}).get("welfare_enrolled", False)
        )
        live_n_high   = sum(1 for p in patients if p.get("risk_level") == "HIGH")
        live_n_medium = sum(1 for p in patients if p.get("risk_level") == "MEDIUM")
        live_n_low    = sum(1 for p in patients if p.get("risk_level") == "LOW")
        live_n_total  = len(patients)

        # ── District Pulse ────────────────────────────────────────────────────
        st.subheader("📊 District Pulse")

        dp1, dp2, dp3, dp4 = st.columns(4)
        dp1.metric("Total Patients",  live_n_total)
        dp2.metric("🔴 HIGH Risk",    live_n_high,
                   delta=f"{live_n_high / max(live_n_total, 1) * 100:.1f}%",
                   delta_color="inverse")
        dp3.metric("🟡 MEDIUM Risk",  live_n_medium,
                   delta=f"{live_n_medium / max(live_n_total, 1) * 100:.1f}%",
                   delta_color="off")
        dp4.metric("🟢 LOW Risk",     live_n_low,
                   delta=f"{live_n_low / max(live_n_total, 1) * 100:.1f}%",
                   delta_color="normal")

        # Risk distribution bar chart
        _risk_fig = go.Figure(go.Bar(
            x=["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
            y=[live_n_high, live_n_medium, live_n_low],
            marker_color=["#f87171", "#fb923c", "#4ade80"],
            text=[
                f"{live_n_high} ({live_n_high/max(live_n_total,1)*100:.0f}%)",
                f"{live_n_medium} ({live_n_medium/max(live_n_total,1)*100:.0f}%)",
                f"{live_n_low} ({live_n_low/max(live_n_total,1)*100:.0f}%)",
            ],
            textposition="outside",
        ))
        _risk_fig.update_layout(
            yaxis_title="Patients", height=220,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(_risk_fig, use_container_width=True)

        # Overloaded ASHA worker table
        st.markdown("**Overloaded ASHA Workers**")
        _overloaded_rows = []
        for _aid, _ls in asha_loads.items():
            _pts     = asha_groups[_aid]
            _n_pts   = len(_pts)
            _n_high  = sum(1 for _xp in _pts if _xp.get("risk_level") == "HIGH")
            _miss_rt = (
                sum(1 for _xp in _pts
                    if (_xp.get("adherence") or {}).get("days_since_last_dose", 0) >= 7)
                / max(_n_pts, 1)
            )
            if _ls > 0.70 or (_n_pts > 12 and _miss_rt > 0.30):
                _overloaded_rows.append({
                    "ASHA ID":         _aid,
                    "Patients":        _n_pts,
                    "HIGH Risk":       _n_high,
                    "Load Score":      round(_ls, 3),
                    "Missed Dose %":   f"{_miss_rt:.0%}",
                })
        if _overloaded_rows:
            st.dataframe(
                pd.DataFrame(_overloaded_rows).sort_values("Load Score", ascending=False),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success("No ASHA workers above overload threshold.")

        st.divider()

        # ── Quick KPI row (retained for at-a-glance summary) ─────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔴 HIGH Risk",           live_n_high)
        k2.metric("⚡ Overloaded ASHAs",    overloaded,         delta_color="inverse")
        k3.metric("👥 Unscreened Contacts", unscreened_total)
        k4.metric("💊 Welfare Gap",          welfare_gap,        delta_color="inverse")

        st.divider()
        st.subheader("🚨 Hospital Referral Alerts")
        st.caption("MO-confirmed severe cases only.")

        dto_alerts  = load_json_file("data/dto_alerts.json")
        pending_dto = [a for a in dto_alerts if a.get("status") == "pending"]

        if not pending_dto:
            st.success("✅ No pending hospital referral cases.")
        else:
            st.error(f"**{len(pending_dto)} patient(s) require hospital admission coordination.**")
            for _da in pending_dto:
                _dpid   = _da["patient_id"]
                with st.expander(
                    f"🚨 {_dpid} — {_da['classification'].replace('_',' ').title()} | "
                    f"MO: {_da.get('mo_classified_by','?')} | {utc_to_local(_da['timestamp'])}",
                    expanded=True,
                ):
                    _cp = _da.get("clinical_params", {})
                    if _cp:
                        _dc1, _dc2, _dc3 = st.columns(3)
                        _dc1.metric("SpO2",  f"{_cp.get('spo2','?')}%")
                        _dc2.metric("RR",    f"{_cp.get('respiratory_rate','?')} /min")
                        _dc3.metric("Pulse", f"{_cp.get('pulse_rate','?')} /min")
                        if _cp.get("severe_reasons"):
                            st.markdown("**Reasons:** " + ", ".join(_cp["severe_reasons"]))
                        if _cp.get("hdu_icu_required"):
                            st.markdown(f"**HDU/ICU:** {_cp['hdu_icu_required']}")

                    _transport = st.selectbox(
                        "Transport arrangement",
                        ["Select...", "108 Ambulance", "104 Mobile Health",
                         "Rogi Kalyan Samiti (RKS)", "PM-JAY transport", "Local arrangement"],
                        key=f"dto_transport_{_dpid}",
                    )
                    if st.button("✅ Acknowledge and initiate referral",
                                 key=f"dto_ack_{_dpid}", type="primary"):
                        try:
                            from datetime import datetime, timezone as _tz2
                            _dto_path = Path("data/dto_alerts.json")
                            _dto_list = json.load(open(_dto_path))
                            for _a in _dto_list:
                                if _a["patient_id"] == _dpid and _a["status"] == "pending":
                                    _a["status"]          = "acknowledged"
                                    _a["acknowledged_at"] = datetime.now(_tz2.utc).isoformat()
                                    _a["transport"]       = _transport
                            with open(_dto_path, "w") as _df:
                                json.dump(_dto_list, _df, indent=2)
                            log_graph_activity("dto_referral_ack", _dpid,
                                f"DTO acknowledged | {_transport}")
                            st.success(f"✅ Acknowledged. Arrange {_transport} for {_dpid}.")
                            st.rerun()
                        except Exception as _de:
                            st.error(f"Could not acknowledge: {_de}")

        st.divider()
        st.subheader("① ASHA Workload")
        load_rows = [{"ASHA ID": aid, "Load": s,
                      "HIGH": sum(1 for p in asha_groups[aid] if p.get("risk_level") == "HIGH"),
                      "Caseload": len(asha_groups[aid])}
                     for aid, s in asha_loads.items()]
        load_df = pd.DataFrame(load_rows).sort_values("Load", ascending=False)
        bar_colours = ["#f87171" if s > 0.70 else "#fb923c" if s > 0.50 else "#4ade80"
                       for s in load_df["Load"]]
        fig_load = go.Figure(go.Bar(
            x=load_df["Load"], y=load_df["ASHA ID"], orientation="h",
            marker_color=bar_colours,
            text=[f"{s:.2f}" for s in load_df["Load"]], textposition="outside",
        ))
        fig_load.add_vline(x=0.60, line_dash="dash", line_color="orange")
        fig_load.update_layout(
            xaxis=dict(range=[0, 1.15], title="Load Score"),
            height=max(280, len(load_df) * 32),
            margin=dict(l=20, r=60, t=10, b=30),
        )
        st.plotly_chart(fig_load, use_container_width=True)

        st.divider()
        st.subheader("② Welfare Gap — NPY")
        _wg_pids = [p["patient_id"] for p in patients
                    if not (p.get("operational") or {}).get("welfare_enrolled", False)]
        if _wg_pids:
            st.warning(f"{len(_wg_pids)} patients not enrolled in Nikshay Poshan Yojana.")
            st.dataframe(pd.DataFrame({"Patient ID": _wg_pids[:30]}),
                         use_container_width=True, hide_index=True)
        else:
            st.success("All patients enrolled in NPY.")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("🔍 Explainability & Audit")

    st.subheader("Recent Score Updates")
    _audit_path = Path("data/score_audit_log.json")
    if not _audit_path.exists():
        st.info("No audit log yet. Submit an update or run the pipeline.")
    else:
        try:
            with open(_audit_path) as _af:
                _audit_log = json.load(_af)
            _audit_rows = []
            for _e in _audit_log[:100]:
                _delta = _e.get("score_delta", 0)
                _audit_rows.append({
                    "Time":         _e.get("timestamp", "")[:19].replace("T", " "),
                    "Patient":      _e.get("patient_id", "?"),
                    "Trigger":      _e.get("trigger", "?"),
                    "Old Score":    round(_e.get("old_score", 0), 3),
                    "New Score":    round(_e.get("new_score", 0), 3),
                    "Change":       f"+{_delta:.3f}" if _delta >= 0 else f"{_delta:.3f}",
                    "Tier Changed": "✅" if _e.get("tier_changed") else "—",
                    "Reason":       _e.get("change_reason", "—"),
                })
            st.dataframe(pd.DataFrame(_audit_rows), use_container_width=True, hide_index=True)
        except Exception as _ae:
            st.error(f"Could not load audit log: {_ae}")

    st.divider()
    st.subheader("Patient Risk Explanations")
    if patients:
        _exp_pid = st.selectbox(
            "Select patient",
            [p["patient_id"] for p in sorted(
                patients, key=lambda x: x.get("risk_score", 0), reverse=True
            )],
            key="exp_pid_sel",
        )
        _exp_p = next((p for p in patients if p["patient_id"] == _exp_pid), {})
        if _exp_p:
            _ep1, _ep2, _ep3 = st.columns(3)
            _ep1.metric("Risk Score", round(_exp_p.get("risk_score", 0), 3))
            _ep2.metric("Risk Tier",  _exp_p.get("risk_level", "?"))
            _ep3.metric("Week",       _exp_p.get("treatment_week", "?"))
            factors = _exp_p.get("top_factors", {})
            if factors:
                st.markdown("**Top risk factors:**")
                for fname, for_val in factors.items():
                    st.markdown(f"- **{fname}** — odds ratio: {for_val:.2f}×")
            if _audit_path.exists():
                try:
                    with open(_audit_path) as _paf:
                        _plog = json.load(_paf)
                    _p_history = [e for e in _plog if e.get("patient_id") == _exp_pid][:10]
                    if _p_history:
                        st.markdown("**Score history:**")
                        st.dataframe(pd.DataFrame([{
                            "Time":    e.get("timestamp", "")[:16].replace("T", " "),
                            "Trigger": e.get("trigger", "?"),
                            "Score":   f"{e.get('old_score',0):.3f} → {e.get('new_score',0):.3f}",
                            "Change":  (f"+{e['score_delta']:.3f}" if e.get("score_delta", 0) >= 0
                                        else f"{e.get('score_delta', 0):.3f}"),
                            "Tier":    "✅" if e.get("tier_changed") else "—",
                            "Reason":  e.get("change_reason", "—"),
                        } for e in _p_history]),
                        use_container_width=True, hide_index=True)
                except Exception:
                    pass



# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TGN GRAPH
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("🕸️ TGN Risk Network")
    st.caption("Patient–ASHA–CHO–PHC graph. Node colour = risk tier.")

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        _g_tier        = st.selectbox("Show patients", ["ALL", "HIGH", "MEDIUM", "LOW"],
                                      key="graph_tier")
        _show_contacts = st.checkbox("Show contact nodes", value=False, key="show_contacts")
        _show_labels   = st.checkbox("Show node labels",   value=True,  key="show_labels")

        _graph_patients = patients if _g_tier == "ALL" else [
            p for p in patients if p.get("risk_level") == _g_tier
        ]

        _nodes, _edges, _node_ids = [], [], set()
        TIER_COLOR = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#1D9E75"}

        # PHC anchor
        _nodes.append({"id": "PHC_DISTRICT", "label": "PHC" if _show_labels else "",
                       "type": "phc", "color": "#2980B9", "size": 18})
        _node_ids.add("PHC_DISTRICT")

        for _gp in _graph_patients:
            _gpid = _gp["patient_id"]
            _tier = _gp.get("risk_level", "LOW")
            _op   = _gp.get("operational") or {}
            _nodes.append({
                "id": _gpid, "label": _gpid if _show_labels else "",
                "type": "patient", "color": TIER_COLOR.get(_tier, "#888780"),
                "size": max(7, int(_gp.get("risk_score", 0) * 24)),
                "tier": _tier, "score": round(_gp.get("risk_score", 0), 3),
                "week": _gp.get("treatment_week", 0),
            })
            _node_ids.add(_gpid)

            _aid = _op.get("asha_id", "")
            if _aid and _aid not in _node_ids:
                _load = _gp.get("asha_load_score", 0.3)
                _nodes.append({"id": _aid, "label": _aid if _show_labels else "",
                               "type": "asha", "color": "#7F77DD", "size": 12,
                               "load": _load, "overloaded": _load > 0.60})
                _node_ids.add(_aid)
                _edges.append({"source": _aid, "target": "PHC_DISTRICT",
                               "weight": 0.5, "type": "facility"})
            if _aid:
                _edges.append({"source": _gpid, "target": _aid,
                               "weight": max(0.2, 1 - _gp.get("asha_load_score", 0.3)),
                               "type": "supervised_by"})

            _cid = _op.get("cho_id", "")
            if _cid and _cid not in _node_ids:
                _nodes.append({"id": _cid, "label": _cid if _show_labels else "",
                               "type": "cho", "color": "#E67E22", "size": 14})
                _node_ids.add(_cid)
                _edges.append({"source": _cid, "target": "PHC_DISTRICT",
                               "weight": 0.8, "type": "facility"})
            if _cid:
                _rbs_list  = _gp.get("records_b", [])
                _last_month = max((_rb.get("month", 0) for _rb in _rbs_list), default=0)
                _edges.append({"source": _gpid, "target": _cid,
                               "weight": max(0.2, _last_month / 6.0),
                               "type": "assessed_by"})

            if _show_contacts:
                for _c in _gp.get("contact_network", [])[:3]:
                    _cname = f"CONTACT_{_c['name'].replace(' ','_').replace('.','')}"
                    if _cname not in _node_ids:
                        _screened = _c.get("screened", False)
                        _nodes.append({"id": _cname,
                                       "label": _c["name"][:10] if _show_labels else "",
                                       "type": "contact",
                                       "color": "#5DCAA5" if _screened else "#F0997B",
                                       "size": 5, "screened": _screened})
                        _node_ids.add(_cname)
                    _edges.append({"source": _gpid, "target": _cname,
                                   "weight": 0.3, "type": "contact"})

        _nodes_json = json.dumps(_nodes)
        _edges_json = json.dumps(_edges)

        _graph_html = f"""<!DOCTYPE html><html><head><style>
body{{margin:0;background:#1a1a2e;font-family:sans-serif;}}
svg{{width:100%;height:580px;}}
.tooltip{{position:absolute;background:rgba(20,20,40,0.92);color:#e0dfd5;
  padding:8px 12px;border-radius:6px;font-size:12px;pointer-events:none;
  display:none;border:1px solid #444;max-width:200px;}}
</style></head><body>
<div id="tooltip" class="tooltip"></div>
<svg id="graph"></svg>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const data={{nodes:{_nodes_json},edges:{_edges_json}}};
const svg=d3.select('#graph');
const W=svg.node().getBoundingClientRect().width||900,H=580;
svg.attr('viewBox',`0 0 ${{W}} ${{H}}`);
const sim=d3.forceSimulation(data.nodes)
  .force('link',d3.forceLink(data.edges).id(d=>d.id)
    .distance(d=>d.type==='facility'?180:d.type==='contact'?40:90)
    .strength(d=>d.type==='facility'?0.8:0.4))
  .force('charge',d3.forceManyBody().strength(d=>
    d.type==='phc'?-800:d.type==='cho'?-400:d.type==='asha'?-250:d.type==='patient'?-120:-60))
  .force('center',d3.forceCenter(W/2,H/2))
  .force('collision',d3.forceCollide().radius(d=>d.size+8))
  .alphaDecay(0.03).velocityDecay(0.4);
const phcN=data.nodes.find(n=>n.id==='PHC_DISTRICT');
if(phcN){{phcN.fx=W/2;phcN.fy=80;}}
const link=svg.append('g').selectAll('line').data(data.edges).join('line')
  .attr('stroke',d=>d.type==='facility'?'#2980B9':d.type==='assessed_by'?'#E67E22':
    d.type==='contact'?'#5DCAA5':'#7F77DD')
  .attr('stroke-opacity',d=>d.type==='contact'?0.3:0.5)
  .attr('stroke-width',d=>Math.max(0.8,(d.weight||0.3)*2.5));
const node=svg.append('g').selectAll('g').data(data.nodes).join('g')
  .call(d3.drag()
    .on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);if(d.id!=='PHC_DISTRICT'){{d.fx=null;d.fy=null;}}}}));
node.filter(d=>d.type==='phc').append('rect')
  .attr('x',-16).attr('y',-16).attr('width',32).attr('height',32)
  .attr('rx',4).attr('fill','#2980B9').attr('stroke','#1a5276').attr('stroke-width',2);
node.filter(d=>d.type==='cho').append('rect')
  .attr('x',-10).attr('y',-10).attr('width',20).attr('height',20)
  .attr('rx',3).attr('fill','#E67E22').attr('stroke','#a04000').attr('stroke-width',1.5);
node.filter(d=>d.type==='asha').append('rect')
  .attr('x',-9).attr('y',-9).attr('width',18).attr('height',18)
  .attr('transform','rotate(45)').attr('fill',d=>d.color)
  .attr('fill-opacity',d=>0.4+d.load*0.6).attr('stroke','#3C3489').attr('stroke-width',1.5);
node.filter(d=>d.type==='patient').append('circle')
  .attr('r',d=>d.size).attr('fill',d=>d.color).attr('fill-opacity',0.85)
  .attr('stroke',d=>d.tier==='HIGH'?'#A32D2D':d.tier==='MEDIUM'?'#854F0B':'#0F6E56')
  .attr('stroke-width',1.5);
node.filter(d=>d.type==='contact').append('circle')
  .attr('r',d=>d.size).attr('fill',d=>d.color).attr('fill-opacity',0.75)
  .attr('stroke','#0F6E56').attr('stroke-width',1);
node.filter(d=>d.overloaded).append('text').attr('dy',4).attr('text-anchor','middle')
  .attr('font-size',10).attr('fill','#EF9F27').text('⚠');
node.append('text')
  .attr('dy',d=>d.type==='phc'?-20:d.type==='cho'?-14:d.type==='asha'?-14:-d.size-3)
  .attr('text-anchor','middle').attr('font-size',d=>d.type==='phc'?11:9)
  .attr('fill','#c2c0b6').text(d=>d.label);
const tip=document.getElementById('tooltip');
node.on('mouseover',(e,d)=>{{
  let h='<b>'+d.id+'</b><br>';
  if(d.type==='phc')h+='PHC facility';
  else if(d.type==='cho')h+='Community Health Officer';
  else if(d.type==='asha')h+='ASHA | Load:'+d.load+(d.overloaded?'<br><b>⚠ Overloaded</b>':'');
  else if(d.type==='patient')h+='Tier:'+d.tier+'<br>Score:'+d.score+'<br>Week:'+d.week;
  else h+='Contact<br>Screened:'+(d.screened?'Yes':'No');
  tip.innerHTML=h;tip.style.display='block';
  tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY-20)+'px';
}}).on('mousemove',e=>{{
  tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY-20)+'px';
}}).on('mouseout',()=>{{tip.style.display='none';}});
sim.on('tick',()=>{{
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
      .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
}});
const leg=svg.append('g').attr('transform','translate(12,12)');
[['■ PHC','#2980B9'],['■ CHO','#E67E22'],['◆ ASHA','#7F77DD'],
 ['● HIGH','#E24B4A'],['● MEDIUM','#EF9F27'],['● LOW','#1D9E75'],
 ['● Contact (unscreened)','#F0997B'],['● Contact (screened)','#5DCAA5']
].forEach(([l,c],i)=>{{
  leg.append('circle').attr('cx',7).attr('cy',i*16+7).attr('r',5).attr('fill',c);
  leg.append('text').attr('x',17).attr('y',i*16+11).attr('font-size',10)
     .attr('fill','#c2c0b6').text(l);
}});
</script></body></html>"""

        st.components.v1.html(_graph_html, height=620, scrolling=False)

        _gcol1, _gcol2, _gcol3, _gcol4 = st.columns(4)
        _gcol1.metric("Patients", sum(1 for n in _nodes if n["type"] == "patient"))
        _gcol2.metric("ASHAs",    sum(1 for n in _nodes if n["type"] == "asha"))
        _gcol3.metric("CHOs",     sum(1 for n in _nodes if n["type"] == "cho"))
        _gcol4.metric("Edges",    len(_edges))

        with st.expander("📐 Graph schema", expanded=False):
            _nc, _ec = st.columns(2)
            with _nc:
                st.markdown("**Node types**")
                st.dataframe(pd.DataFrame([
                    {"Type": "Patient", "Shape": "Circle",          "Colour": "Risk tier"},
                    {"Type": "ASHA",    "Shape": "Diamond (purple)", "Colour": "Purple"},
                    {"Type": "CHO",     "Shape": "Square (orange)",  "Colour": "Orange"},
                    {"Type": "PHC",     "Shape": "Square (blue)",    "Colour": "Blue"},
                    {"Type": "Contact", "Shape": "Small circle",     "Colour": "Green/Salmon"},
                ]), use_container_width=True, hide_index=True)
            with _ec:
                st.markdown("**Edge types**")
                st.dataframe(pd.DataFrame([
                    {"Edge": "supervised_by", "Connects": "Patient → ASHA", "GATConv": "✅"},
                    {"Edge": "assessed_by",   "Connects": "Patient → CHO",  "GATConv": "✅"},
                    {"Edge": "registered_at", "Connects": "Patient → PHC",  "GATConv": "✅"},
                    {"Edge": "facility",      "Connects": "ASHA/CHO → PHC", "GATConv": "—"},
                    {"Edge": "contact",       "Connects": "Patient → Contact", "GATConv": "—"},
                ]), use_container_width=True, hide_index=True)
