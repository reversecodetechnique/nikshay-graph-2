"""
stage5_voice.py
===============
Stage 5: ASHA Communication & Voice Output

Translation and TTS use Azure native services exclusively:
  - Azure AI Translator  : English → Tamil / Telugu / Kannada / Bengali /
                           Hindi / Marathi / Gujarati
  - Azure AI Speech      : Neural TTS for all 7 languages (single SDK,
                           single resource, single .env key)

Bhashini has been removed entirely. No non-Azure dependencies.

Delivery: all ASHA updates come through the Streamlit dashboard
(app.py), not WhatsApp or SMS. This module generates the briefing
dict + audio file that the dashboard reads and plays with st.audio().

Azure services used:
  TRANSLATOR_KEY / TRANSLATOR_REGION  → Azure AI Translator (Text API v3)
  SPEECH_KEY     / SPEECH_REGION      → Azure AI Speech (Neural TTS)

Required .env keys:
    TRANSLATOR_KEY=<Translator resource → Keys and Endpoint → Key 1>
    TRANSLATOR_REGION=centralindia
    SPEECH_KEY=<Speech resource → Keys and Endpoint → Key 1>
    SPEECH_REGION=centralindia

Usage:
    from stage5_voice import format_morning_briefing, generate_voice_note
    from stage5_voice import process_asha_dashboard_reply
"""

import os
import json
import uuid
import tempfile
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE CONFIG
# Covers all 7 Indic languages used across ASHA worker regions.
# Azure AI Translator language codes: https://learn.microsoft.com/azure/ai-services/translator/language-support
# Azure AI Speech Neural voices:      https://learn.microsoft.com/azure/ai-services/speech-service/language-support
# ─────────────────────────────────────────────────────────────────────────────

# Fix 14/15: full Indian language support — all 22 scheduled languages + English.
# Translator codes: https://learn.microsoft.com/azure/ai-services/translator/language-support
# Speech voices:    https://learn.microsoft.com/azure/ai-services/speech-service/language-support
LANGUAGE_CONFIG = {
    "English":    ("en",  "en-IN",  "en-IN-NeerjaNeural"),
    "Hindi":      ("hi",  "hi-IN",  "hi-IN-SwaraNeural"),
    "Tamil":      ("ta",  "ta-IN",  "ta-IN-PallaviNeural"),
    "Telugu":     ("te",  "te-IN",  "te-IN-ShrutiNeural"),
    "Kannada":    ("kn",  "kn-IN",  "kn-IN-SapnaNeural"),
    "Malayalam":  ("ml",  "ml-IN",  "ml-IN-SobhanaNeural"),
    "Marathi":    ("mr",  "mr-IN",  "mr-IN-AarohiNeural"),
    "Gujarati":   ("gu",  "gu-IN",  "gu-IN-DhwaniNeural"),
    "Bengali":    ("bn",  "bn-IN",  "bn-IN-TanishaaNeural"),
    "Punjabi":    ("pa",  "pa-IN",  "pa-IN-OjasNeural"),
    "Odia":       ("or",  "or-IN",  "or-IN-SubhasiniNeural"),
    "Assamese":   ("as",  "as-IN",  "as-IN-YashicaNeural"),
    "Urdu":       ("ur",  "ur-IN",  "ur-IN-GulNeural"),
    "Maithili":   ("mai", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback to Hindi voice
    "Santali":    ("sat", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Kashmiri":   ("ks",  "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Nepali":     ("ne",  "ne-NP",  "ne-NP-HemkalaNeural"),
    "Sindhi":     ("sd",  "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Konkani":    ("gom", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Dogri":      ("doi", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Manipuri":   ("mni", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Bodo":       ("brx", "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
    "Sanskrit":   ("sa",  "hi-IN",  "hi-IN-SwaraNeural"),   # fallback
}

ALL_INDIAN_LANGUAGES = list(LANGUAGE_CONFIG.keys())


# ─────────────────────────────────────────────────────────────────────────────
# AZURE AI TRANSLATOR  (Text Translation API v3)
# ─────────────────────────────────────────────────────────────────────────────

TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"


def translate_text(text: str, target_language: str) -> str:
    """
    Translate English text to the ASHA worker's language using
    Azure AI Translator (Cognitive Services Text Translation v3).

    Requires .env:
        TRANSLATOR_KEY    — from Translator resource → Keys and Endpoint
        TRANSLATOR_REGION — e.g. centralindia

    Falls back to English if not configured or if target is English.
    """
    import requests

    if target_language == "English":
        return text

    lang_cfg = LANGUAGE_CONFIG.get(target_language)
    if not lang_cfg:
        print(f"  [Translator] Unsupported language '{target_language}' — using English")
        return text

    translator_code = lang_cfg[0]
    key             = os.getenv("TRANSLATOR_KEY")
    region          = os.getenv("TRANSLATOR_REGION", "centralindia")

    if not key:
        print("  [Translator] TRANSLATOR_KEY not set — using English text")
        return text

    try:
        url     = f"{TRANSLATOR_ENDPOINT}/translate"
        headers = {
            "Ocp-Apim-Subscription-Key":    key,
            "Ocp-Apim-Subscription-Region": region,
            "Content-Type":                 "application/json",
            "X-ClientTraceId":              str(uuid.uuid4()),
        }
        params   = {"api-version": "3.0", "from": "en", "to": translator_code}
        body     = [{"text": text}]

        resp = requests.post(url, headers=headers, params=params,
                             json=body, timeout=15)
        resp.raise_for_status()

        translated = resp.json()[0]["translations"][0]["text"]
        print(f"  [Translator] en → {target_language} ({translator_code}) — OK")
        return translated

    except Exception as e:
        print(f"  [Translator] Error: {e} — using English")
        return text


# ─────────────────────────────────────────────────────────────────────────────
# AZURE AI SPEECH  (Neural TTS — all 7 languages)
# ─────────────────────────────────────────────────────────────────────────────

def generate_voice_note(text: str, language: str) -> str | None:
    """
    Generate a speech audio file from text using Azure AI Speech Neural TTS.

    Supports all 7 Indic languages in LANGUAGE_CONFIG via a single
    Azure AI Speech resource and a single SDK call — no third-party
    TTS service required.

    Requires .env:
        SPEECH_KEY    — from Speech resource → Keys and Endpoint
        SPEECH_REGION — e.g. centralindia

    Returns:
        Path to generated .mp3 file, or None if TTS is not configured
        or synthesis fails.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        print("  [Azure TTS] azure-cognitiveservices-speech not installed. "
              "Run: pip install azure-cognitiveservices-speech")
        return None

    speech_key    = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION", "centralindia")

    if not speech_key:
        print("  [Azure TTS] SPEECH_KEY not set — skipping audio generation")
        return None

    lang_cfg = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["English"])
    _, locale, voice_name = lang_cfg

    try:
        config      = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        config.speech_synthesis_voice_name = voice_name

        _, output_path = tempfile.mkstemp(suffix=".mp3")
        audio_cfg   = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config,
                                                   audio_config=audio_cfg)

        result = synthesizer.speak_text_async(text).get()

        if result.reason.name == "SynthesizingAudioCompleted":
            print(f"  [Azure TTS] {language} ({voice_name}) → {output_path}")
            return output_path

        # Detailed error from CancellationDetails
        details = speechsdk.CancellationDetails(result)
        print(f"  [Azure TTS] Synthesis failed: {details.reason} — {details.error_details}")
        return None

    except Exception as e:
        print(f"  [Azure TTS] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MORNING BRIEFING FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_morning_briefing(visit_list: list, screening_list: list,
                             systemic_alerts: list, asha_id: str,
                             language: str = "English") -> dict:
    """
    Build the morning briefing dict consumed directly by the dashboard.

    Steps:
      1. Compose English briefing text from visit/screening lists
      2. Translate to ASHA worker's language via Azure AI Translator
      3. Generate voice note via Azure AI Speech Neural TTS
      4. Return dict with text, audio path, and per-patient visit_cards
         for the dashboard to render as interactive cards

    Returns dict with keys:
        asha_id, english_text, translated_text, audio_path,
        language, patient_count, visit_cards
    """
    # Priority patients only — HIGH risk first, cap at 3 for a concise briefing.
    priority_items = [i for i in visit_list if i.get("risk_level") == "HIGH"]
    if not priority_items:
        priority_items = visit_list  # fall back if no HIGH risk patients
    priority_items = priority_items[:3]

    lines = [f"Good morning. You have {len(priority_items)} priority patient(s) today."]

    visit_cards = []
    for item in priority_items:
        pid = item["patient_id"]
        exp = item["asha_explanation"]
        lines.append(f"Patient {pid}: {exp}")
        visit_cards.append({
            "rank":        item["rank"],
            "patient_id":  pid,
            "risk_level":  item["risk_level"],
            "risk_score":  item["risk_score"],
            "days_missed": item["days_missed"],
            "explanation": exp,
            "asha_id":     item["asha_id"],
            "block":       item["block"],
        })

    if screening_list:
        lines.append(
            f"You also have {min(len(screening_list), 2)} contact(s) to screen for TB symptoms."
        )

    lines.append("Please update each card after your visit.")

    english_text    = " ".join(lines)
    translated_text = translate_text(english_text, language)
    audio_path      = generate_voice_note(translated_text, language)

    return {
        "asha_id":         asha_id,
        "english_text":    english_text,
        "translated_text": translated_text,
        "audio_path":      audio_path,
        "language":        language,
        "patient_count":   len(visit_list[:5]),
        "visit_cards":     visit_cards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASHA DASHBOARD REPLY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_asha_dashboard_reply(gc, producer,
                                  action: str,
                                  patient_id: str,
                                  asha_id: str,
                                  free_text: str = "",
                                  contact_name: str = "") -> dict:
    """
    Called by the dashboard when the ASHA worker taps a quick-action
    button or submits a free-text note on a patient card.

    Actions:
        "done"             → dose_confirmed — resets silence and days_missed
        "could_not_visit"  → dose_missed   — increments silence_days
        "contact_screened" → marks contact node screened=true in Cosmos DB
        "issue"            → flags for District Officer review
        "free_text"        → passes note to Stage 1 NER pipeline via Event Hubs

    Parameters:
        gc           : Gremlin client (may be None in offline mode)
        producer     : Event Hubs producer (may be None)
        action       : action string from dashboard button
        patient_id   : Nikshay patient ID
        asha_id      : ASHA worker ID
        free_text    : note text (for "free_text" action)
        contact_name : contact's name (for "contact_screened" action)

    Returns event dict the dashboard displays as confirmation.
    """
    from stage1_nlp import (
        writeback_dose_confirmed,
        writeback_dose_missed,
        writeback_contact_screened,
        publish_event,
    )

    event = {
        "asha_id":    asha_id,
        "patient_id": patient_id,
        "action":     action,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    if action == "done":
        event["event_type"] = "dose_confirmed"
        if gc:
            delta = writeback_dose_confirmed(gc, producer, patient_id, asha_id)
            event["graph_delta"] = delta
        else:
            publish_event(producer, "dose_confirmed", asha_id, patient_id,
                          {"source": "dashboard"})

    elif action == "could_not_visit":
        event["event_type"] = "dose_missed"
        if gc:
            delta = writeback_dose_missed(gc, producer, patient_id, asha_id)
            event["graph_delta"] = delta
        else:
            publish_event(producer, "dose_missed", asha_id, patient_id,
                          {"source": "dashboard"})

    elif action == "contact_screened":
        event["event_type"]   = "contact_screened"
        event["contact_name"] = contact_name
        if gc and contact_name:
            writeback_contact_screened(gc, producer, patient_id,
                                       contact_name, asha_id)
        else:
            publish_event(producer, "contact_screened", asha_id, patient_id,
                          {"contact_name": contact_name, "source": "dashboard"})

    elif action == "issue":
        event["event_type"]   = "issue_flagged"
        event["flag_officer"] = True
        publish_event(producer, "issue_flagged", asha_id, patient_id,
                      {"source": "dashboard"})

    elif action == "free_text":
        if not free_text or not free_text.strip():
            event["event_type"] = "free_text_empty"
            event["error"]      = "Empty free text — not published to Event Hubs"
        else:
            event["event_type"]   = "free_text_update"
            event["requires_ner"] = True
            event["text"]         = free_text
            publish_event(producer, "free_text_update", asha_id, patient_id,
                          {"text": free_text, "source": "dashboard"})

    else:
        event["event_type"] = "unknown"
        event["error"]      = f"Unrecognised action: {action}"

    print(f"  [Dashboard reply] {asha_id} → {patient_id}: {event['event_type']}")
    return event


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER  (called by main.py after Stage 4)
# ─────────────────────────────────────────────────────────────────────────────

def run_morning_briefings(visit_list: list, screening_list: list,
                           systemic_alerts: list, patients: list,
                           preferred_language: str = "English") -> dict:
    """
    Build one briefing per ASHA worker and return the full output dict.
    Output is saved to briefings_output.json for app.py to load.

    Returns:
        {"asha_briefings": {asha_id: briefing_dict}, "systemic_alerts": [...]}
    """
    from collections import defaultdict

    asha_visits    = defaultdict(list)
    for v in visit_list:
        asha_visits[v["asha_id"]].append(v)

    patient_lookup = {p["patient_id"]: p for p in patients}
    briefings      = {}

    print(f"\nBuilding morning briefings for {len(asha_visits)} ASHA workers...")

    for asha_id, visits in asha_visits.items():
        first_pid = visits[0]["patient_id"]
        language  = (patient_lookup.get(first_pid, {})
                     .get("operational", {})
                     .get("language", "Tamil"))

        # Scope screening list to contacts belonging to this ASHA's patients only
        asha_pids = {v["patient_id"] for v in visits}
        asha_screening = [c for c in screening_list
                          if c.get("source_patient") in asha_pids]

        briefing = format_morning_briefing(
            visits, asha_screening, systemic_alerts, asha_id, language
        )
        briefings[asha_id] = briefing
        audio_status = "yes" if briefing["audio_path"] else "no (check SPEECH_KEY)"
        print(f"  ✓ {asha_id} ({language}): "
              f"{len(visits)} patients | audio={audio_status}")

    if systemic_alerts:
        print(f"\n⚠  {len(systemic_alerts)} systemic alerts — "
              f"visible on District Officer tab")

    return {
        "asha_briefings":  briefings,
        "systemic_alerts": systemic_alerts,
    }


if __name__ == "__main__":
    with open("agent3_output.json") as f:
        agent3 = json.load(f)
    with open("nikshay_scored_dataset.json") as f:
        patients = json.load(f)

    result = run_morning_briefings(
        visit_list      = agent3["visit_list"],
        screening_list  = agent3["screening_list"],
        systemic_alerts = agent3["systemic_alerts"],
        patients        = patients,
    )
    print(f"\nBriefings built for {len(result['asha_briefings'])} ASHA workers.")
    print("Open the dashboard (streamlit run app.py) to view and play them.")
