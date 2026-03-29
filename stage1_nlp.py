"""
stage1_nlp.py
=============
Stage 1: NLP ingestion, graph construction, and ASHA field writeback.

Node types (7):  Patient, ASHA Worker, CHO, PHC, WelfareScheme, Village, Contact
Edge types (7):  household_contact, workplace_contact, shared_contact,
                 assigned_to, assessed_by, attends, enrolled_in

Key changes from stage1_ingest.py:
  - UPSERT throughout — graph is never destroyed between runs
  - PHC node replaces DOTS centre; attends edge replaces treats edge
  - WelfareScheme node + enrolled_in edges
  - Load score denominator: 15 (was 50)
  - Phase-adaptive silence: 5d intensive / 6d late continuation / 7d continuation
  - Data-inputting workers: ASHA (dose tracking, field notes),
    CHO (Annexure 1 clinical assessment), MO at PHC (Record A at diagnosis)
  - ASHA writeback functions close the feedback loop to Cosmos immediately:
      writeback_dose_confirmed, writeback_dose_missed,
      writeback_contact_screened, writeback_risk_scores,
      writeback_pagerank_scores, promote_contact_to_patient
  - inject_silence_events NOT called inside ingest_all (was causing double injection)
"""

import os, sys, json, asyncio
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ── Azure AI Language ─────────────────────────────────────────────────────────

def get_language_client():
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        ep  = os.getenv("LANGUAGE_ENDPOINT")
        key = os.getenv("LANGUAGE_KEY")
        if not ep or not key:
            print("  [NER] LANGUAGE_ENDPOINT/KEY not set — NER disabled")
            return None
        return TextAnalyticsClient(endpoint=ep, credential=AzureKeyCredential(key))
    except ImportError:
        return None

SAMPLE_NOTES = {
    "NIK-100001": "Patient lives with wife Meena Devi (42) and son Karthik (16). Works at Tondiarpet tannery.",
    "NIK-100002": "Household: mother Savitri (68, diabetic) and brother Rajan (35).",
    "NIK-100003": "Wife Anitha (30) and daughter Priya (8) at home. Father Murugan (72) also present.",
    "NIK-100004": "Gave dose today. Wife has been coughing for 3 days. Patient reluctant to continue.",
}

def extract_contacts_from_note(lc, note: str) -> list:
    if lc is None:
        return []
    result = lc.recognize_entities([note])[0]
    if result.is_error:
        return []
    contacts, current = [], {}
    for ent in result.entities:
        if ent.category == "Person":
            if current: contacts.append(current)
            current = {"name": ent.text, "confidence": round(ent.confidence_score, 3)}
        elif ent.category == "Age" and current:
            current["age_text"] = ent.text
        elif ent.category == "PersonType" and current:
            current["relationship"] = ent.text
    if current:
        contacts.append(current)
    return contacts

def extract_update_intent(lc, note: str) -> dict:
    n = note.lower()
    return {
        "dose_confirmed":         any(w in n for w in ["took dose","gave dose","confirmed"]),
        "dose_missed":            any(w in n for w in ["could not visit","not home","missed"]),
        "patient_reluctance":     any(w in n for w in ["reluctant","feels better","doesn't want"]),
        "new_symptom_in_contact": any(w in n for w in ["coughing","fever","symptoms"]),
    }

def demo_ner(lc):
    if lc is None:
        print("  [NER] Skipped — not configured")
        return
    for pid, note in list(SAMPLE_NOTES.items())[:2]:
        contacts = extract_contacts_from_note(lc, note)
        intent   = extract_update_intent(lc, note)
        print(f"    {pid}: {len(contacts)} contacts, intent={intent}")

def transcribe_voice_note(audio_path: str, language: str = "Tamil") -> str:
    bhashini = {"Tamil","Telugu","Kannada","Bengali"}
    if language in bhashini:
        return f"[Bhashini ASR stub — {language} transcription pending]"
    return f"[Azure Speech stub — {language} transcription pending]"


# ── Azure Event Hubs ──────────────────────────────────────────────────────────

def get_eventhub_producer():
    try:
        from azure.eventhub import EventHubProducerClient
        conn = os.getenv("EVENTHUB_CONNECTION_STRING")
        hub  = os.getenv("EVENTHUB_NAME", "graph-events")
        if not conn: return None
        return EventHubProducerClient.from_connection_string(conn, eventhub_name=hub)
    except ImportError:
        return None

def publish_event(producer, event_type: str, source_id: str, target_id: str, features: dict):
    if producer is None: return
    try:
        from azure.eventhub import EventData
        payload = json.dumps({
            "event_type": event_type, "source_node": source_id, "target_node": target_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "features": features,
        })
        batch = producer.create_batch()
        batch.add(EventData(payload))
        producer.send_batch(batch)
    except Exception as e:
        print(f"  [EventHubs] {e}")



# ─────────────────────────────────────────────────────────────────────────────
# RED FLAG ALERT — immediate triage-positive referral (Fix 10)
# Bypasses overnight pipeline. Fires immediately when ASHA checks any red flag.
# Routes to CHO (first clinical responder per Annexure 5) and then to MO at PHC.
# Per Section 2.3.2 National Guidance: triage-positive = immediate referral.
# ─────────────────────────────────────────────────────────────────────────────

RED_FLAG_ALERTS_FILE = "data/red_flag_alerts.json"

def publish_red_flag_alert(producer, patient_id: str, asha_id: str,
                            cho_id: str, active_flags: list,
                            patient_record: dict = None) -> dict:
    """
    Publish an immediate red_flag_referral event and write to
    data/red_flag_alerts.json.

    Alert routes to CHO at AAM level (Annexure 5 — CHO is the first
    clinical responder per National Guidance 2025).

    active_flags: list of flag key strings that are True, e.g.
        ["breathlessness", "haemoptysis_one_cup"]
    """
    from datetime import datetime, timezone
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    alert = {
        "alert_id":               f"RF-{patient_id}-{timestamp[:19].replace(':','')}",
        "patient_id":             patient_id,
        "asha_id":                asha_id,
        "cho_id":                 cho_id,
        "active_flags":           active_flags,
        "timestamp":              timestamp,
        "status":                 "pending",
        "recipient":              "CHO",
        "mo_alert":               False,   # set True when CHO submits referral
        "pre_arrival_coordination": True,
        "alert_type":             "red_flag_referral",
        "severity_classification": None,
        "mo_clinical_params":      None,
        "acknowledged_at":         None,
        "acknowledged_by":         None,
    }

    alerts = []
    alerts_path = Path(RED_FLAG_ALERTS_FILE)
    if alerts_path.exists():
        try:
            with open(alerts_path) as f:
                alerts = json.load(f)
        except Exception:
            alerts = []

    today = timestamp[:10]
    existing_idx = next(
        (i for i, a in enumerate(alerts)
         if a["patient_id"] == patient_id
         and a["timestamp"][:10] == today
         and a["status"] == "pending"),
        None
    )
    if existing_idx is not None:
        existing_flags = set(alerts[existing_idx].get("active_flags", []))
        existing_flags.update(active_flags)
        alerts[existing_idx]["active_flags"] = list(existing_flags)
        alerts[existing_idx]["timestamp"]    = timestamp
    else:
        alerts.insert(0, alert)

    alerts = alerts[:500]
    with open(alerts_path, "w") as f:
        json.dump(alerts, f, indent=2)

    # Propagate active flags into the in-memory patient record so that
    # rescore_patient_locally() sees them when compute_bbn_prior() runs.
    # BBN has a log-odds term for adr_symptoms; other flags mark acute severity.
    if patient_record is not None:
        existing = set(patient_record.get("active_red_flags", []))
        existing.update(active_flags)
        patient_record["active_red_flags"] = list(existing)

        adh = patient_record.get("adherence") or {}
        if any("adr" in f.lower() for f in active_flags):
            adh["adr_symptoms"] = True
        patient_record["adherence"] = adh

        # Store each flag key as a top-level bool for Explainability tab lookups
        for flag in active_flags:
            patient_record[flag] = True

    publish_event(producer, "red_flag_referral", asha_id, patient_id, {
        "cho_id":       cho_id,
        "active_flags": active_flags,
        "pre_arrival":  True,
        "timestamp":    timestamp,
    })

    print(f"  [RedFlag] IMMEDIATE ALERT: {patient_id} | flags={active_flags} | "
          f"→ CHO {cho_id}")
    return alert


def publish_mo_alert(patient_id: str, cho_id: str, management_decision: str,
                     clinical_summary: dict = None) -> dict:
    """
    Write a CHO referral decision to data/mo_alerts.json so the MO tab
    displays it as a pending assessment case.

    Called when CHO submits Annexure 1 with management_decision ==
    'referral_to_higher_centre' or 'referral_for_hospitalisation'.
    The MO is at PHC/CHC level — they receive referred patients from CHO.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    alert = {
        "alert_id":           f"MO-{patient_id}-{timestamp[:19].replace(':','')}",
        "patient_id":         patient_id,
        "cho_id":             cho_id,
        "management_decision": management_decision,
        "clinical_summary":   clinical_summary or {},
        "timestamp":          timestamp,
        "status":             "pending",
        "recipient":          "MO",
    }

    mo_alerts_path = Path("data/mo_alerts.json")
    alerts = []
    if mo_alerts_path.exists():
        try:
            with open(mo_alerts_path) as f:
                alerts = json.load(f)
        except Exception:
            alerts = []

    # Upsert — one pending alert per patient
    alerts = [a for a in alerts if a["patient_id"] != patient_id or a["status"] != "pending"]
    alerts.insert(0, alert)
    alerts = alerts[:500]

    with open(mo_alerts_path, "w") as f:
        json.dump(alerts, f, indent=2)

    # Also mark the corresponding red_flag_alert as mo_alert=True
    rf_path = Path(RED_FLAG_ALERTS_FILE)
    if rf_path.exists():
        try:
            with open(rf_path) as f:
                rf_alerts = json.load(f)
            for a in rf_alerts:
                if a["patient_id"] == patient_id and a["status"] == "pending":
                    a["mo_alert"] = True
            with open(rf_path, "w") as f:
                json.dump(rf_alerts, f, indent=2)
        except Exception:
            pass

    print(f"  [MO Alert] {patient_id} referred by CHO {cho_id}: {management_decision}")
    return alert


def publish_dto_alert(patient_id: str, mo_classified_by: str,
                      classification: str, clinical_params: dict = None) -> dict:
    """
    Write a severe MO classification to data/dto_alerts.json so the DTO tab
    displays it. Called ONLY when MO classifies as 'referral_for_hospitalisation'.

    The DTO never sees raw ASHA red flags or unassessed CHO referrals —
    only MO-confirmed severe cases reach the DTO level.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    alert = {
        "alert_id":        f"DTO-{patient_id}-{timestamp[:19].replace(':','')}",
        "patient_id":      patient_id,
        "mo_classified_by": mo_classified_by,
        "classification":  classification,
        "clinical_params": clinical_params or {},
        "timestamp":       timestamp,
        "status":          "pending",
        "recipient":       "DTO",
    }

    dto_path = Path("data/dto_alerts.json")
    alerts = []
    if dto_path.exists():
        try:
            with open(dto_path) as f:
                alerts = json.load(f)
        except Exception:
            alerts = []

    alerts = [a for a in alerts if a["patient_id"] != patient_id or a["status"] != "pending"]
    alerts.insert(0, alert)
    alerts = alerts[:500]

    with open(dto_path, "w") as f:
        json.dump(alerts, f, indent=2)

    print(f"  [DTO Alert] SEVERE: {patient_id} — {classification} | MO: {mo_classified_by}")
    return alert


def save_severity_classification(patient_id: str, classification: str,
                                  clinical_params: dict,
                                  classified_by: str = "MO") -> dict:
    """
    Save MO/CHO severity classification to data/severity_classifications.json.
    Also updates any pending red_flag_alert for this patient with the MO result.

    classification: "ambulatory_care" | "referral_to_higher_centre" | "referral_for_hospitalisation"
    clinical_params: {spo2, respiratory_rate, systolic_bp, pulse_rate, bmi}

    Does NOT touch risk_score — this is the clinical severity track, orthogonal
    to the LTFU dropout prediction track (Table 2.1, National Guidance 2025).
    """
    from datetime import datetime, timezone
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "patient_id":       patient_id,
        "classification":   classification,
        "clinical_params":  clinical_params,
        "classified_by":    classified_by,
        "timestamp":        timestamp,
        "severe":           classification == "referral_for_hospitalisation",
    }

    # Load and update
    path = Path("data/severity_classifications.json")
    data = []
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = []

    # Upsert by patient_id (latest wins)
    data = [d for d in data if d["patient_id"] != patient_id]
    data.insert(0, record)
    data = data[:500]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Update pending red_flag_alert for this patient
    alerts_path = Path(RED_FLAG_ALERTS_FILE)
    if alerts_path.exists():
        try:
            with open(alerts_path) as f:
                alerts = json.load(f)
            for a in alerts:
                if a["patient_id"] == patient_id and a["status"] == "pending":
                    a["severity_classification"] = classification
                    a["mo_clinical_params"]      = clinical_params
            with open(alerts_path, "w") as f:
                json.dump(alerts, f, indent=2)
        except Exception:
            pass

    print(f"  [Severity] {patient_id}: {classification} | params={clinical_params}")

    # Escalate to DTO only for severe cases — raw referrals never reach DTO
    if classification == "referral_for_hospitalisation":
        publish_dto_alert(patient_id, classified_by, classification, clinical_params)

    return record


# ── Gremlin helpers ───────────────────────────────────────────────────────────

def safe(val) -> str:
    return str(val).replace("'", "\\'")

def run_query(gc, query: str):
    try:
        return gc.submit(query).all().result()
    except Exception as e:
        print(f"  [Gremlin] {e}")
        return []


# ── ASHA summaries ────────────────────────────────────────────────────────────

def build_asha_summaries(records: list) -> dict:
    """
    Load score denominator is 15 (system-defined max caseload).
    At 15 patients → caseload component = 0.40 (ceiling).
    At  5 patients → caseload component = 0.13 (low).
    """
    groups = defaultdict(list)
    for r in records:
        groups[r["operational"]["asha_id"]].append(r)

    summaries = {}
    for asha_id, pts in groups.items():
        n          = len(pts)
        avg_dist   = sum(p["adherence"]["distance_to_center_km"] for p in pts) / n
        visited_7d = sum(1 for p in pts if p["operational"]["last_asha_visit_days_ago"] <= 7)
        high_risk  = sum(1 for p in pts if p.get("risk_score", 0) > 0.65)
        avg_missed = sum(p["adherence"]["days_since_last_dose"] for p in pts) / n
        load = min(1.0, (
            (n / 15)           * 0.4 +   # caseload pressure — denominator 15
            (avg_missed / 14)  * 0.3 +   # patient engagement decline
            (high_risk / max(n,1)) * 0.3  # proportion high-risk
        ))
        summaries[asha_id] = {
            "asha_id":        asha_id,
            "caseload":       n,
            "avg_distance_km":round(avg_dist, 2),
            "visit_freq_7d":  round(visited_7d / n, 3),
            "high_risk_count":high_risk,
            "load_score":     round(load, 4),
            "district":       pts[0]["location"]["district"],
            "block":          pts[0]["location"]["block"],
        }
    return summaries


def build_cho_summaries(records: list) -> dict:
    """
    Aggregate per-CHO statistics from patient records.

    CHO caseload = all patients whose ASHA is within the CHO's AAM zone.
    Typical CHO zone: ~3 ASHAs, ~45 patients.

    Properties stored on CHO node:
      - caseload:          total patients in zone
      - asha_count:        number of ASHAs supervised
      - high_risk_count:   patients with risk_score > 0.65
      - avg_adherence:     mean adherence_rate_30d across zone
      - pending_flags:     count of patients with any red flag in last Record B
      - district / block
    """
    groups = defaultdict(list)
    for r in records:
        cho_id = r["operational"].get("cho_id", "")
        if cho_id:
            groups[cho_id].append(r)

    summaries = {}
    for cho_id, pts in groups.items():
        n            = len(pts)
        high_risk    = sum(1 for p in pts if p.get("risk_score", 0) > 0.65)
        avg_adh      = sum(p["adherence"]["adherence_rate_30d"] for p in pts) / n
        asha_ids     = {p["operational"]["asha_id"] for p in pts}
        pending_flags = sum(
            1 for p in pts
            if any(
                rb.get("red_flags", {}).get("any_red_flag_positive", False)
                for rb in p.get("records_b", [])[-1:]   # last Record B only
            )
        )
        summaries[cho_id] = {
            "cho_id":          cho_id,
            "caseload":        n,
            "asha_count":      len(asha_ids),
            "high_risk_count": high_risk,
            "avg_adherence":   round(avg_adh, 4),
            "pending_flags":   pending_flags,
            "district":        pts[0]["location"]["district"],
            "block":           pts[0]["location"]["block"],
        }
    return summaries

def build_village_summaries(records: list) -> dict:
    groups = defaultdict(list)
    for r in records:
        key = f"{r['location']['district']}_{r['location']['block']}"
        groups[key].append(r)
    result = {}
    for vid, pts in groups.items():
        avg_dist = sum(p["adherence"]["distance_to_center_km"] for p in pts) / len(pts)
        low_edu  = sum(1 for p in pts if p["social"]["low_education"]) / len(pts)
        result[vid] = {
            "village_id": vid,
            "district":   pts[0]["location"]["district"],
            "block":      pts[0]["location"]["block"],
            "patient_count":   len(pts),
            "avg_distance_km": round(avg_dist, 2),
            "low_edu_rate":    round(low_edu, 3),
            "connectivity_score": round(max(0, 1 - avg_dist / 20), 3),
        }
    return result


# ── Phase-adaptive silence detection ─────────────────────────────────────────

def _silence_threshold(record: dict) -> int:
    """
    Intensive phase: flag at 5 days (daily doses expected).
    Late continuation (week 17+): flag at 6 days.
    Continuation: flag at 7 days.
    """
    phase = record["clinical"]["phase"]
    week  = min(record["clinical"]["total_treatment_days"] // 7, 26)
    if phase == "Intensive":
        return 5
    elif week >= 17:
        return 6
    return 7

def inject_silence_events(records: list, producer) -> list:
    count = 0
    for r in records:
        effective = max(r["adherence"]["days_since_last_dose"],
                        r["operational"]["last_asha_visit_days_ago"])
        threshold = _silence_threshold(r)
        if effective >= threshold:
            r["silence_event"] = {
                "duration_days": effective,
                "type": "complete" if r["adherence"]["days_since_last_dose"] >= 14 else "partial",
                "threshold_used": threshold,
            }
            publish_event(producer, "silence", r["patient_id"], r["patient_id"],
                          {"duration_days": effective, "threshold": threshold})
            count += 1
    print(f"  Silence events: {count} patients (thresholds: 5d intensive / 6d late / 7d continuation)")
    return records


# ── Node upserts (coalesce pattern — safe to run multiple times) ──────────────

def _upsert(gc, label: str, vid: str, props: dict, district: str):
    """Generic upsert: update if exists, create if not. Never destroys data."""
    set_str    = "".join(f".property('{k}', {v if isinstance(v,(int,float,bool)) else repr(str(v))})"
                         for k, v in props.items())
    create_str = set_str + f".property('pk', '{safe(district)}')"
    _q = (
        f"g.V('{safe(vid)}').fold().coalesce("
        f"unfold(){set_str},"
        f"addV('{label}').property('id','{safe(vid)}'){create_str}"
        f")"
    )
    run_query(gc, _q)

def ingest_patient_node(gc, producer, record: dict):
    pid      = record["patient_id"]
    district = record["location"]["district"]
    silence  = str(record.get("silence_event") is not None).lower()
    s_days   = record.get("silence_event", {}).get("duration_days", 0)
    mem_init = json.dumps([0.0] * 64)

    # Build the full upsert query manually to handle the memory_vector init carefully.
    # CRITICAL: 'pk' (partition key) is IMMUTABLE in Cosmos DB Gremlin.
    # It must ONLY appear in the addV() branch, NEVER in the unfold()/update branch.
    # Setting pk on an existing vertex causes GraphSyntaxException code 597.
    shared_props = (
        f".property('risk_score',       {record['risk_score']})"
        f".property('phase',            '{safe(record['clinical']['phase'])}')"
        f".property('regimen',          '{safe(record['clinical'].get('regimen','Cat_I'))}')"
        f".property('treatment_week',   {min(record['clinical']['total_treatment_days']//7,26)})"
        f".property('days_missed',      {record['adherence']['days_since_last_dose']})"
        f".property('adherence_rate',   {record['adherence']['adherence_rate_30d']})"
        f".property('phase_adherence',  {record['adherence'].get('phase_adherence_rate', record['adherence']['adherence_rate_30d'])})"
        f".property('distance_km',      {record['adherence']['distance_to_center_km']})"
        f".property('silence',          {silence})"
        f".property('silence_days',     {s_days})"
        f".property('welfare_enrolled', {str(record['operational'].get('welfare_enrolled',False)).lower()})"
        f".property('risk_velocity',    {record.get('risk_velocity',0.0)})"
        f".property('asha_id',          '{safe(record['operational']['asha_id'])}')"
        f".property('cho_id',           '{safe(record['operational'].get('cho_id',''))}')"
        f".property('block',            '{safe(record['location']['block'])}')"
        f".property('district',         '{safe(district)}')"
    )
    run_query(gc,
        f"g.V('{safe(pid)}').fold().coalesce("
        f"unfold(){shared_props},"
        f"addV('patient')"
        f".property('id',            '{safe(pid)}')"
        f".property('pk',            '{safe(district)}')"   # pk ONLY here
        f".property('memory_vector', '{safe(mem_init)}')"
        f"{shared_props}"
        f")"
    )
    publish_event(producer, "patient_upserted", pid, pid,
                  {"risk_score": record["risk_score"]})

def ingest_asha_node(gc, producer, summary: dict):
    aid = summary["asha_id"]
    run_query(gc,
        f"g.V('{safe(aid)}').fold().coalesce("
        f"unfold()"
        f".property('caseload',        {summary['caseload']})"
        f".property('load_score',      {summary['load_score']})"
        f".property('visit_freq_7d',   {summary['visit_freq_7d']})"
        f".property('high_risk_count', {summary['high_risk_count']}),"
        f"addV('asha_worker')"
        f".property('id',              '{safe(aid)}')"
        f".property('caseload',        {summary['caseload']})"
        f".property('load_score',      {summary['load_score']})"
        f".property('avg_distance_km', {summary['avg_distance_km']})"
        f".property('visit_freq_7d',   {summary['visit_freq_7d']})"
        f".property('high_risk_count', {summary['high_risk_count']})"
        f".property('district',        '{safe(summary['district'])}')"
        f".property('block',           '{safe(summary['block'])}')"
        f".property('pk',              '{safe(summary['district'])}')"
        f")"
    )


def ingest_cho_node(gc, producer, summary: dict):
    """
    CHO node — one per AAM catchment area (~3 ASHA zones, ~45 patients).
    Updated each pipeline run with current caseload stats.
    """
    cid = summary["cho_id"]
    run_query(gc,
        f"g.V('{safe(cid)}').fold().coalesce("
        f"unfold()"
        f".property('caseload',        {summary['caseload']})"
        f".property('asha_count',      {summary['asha_count']})"
        f".property('high_risk_count', {summary['high_risk_count']})"
        f".property('avg_adherence',   {summary['avg_adherence']})"
        f".property('pending_flags',   {summary['pending_flags']}),"
        f"addV('cho_worker')"
        f".property('id',              '{safe(cid)}')"
        f".property('caseload',        {summary['caseload']})"
        f".property('asha_count',      {summary['asha_count']})"
        f".property('high_risk_count', {summary['high_risk_count']})"
        f".property('avg_adherence',   {summary['avg_adherence']})"
        f".property('pending_flags',   {summary['pending_flags']})"
        f".property('district',        '{safe(summary['district'])}')"
        f".property('block',           '{safe(summary['block'])}')"
        f".property('pk',              '{safe(summary['district'])}')"
        f")"
    )
    publish_event(producer, "cho_upserted", cid, cid,
                  {"caseload": summary["caseload"],
                   "high_risk_count": summary["high_risk_count"]})


def ingest_cho_patient_edge(gc, producer, record: dict, cho_summary: dict):
    """
    assessed_by edge: Patient → CHO.

    Weight reflects recency and depth of CHO assessment:
      - Base weight 0.8 (CHO sees patient monthly, less frequent than ASHA)
      - Boosted toward 1.0 if patient has a pending red flag (CHO actively engaged)
      - Decays at 0.02/day between monthly assessments
    """
    pid    = record["patient_id"]
    cho_id = record["operational"].get("cho_id", "")
    if not cho_id:
        return

    # Recency: months since last Record B (0 = assessed this month)
    records_b   = record.get("records_b", [])
    last_month  = max((rb.get("month", 0) for rb in records_b), default=0)
    total_weeks = record["clinical"].get("total_treatment_days", 0) // 7
    months_elapsed = max(total_weeks // 4, 1)
    recency     = min(float(last_month) / max(months_elapsed, 1), 1.0)

    # Flag boost — CHO is actively engaged if patient had a recent red flag
    any_flag = any(
        rb.get("red_flags", {}).get("any_red_flag_positive", False)
        for rb in records_b[-1:]
    )
    base_w = 0.9 if any_flag else 0.8
    w      = round(base_w * recency, 4)

    run_query(gc,
        f"g.V('{safe(pid)}').outE('assessed_by')"
        f".where(inV().hasId('{safe(cho_id)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',        {w})"
        f".property('last_month',    {last_month})"
        f".property('flag_active',   {str(any_flag).lower()}),"
        f"g.V('{safe(pid)}').addE('assessed_by')"
        f".to(g.V('{safe(cho_id)}'))"
        f".property('weight',        {w})"
        f".property('base_weight',   {base_w})"
        f".property('last_month',    {last_month})"
        f".property('flag_active',   {str(any_flag).lower()})"
        f".property('decay_rate',    0.02)"
        f")"
    )
    publish_event(producer, "patient_assessed_by_cho", pid, cho_id,
                  {"weight": w, "flag_active": any_flag})


def ingest_phc_node(gc, producer, district: str, block: str) -> str:
    """
    PHC node replaces DOTS centre.
    drug_available and staffed are operational state properties.
    A stockout sets drug_available=false → risk propagates to all connected patients.
    """
    phc_id = f"PHC_{block}"
    run_query(gc,
        f"g.V('{safe(phc_id)}').fold().coalesce("
        f"unfold(),"
        f"addV('phc')"
        f".property('id',             '{safe(phc_id)}')"
        f".property('block',          '{safe(block)}')"
        f".property('district',       '{safe(district)}')"
        f".property('drug_available', true)"
        f".property('staffed',        true)"
        f".property('pk',             '{safe(district)}')"
        f")"
    )
    return phc_id

def ingest_welfare_scheme_node(gc, producer, district: str) -> str:
    """
    Nikshay Poshan Yojana node.
    payment_status: 'active' or 'delayed'.
    When delayed → enrolled_in edge weight drops → risk rises for all enrolled patients.
    """
    scheme_id = "SCHEME_NikshayPoshanYojana"
    run_query(gc,
        f"g.V('{safe(scheme_id)}').fold().coalesce("
        f"unfold(),"
        f"addV('welfare_scheme')"
        f".property('id',             '{safe(scheme_id)}')"
        f".property('name',           'Nikshay Poshan Yojana')"
        f".property('payment_status', 'active')"
        f".property('amount_inr',     500)"
        f".property('district',       '{safe(district)}')"
        f".property('pk',             '{safe(district)}')"
        f")"
    )
    return scheme_id

def ingest_village_node(gc, producer, summary: dict):
    vid = summary["village_id"]
    run_query(gc,
        f"g.V('{safe(vid)}').fold().coalesce("
        f"unfold(),"
        f"addV('village')"
        f".property('id',                 '{safe(vid)}')"
        f".property('district',           '{safe(summary['district'])}')"
        f".property('block',              '{safe(summary['block'])}')"
        f".property('patient_count',      {summary['patient_count']})"
        f".property('avg_distance_km',    {summary['avg_distance_km']})"
        f".property('connectivity_score', {summary['connectivity_score']})"
        f".property('low_edu_rate',       {summary['low_edu_rate']})"
        f".property('pk',                 '{safe(summary['district'])}')"
        f")"
    )

def ingest_contact_node(gc, producer, record: dict, contact: dict,
                        contact_registry: dict) -> tuple:
    pid      = record["patient_id"]
    district = record["location"]["district"]
    name     = contact["name"]
    cid      = f"CONTACT_{safe(name).replace(' ','_').replace('.','')}"

    if name not in contact_registry:
        run_query(gc,
            f"g.V('{safe(cid)}').fold().coalesce("
            f"unfold(),"
            f"addV('contact')"
            f".property('id',            '{safe(cid)}')"
            f".property('name',          '{safe(name)}')"
            f".property('age',           {contact['age']})"
            f".property('vulnerability', {contact['vulnerability_score']})"
            f".property('rel',           '{safe(contact['rel'])}')"
            f".property('screened',      {str(contact['screened']).lower()})"
            f".property('patient_count', 1)"
            f".property('district',      '{safe(district)}')"
            f".property('pk',            '{safe(district)}')"
            f")"
        )
        contact_registry[name] = (cid, pid)
        return cid, True
    else:
        existing_cid, existing_pid = contact_registry[name]
        bridge_w = round(contact["vulnerability_score"] * record.get("risk_score", 0.1), 4)
        run_query(gc,
            f"g.V('{safe(pid)}').addE('shared_contact')"
            f".to(g.V('{safe(existing_pid)}'))"
            f".property('weight',       {bridge_w})"
            f".property('contact_name', '{safe(name)}')"
            f".property('decay_rate',   0.04)"
        )
        publish_event(producer, "shared_contact_bridge", pid, existing_pid,
                      {"contact_name": name, "weight": bridge_w})
        return existing_cid, False


# ── Edge ingestion ────────────────────────────────────────────────────────────

def ingest_contact_edge(gc, producer, record: dict, contact: dict, cid: str):
    pid   = record["patient_id"]
    base  = 0.9 if contact["rel"] == "Household" else 0.6
    w     = round(base * contact["vulnerability_score"], 4)
    label = "household_contact" if contact["rel"] == "Household" else "workplace_contact"
    decay = 0.02 if contact["rel"] == "Household" else 0.05
    run_query(gc,
        f"g.V('{safe(pid)}').outE('{label}').where(inV().hasId('{safe(cid)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',                    {w})"
        f".property('last_interaction_days_ago', 0),"
        f"g.V('{safe(pid)}').addE('{label}')"
        f".to(g.V('{safe(cid)}'))"
        f".property('weight',                    {w})"
        f".property('base_weight',               {base})"
        f".property('last_interaction_days_ago', 0)"
        f".property('decay_rate',                {decay})"
        f")"
    )
    publish_event(producer, label, pid, cid, {"weight": w})

def ingest_asha_patient_edge(gc, producer, record: dict, asha_summary: dict):
    pid        = record["patient_id"]
    aid        = record["operational"]["asha_id"]
    load       = asha_summary["load_score"]
    days_since = record["operational"]["last_asha_visit_days_ago"]
    recency    = max(0, 1 - days_since / 30)
    w          = round((1 - load) * recency, 4)
    run_query(gc,
        f"g.V('{safe(aid)}').outE('assigned_to').where(inV().hasId('{safe(pid)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',          {w})"
        f".property('load_score',      {load})"
        f".property('days_since_visit',{days_since}),"
        f"g.V('{safe(aid)}').addE('assigned_to')"
        f".to(g.V('{safe(pid)}'))"
        f".property('weight',          {w})"
        f".property('load_score',      {load})"
        f".property('days_since_visit',{days_since})"
        f".property('decay_rate',      0.03)"
        f")"
    )
    publish_event(producer, "asha_patient_assigned", aid, pid,
                  {"weight": w, "load_score": load})


def ingest_phc_patient_edge(gc, producer, record: dict, phc_id: str):
    """attends edge replaces treats/DOTS edge."""
    pid     = record["patient_id"]
    dist    = record["adherence"]["distance_to_center_km"]
    tf      = 1.0 if record["operational"]["phone_type"] == "Smartphone" else 0.8
    w       = round((1 / (1 + dist * 0.1)) * tf, 4)
    run_query(gc,
        f"g.V('{safe(pid)}').addE('attends')"
        f".to(g.V('{safe(phc_id)}'))"
        f".property('weight',      {w})"
        f".property('distance_km', {dist})"
        f".property('decay_rate',  0.01)"
    )
    publish_event(producer, "patient_attends_phc", pid, phc_id,
                  {"weight": w, "distance_km": dist})

def ingest_welfare_edge(gc, producer, record: dict, scheme_id: str):
    """enrolled_in edge — only created if patient is enrolled."""
    if not record["operational"].get("welfare_enrolled", False):
        return
    pid = record["patient_id"]
    run_query(gc,
        f"g.V('{safe(pid)}').addE('enrolled_in')"
        f".to(g.V('{safe(scheme_id)}'))"
        f".property('weight',         1.0)"
        f".property('payment_status', 'active')"
    )
    publish_event(producer, "patient_enrolled_welfare", pid, scheme_id, {})


# ── ASHA writeback functions (close the feedback loop) ────────────────────────

def writeback_dose_confirmed(gc, producer, patient_id: str, asha_id: str) -> dict:
    """
    ASHA tapped Done.
    Resets days_missed/silence on patient node.
    Recalculates assigned_to edge weight at full recency (days_since_visit=0).
    Returns a delta dict describing what changed in the graph.
    """
    # --- Patient node ---
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('days_missed', 0)"
        f".property('silence',     false)"
        f".property('silence_days', 0)"
    )

    # --- Read current edge state before update ---
    old_weight_r     = run_query(gc,
        f"g.V('{safe(asha_id)}').outE('assigned_to')"
        f".where(inV().has('id', '{safe(patient_id)}'))"
        f".values('weight')"
    )
    old_load_r       = run_query(gc,
        f"g.V('{safe(asha_id)}').outE('assigned_to')"
        f".where(inV().has('id', '{safe(patient_id)}'))"
        f".values('load_score')"
    )
    old_days_r       = run_query(gc,
        f"g.V('{safe(asha_id)}').outE('assigned_to')"
        f".where(inV().has('id', '{safe(patient_id)}'))"
        f".values('days_since_visit')"
    )
    old_weight       = float(old_weight_r[0]) if old_weight_r else None
    load_score       = float(old_load_r[0])   if old_load_r   else 0.5
    old_days         = int(old_days_r[0])      if old_days_r   else 0

    # New weight: full recency (visit just happened → days_since_visit = 0)
    new_recency      = 1.0
    new_weight       = round((1 - load_score) * new_recency, 4)

    # --- Update edge ---
    run_query(gc,
        f"g.V('{safe(asha_id)}').outE('assigned_to')"
        f".where(inV().has('id', '{safe(patient_id)}'))"
        f".property('days_since_visit', 0)"
        f".property('weight', {new_weight})"
    )

    delta = {
        "patient_id":       patient_id,
        "asha_id":          asha_id,
        "change":           "dose_confirmed",
        "edge_weight_old":  old_weight,
        "edge_weight_new":  new_weight,
        "days_since_visit_old": old_days,
        "days_since_visit_new": 0,
        "load_score":       load_score,
        "node_changes":     {"days_missed": "→ 0", "silence": "→ false", "silence_days": "→ 0"},
    }
    publish_event(producer, "dose_confirmed", asha_id, patient_id,
                  {"days_missed_reset": True, "silence_cleared": True,
                   "edge_weight_old": old_weight, "edge_weight_new": new_weight})
    print(f"  [Writeback] dose_confirmed: {patient_id} | edge weight {old_weight} → {new_weight}")
    return delta

def writeback_dose_missed(gc, producer, patient_id: str, asha_id: str) -> dict:
    """
    ASHA tapped Could Not Visit.
    Increments silence_days on patient node.
    Increments days_since_visit and decays edge weight (recency drops).
    Returns a delta dict describing what changed in the graph.
    NOTE: does NOT touch BBN ORs — only unconfirmed visit status.
    """
    # --- Patient node ---
    sil_r     = run_query(gc, f"g.V(\'{safe(patient_id)}\').values(\'silence_days\')")
    new_sdays = (int(sil_r[0]) if sil_r else 0) + 1
    run_query(gc,
        f"g.V(\'{safe(patient_id)}\')"
        f".property(\'silence_days\', {new_sdays})"
        f".property(\'silence\',      true)"
    )

    # --- Read current edge state ---
    old_weight_r = run_query(gc,
        f"g.V(\'{safe(asha_id)}\').outE(\'assigned_to\')"
        f".where(inV().has(\'id\', \'{safe(patient_id)}\'))"
        f".values(\'weight\')"
    )
    old_load_r   = run_query(gc,
        f"g.V(\'{safe(asha_id)}\').outE(\'assigned_to\')"
        f".where(inV().has(\'id\', \'{safe(patient_id)}\'))"
        f".values(\'load_score\')"
    )
    old_days_r   = run_query(gc,
        f"g.V(\'{safe(asha_id)}\').outE(\'assigned_to\')"
        f".where(inV().has(\'id\', \'{safe(patient_id)}\'))"
        f".values(\'days_since_visit\')"
    )
    old_weight   = float(old_weight_r[0]) if old_weight_r else None
    load_score   = float(old_load_r[0])   if old_load_r   else 0.5
    old_days     = int(old_days_r[0])     if old_days_r   else 0

    # Recalculate weight with decayed recency
    new_days     = old_days + 1
    new_recency  = max(0.0, 1.0 - new_days / 30.0)
    new_weight   = round((1 - load_score) * new_recency, 4)

    # --- Update edge ---
    run_query(gc,
        f"g.V(\'{safe(asha_id)}\').outE(\'assigned_to\')"
        f".where(inV().has(\'id\', \'{safe(patient_id)}\'))"
        f".property(\'days_since_visit\', {new_days})"
        f".property(\'weight\', {new_weight})"
    )

    delta = {
        "patient_id":           patient_id,
        "asha_id":              asha_id,
        "change":               "dose_missed",
        "edge_weight_old":      old_weight,
        "edge_weight_new":      new_weight,
        "days_since_visit_old": old_days,
        "days_since_visit_new": new_days,
        "load_score":           load_score,
        "node_changes":         {"silence_days": f"-> {new_sdays}", "silence": "-> true"},
    }
    publish_event(producer, "dose_missed", asha_id, patient_id,
                  {"silence_days": new_sdays,
                   "edge_weight_old": old_weight, "edge_weight_new": new_weight})
    print(f"  [Writeback] dose_missed: {patient_id} silence_days={new_sdays} | edge weight {old_weight} -> {new_weight}")
    return delta

def writeback_contact_screened(gc, producer, patient_id: str,
                                contact_name: str, asha_id: str):
    """ASHA reports contact screened. Sets contact.screened = true in graph."""
    cid = f"CONTACT_{safe(contact_name).replace(' ','_').replace('.','')}"
    run_query(gc, f"g.V('{safe(cid)}').property('screened', true)")
    publish_event(producer, "contact_screened", asha_id, cid,
                  {"patient_id": patient_id, "contact_name": contact_name})
    print(f"  [Writeback] contact_screened: {contact_name}")

def writeback_risk_scores(gc, patients: list):
    """
    Write Stage 3 scores back to patient nodes.
    Stores previous_risk_score to enable velocity computation next cycle.
    Called by main.py after score_all_patients().
    """
    if gc is None:
        return
    updated = 0
    for p in patients:
        pid      = p["patient_id"]
        score    = p.get("risk_score", 0)
        level    = p.get("risk_level", "LOW")
        velocity = p.get("risk_velocity", 0.0)
        run_query(gc,
            f"g.V('{safe(pid)}')"
            f".property('risk_score',          {score})"
            f".property('risk_level',          '{safe(level)}')"
            f".property('risk_velocity',       {velocity})"
            f".property('previous_risk_score', {score})"
        )
        updated += 1
    print(f"  [Writeback] Risk scores written to {updated} patient nodes")

def writeback_pagerank_scores(gc, pagerank_scores: dict):
    """Write PageRank scores to nodes after Stage 3b."""
    if gc is None:
        return
    updated = 0
    for node_id, score in pagerank_scores.items():
        run_query(gc,
            f"g.V('{safe(node_id)}')"
            f".property('pagerank_score', {round(score, 8)})"
        )
        updated += 1
    print(f"  [Writeback] PageRank scores written to {updated} nodes")

def promote_contact_to_patient(gc, producer, contact_name: str,
                                source_patient_id: str, asha_id: str,
                                district: str, block: str) -> str:
    """
    Contact tests positive for TB → promote contact node to patient.
    Draws a transmission_likely edge from source patient.
    Returns new patient_id.
    """
    import random
    new_pid = f"NIK-NEW-{contact_name.replace(' ','')[:8].upper()}-{random.randint(1000,9999)}"
    cid     = f"CONTACT_{safe(contact_name).replace(' ','_').replace('.','')}"

    # Cosmos DB does not allow mutating a vertex's id after creation.
    # Correct approach: copy contact properties to a new patient vertex,
    # draw transmission edge from source, then drop the old contact node.
    contact_props = run_query(gc, f"g.V('{safe(cid)}').valueMap()")
    props = contact_props[0] if contact_props else {}
    age   = props.get('age', [30])
    age   = age[0] if isinstance(age, list) else age
    vuln  = props.get('vulnerability', [1.0])
    vuln  = vuln[0] if isinstance(vuln, list) else vuln
    mem_init = json.dumps([0.0] * 64)
    run_query(gc,
        f"g.V('{safe(new_pid)}').fold().coalesce("
        f"unfold(),"
        f"addV('patient')"
        f".property('id',           '{safe(new_pid)}')"
        f".property('node_type',    'patient')"
        f".property('asha_id',      '{safe(asha_id)}')"
        f".property('district',     '{safe(district)}')"
        f".property('block',        '{safe(block)}')"
        f".property('age',          {age})"
        f".property('risk_score',   0.0)"
        f".property('days_missed',  0)"
        f".property('silence',      false)"
        f".property('memory_vector','{safe(mem_init)}')"
        f".property('pk',           '{safe(district)}')"
        f")"
    )
    # Drop old contact node — it is now represented as a patient
    run_query(gc, f"g.V('{safe(cid)}').drop()")
    run_query(gc,
        f"g.V('{safe(source_patient_id)}').addE('transmission_likely')"
        f".to(g.V('{safe(new_pid)}'))"
        f".property('weight',    0.85)"
        f".property('confirmed', false)"
    )
    publish_event(producer, "new_tb_case_from_contact", source_patient_id, new_pid,
                  {"contact_name": contact_name})
    print(f"  [Promote] {contact_name} → {new_pid} (from {source_patient_id})")
    return new_pid



# ─────────────────────────────────────────────────────────────────────────────
# NER ON SINGLE NOTE — runs at end-of-day batch or via Azure Function trigger
# ─────────────────────────────────────────────────────────────────────────────

def run_ner_on_note(note: str, patient_id: str, lc=None) -> dict:
    """
    Run NER on a single free-text ASHA note.
    Returns structured extraction result used by writeback functions.

    Extracts:
      - contacts: [{name, age, rel, has_symptom, symptom}]
      - intents:  {dose_confirmed, dose_missed, patient_reluctance,
                   new_symptom_in_contact, welfare_issue, side_effect}
      - raw_note: original text (stored on patient node for audit)

    Uses Azure AI Language if configured, falls back to rule-based extraction.
    Rule-based is good enough for the clinical phrases ASHA workers actually use.
    """
    contacts = []
    intents  = extract_update_intent(lc, note)

    # Azure NER path
    if lc:
        raw_contacts = extract_contacts_from_note(lc, note)
        for c in raw_contacts:
            name = c.get("name", "")
            if not name:
                continue
            # Infer relationship from surrounding text
            note_lower = note.lower()
            rel = "Household"
            for workplace_word in ["coworker", "colleague", "works", "factory", "tannery", "market"]:
                if workplace_word in note_lower:
                    rel = "Workplace"
                    break
            # Detect symptom on this contact
            has_symptom = False
            symptom     = None
            name_idx    = note_lower.find(name.lower())
            snippet     = note_lower[max(0, name_idx - 30):name_idx + 80]
            for s in ["coughing", "cough", "fever", "blood", "weight loss", "night sweat"]:
                if s in snippet:
                    has_symptom = True
                    symptom     = s
                    break
            contacts.append({
                "name":        name,
                "age":         _parse_age_from_note(note, name),
                "rel":         rel,
                "has_symptom": has_symptom,
                "symptom":     symptom,
                "confidence":  c.get("confidence", 0.8),
            })
    else:
        # Rule-based fallback — covers common ASHA note patterns
        contacts = _rule_based_contact_extract(note)

    # Extended intents not in the original extract_update_intent
    n = note.lower()
    intents["welfare_issue"]  = any(w in n for w in ["poshan", "payment", "money", "allowance", "not received"])
    intents["side_effect"]    = any(w in n for w in ["side effect", "vomiting", "nausea", "rash", "pain"])
    intents["needs_screening"]= any(w in n for w in ["coughing", "cough", "fever", "symptoms", "unwell"])

    return {
        "patient_id": patient_id,
        "raw_note":   note,
        "contacts":   contacts,
        "intents":    intents,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_age_from_note(note: str, name: str) -> int:
    """Extract age near a person's name from a note. Returns 35 as default."""
    import re
    name_idx = note.lower().find(name.lower())
    snippet  = note[max(0, name_idx):name_idx + 60]
    match    = re.search(r'(\d{1,2})', snippet)
    if match:
        age = int(match.group(1))
        if 1 <= age <= 99:
            return age
    return 35


def _rule_based_contact_extract(note: str) -> list:
    """
    Rule-based NER fallback. Handles common patterns like:
      'wife Meena (42) has been coughing'
      'son Karthik (16) at home'
      'mother-in-law (68) has fever'
    """
    import re
    contacts  = []
    note_low  = note.lower()

    rel_keywords = {
        "wife": "Household", "husband": "Household", "mother": "Household",
        "father": "Household", "son": "Household", "daughter": "Household",
        "brother": "Household", "sister": "Household", "uncle": "Household",
        "aunt": "Household", "grandmother": "Household", "grandfather": "Household",
        "coworker": "Workplace", "colleague": "Workplace",
    }
    symptom_words = ["coughing", "cough", "fever", "blood", "weight loss",
                     "night sweat", "unwell", "sick", "breathless"]

    # Pattern: [rel] [Name] ([age])  OR  [rel] ([age])
    name_pattern = re.compile(
        r'(wife|husband|mother|father|son|daughter|brother|sister|uncle|aunt|'
        r'grandmother|grandfather|coworker|colleague)'
        r'(?:\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?))?\s*(?:\((\d{1,2})\))?',
        re.IGNORECASE
    )
    for m in name_pattern.finditer(note):
        rel_word = m.group(1).lower()
        name     = m.group(2) or rel_word.title()
        age_str  = m.group(3)
        age      = int(age_str) if age_str else 35
        rel      = rel_keywords.get(rel_word, "Household")

        # Check for symptom in ±60 chars around this match
        start    = max(0, m.start() - 20)
        end      = min(len(note), m.end() + 60)
        snippet  = note_low[start:end]
        has_sym  = any(s in snippet for s in symptom_words)
        symptom  = next((s for s in symptom_words if s in snippet), None)

        contacts.append({
            "name":        name,
            "age":         age,
            "rel":         rel,
            "has_symptom": has_sym,
            "symptom":     symptom,
            "confidence":  0.75,
        })

    return contacts


# ─────────────────────────────────────────────────────────────────────────────
# WRITEBACK: NEW CONTACT FROM NOTE
# Adds or updates a contact node + edge in the graph based on NER extraction.
# ─────────────────────────────────────────────────────────────────────────────

def writeback_new_contact(gc, producer, patient_id: str, asha_id: str,
                           contact: dict, district: str) -> dict:
    """
    Upsert a contact node extracted from an ASHA note into the graph.
    If the contact already exists (by name), update their vulnerability score.
    Always upserts the household_contact / workplace_contact edge.

    contact dict: {name, age, rel, has_symptom, symptom, confidence}

    Returns a delta dict describing what was added/updated.
    """
    name  = contact["name"]
    age   = contact.get("age", 35)
    rel   = contact.get("rel", "Household")
    vuln  = 1.5 if age < 10 or age > 60 else 1.0
    # Symptom in a contact = high transmission risk → bump vulnerability
    if contact.get("has_symptom"):
        vuln = min(vuln * 1.8, 3.0)

    cid       = f"CONTACT_{safe(name).replace(' ','_').replace('.','')}"
    edge_label = "household_contact" if rel == "Household" else "workplace_contact"
    base_w     = 0.9 if rel == "Household" else 0.6
    edge_w     = round(base_w * vuln, 4)
    decay      = 0.02 if rel == "Household" else 0.05

    # Upsert contact node — pk ONLY in addV() branch, never on update branch
    run_query(gc,
        f"g.V('{safe(cid)}').fold().coalesce("
        f"unfold()"
        f".property('name',           '{safe(name)}')"
        f".property('age',            {age})"
        f".property('vulnerability',  {vuln})"
        f".property('rel',            '{safe(rel)}')"
        f".property('screened',       false)"
        f".property('note_sourced',   true)"
        f".property('has_symptom',    {str(contact.get('has_symptom', False)).lower()})"
        f".property('symptom',        '{safe(contact.get('symptom') or '')}')"
        f".property('source_patient', '{safe(patient_id)}'),"
        f"addV('contact')"
        f".property('id',             '{safe(cid)}')"
        f".property('pk',             '{safe(district)}')"
        f".property('name',           '{safe(name)}')"
        f".property('age',            {age})"
        f".property('vulnerability',  {vuln})"
        f".property('rel',            '{safe(rel)}')"
        f".property('screened',       false)"
        f".property('note_sourced',   true)"
        f".property('has_symptom',    {str(contact.get('has_symptom', False)).lower()})"
        f".property('symptom',        '{safe(contact.get('symptom') or '')}')"
        f".property('source_patient', '{safe(patient_id)}')"
        f".property('district',       '{safe(district)}')"
        f")"
    )

    # Upsert edge patient → contact
    # .to() must be closed before .property() calls — Cosmos Gremlin requirement
    run_query(gc,
        f"g.V('{safe(patient_id)}').outE('{edge_label}')"
        f".where(inV().hasId('{safe(cid)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',                    {edge_w})"
        f".property('last_interaction_days_ago', 0)"
        f".property('note_sourced',              true),"
        f"g.V('{safe(patient_id)}').addE('{edge_label}')"
        f".to(g.V('{safe(cid)}'))"
        f".property('weight',                    {edge_w})"
        f".property('base_weight',               {base_w})"
        f".property('last_interaction_days_ago', 0)"
        f".property('decay_rate',                {decay})"
        f".property('note_sourced',              true)"
        f")"
    )

    publish_event(producer, edge_label, patient_id, cid, {
        "weight":       edge_w,
        "note_sourced": True,
        "has_symptom":  contact.get("has_symptom", False),
    })

    delta = {
        "change":      "contact_added_from_note",
        "contact_id":  cid,
        "contact_name":name,
        "age":         age,
        "rel":         rel,
        "edge_label":  edge_label,
        "edge_weight": edge_w,
        "has_symptom": contact.get("has_symptom", False),
        "symptom":     contact.get("symptom"),
        "vulnerability_score": vuln,
    }
    print(f"  [Writeback] new contact from note: {name} ({rel}, age {age}) "
          f"→ {patient_id} | edge={edge_label} w={edge_w}"
          f"{' ⚠ HAS SYMPTOM: ' + str(contact.get('symptom')) if contact.get('has_symptom') else ''}")
    return delta


def writeback_symptom_flag(gc, producer, patient_id: str, contact_name: str,
                            symptom: str) -> dict:
    """
    Flag a symptom on an existing contact node.
    Boosts vulnerability score → raises edge weight → propagates via PageRank.
    Called when note says 'wife has been coughing' for an already-known contact.
    """
    cid = f"CONTACT_{safe(contact_name).replace(' ','_').replace('.','')}"

    # Read current vulnerability
    vuln_r = run_query(gc, f"g.V('{safe(cid)}').values('vulnerability')")
    old_vuln = float(vuln_r[0]) if vuln_r else 1.0
    new_vuln = round(min(old_vuln * 1.8, 3.0), 4)

    run_query(gc,
        f"g.V('{safe(cid)}')"
        f".property('has_symptom',  true)"
        f".property('symptom',      '{safe(symptom)}')"
        f".property('vulnerability', {new_vuln})"
        f".property('screened',     false)"
    )

    # Recalculate edge weight from patient → this contact
    for edge_label in ["household_contact", "workplace_contact"]:
        old_w_r = run_query(gc,
            f"g.V('{safe(patient_id)}').outE('{edge_label}')"
            f".where(inV().hasId('{safe(cid)}')).values('base_weight')"
        )
        if old_w_r:
            base_w  = float(old_w_r[0])
            new_w   = round(base_w * new_vuln, 4)
            run_query(gc,
                f"g.V('{safe(patient_id)}').outE('{edge_label}')"
                f".where(inV().hasId('{safe(cid)}')).property('weight', {new_w})"
            )
            publish_event(producer, "symptom_flagged_on_contact", patient_id, cid, {
                "symptom":      symptom,
                "old_vuln":     old_vuln,
                "new_vuln":     new_vuln,
                "new_edge_w":   new_w,
            })
            print(f"  [Writeback] symptom '{symptom}' on {contact_name} "
                  f"| vuln {old_vuln} → {new_vuln} | edge weight → {new_w}")
            return {
                "change":       "symptom_flagged",
                "contact_name": contact_name,
                "contact_id":   cid,
                "symptom":      symptom,
                "vuln_old":     old_vuln,
                "vuln_new":     new_vuln,
            }

    return {"change": "symptom_flagged_no_edge", "contact_name": contact_name}


def writeback_note_to_patient(gc, producer, patient_id: str,
                               note: str, asha_id: str):
    """
    Append the ASHA note to the patient node as a timestamped audit field.
    Also updates free_text_note — picked up by NER on next pipeline run.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    note_key  = f"note_{timestamp}"
    # Cosmos DB Gremlin: store note as a property with a unique key suffix
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('{note_key}', '{safe(note[:500])}')"  # cap at 500 chars
        f".property('last_note_at', '{timestamp}')"
        f".property('asha_id', '{safe(asha_id)}')"
    )
    publish_event(producer, "note_written", asha_id, patient_id,
                  {"note_preview": note[:100], "timestamp": timestamp})
    print(f"  [Writeback] note stored on {patient_id}: '{note[:60]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# OVERNIGHT BATCH PROCESSOR
# Called once per day (by Azure Function timer or main.py --overnight flag).
# Reads all pending notes, runs NER, applies all graph writebacks, rescores.
# ─────────────────────────────────────────────────────────────────────────────

PENDING_NOTES_FILE = "data/pending_notes.json"


def queue_note_for_overnight(patient_id: str, asha_id: str,
                              note: str, action: str = "free_text"):
    """
    Append a note to the overnight processing queue.
    Called immediately when ASHA submits a note — no NER runs yet.
    NER runs at end of day in process_overnight_notes().
    """
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)

    pending = []
    if Path(PENDING_NOTES_FILE).exists():
        with open(PENDING_NOTES_FILE) as f:
            pending = json.load(f)

    pending.append({
        "patient_id":  patient_id,
        "asha_id":     asha_id,
        "note":        note,
        "action":      action,
        "queued_at":   datetime.now(timezone.utc).isoformat(),
        "processed":   False,
    })

    with open(PENDING_NOTES_FILE, "w") as f:
        json.dump(pending, f, indent=2)
    print(f"  [Queue] Note queued for overnight NER: {patient_id} — '{note[:50]}...'")


def process_overnight_notes(gc, producer, json_path: str = "nikshay_scored_dataset.json") -> dict:
    """
    End-of-day batch processor. Runs NER on all pending ASHA notes and
    applies graph writebacks. Called by Azure Function timer at 22:00 IST
    or by main.py --overnight.

    For each note:
      1. Run NER → extract contacts + intents
      2. writeback_note_to_patient() — store note on patient node
      3. For each new contact → writeback_new_contact()
      4. For symptomatic contacts → writeback_symptom_flag()
      5. For dose_confirmed intent → writeback_dose_confirmed()
      6. Rescore patient locally → update JSON
      7. Mark note as processed

    Returns a summary dict for display in the Officer dashboard.
    """
    from pathlib import Path
    from stage3_score import (compute_bbn_prior, compute_risk_score_v8,
                               compute_tgn_weight, apply_urgency_multiplier,
                               assign_risk_tier, get_adaptive_thresholds)

    if not Path(PENDING_NOTES_FILE).exists():
        print("  [Overnight] No pending notes file found.")
        return {"processed": 0, "graph_deltas": [], "errors": []}

    with open(PENDING_NOTES_FILE) as f:
        pending = json.load(f)

    unprocessed = [n for n in pending if not n.get("processed")]
    if not unprocessed:
        print("  [Overnight] No unprocessed notes.")
        return {"processed": 0, "graph_deltas": [], "errors": []}

    print(f"  [Overnight] Processing {len(unprocessed)} pending notes...")

    # Load patient JSON for local rescore
    patients_json = Path(json_path)
    if not patients_json.exists():
        patients_json = Path("data") / json_path
    all_patients = []
    if patients_json.exists():
        with open(patients_json, encoding="utf-8") as f:
            all_patients = json.load(f)
    patient_map = {p["patient_id"]: p for p in all_patients}

    lc          = get_language_client()
    graph_deltas = []
    errors       = []

    for entry in unprocessed:
        pid    = entry["patient_id"]
        asha   = entry["asha_id"]
        note   = entry["note"]
        action = entry.get("action", "free_text")

        try:
            # 1 — NER
            ner_result = run_ner_on_note(note, pid, lc)
            intents    = ner_result["intents"]
            contacts   = ner_result["contacts"]

            deltas_for_note = {
                "patient_id":    pid,
                "asha_id":       asha,
                "note_preview":  note[:80],
                "processed_at":  datetime.now(timezone.utc).isoformat(),
                "contacts_added": [],
                "symptoms_flagged": [],
                "dose_update":   None,
                "score_change":  None,
            }

            # 2 — Store note on patient node in graph
            if gc:
                writeback_note_to_patient(gc, producer, pid, note, asha)

            # 3+4 — Process extracted contacts
            patient_record = patient_map.get(pid, {})
            district = patient_record.get("location", {}).get("district", "Chennai")

            for contact in contacts:
                # Check if contact already known
                existing_contacts = patient_record.get("contact_network", [])
                known_names = {c["name"].lower() for c in existing_contacts}
                is_known = contact["name"].lower() in known_names

                if is_known and contact.get("has_symptom"):
                    # Contact known + new symptom → symptom flag only
                    if gc:
                        delta = writeback_symptom_flag(
                            gc, producer, pid, contact["name"], contact["symptom"]
                        )
                    else:
                        delta = {"change": "symptom_flagged_offline",
                                 "contact_name": contact["name"],
                                 "symptom": contact["symptom"]}
                    deltas_for_note["symptoms_flagged"].append(delta)

                    # Update local JSON contact too
                    for c in patient_record.get("contact_network", []):
                        if c["name"].lower() == contact["name"].lower():
                            c["has_symptom"]    = True
                            c["symptom"]        = contact["symptom"]
                            c["vulnerability_score"] = min(
                                c.get("vulnerability_score", 1.0) * 1.8, 3.0
                            )
                            break

                elif not is_known:
                    # Brand new contact from note → add to graph + local JSON
                    if gc:
                        delta = writeback_new_contact(
                            gc, producer, pid, asha, contact, district
                        )
                    else:
                        delta = {
                            "change":       "contact_added_offline",
                            "contact_name": contact["name"],
                            "rel":          contact.get("rel", "Household"),
                            "has_symptom":  contact.get("has_symptom", False),
                            "symptom":      contact.get("symptom"),
                        }
                    deltas_for_note["contacts_added"].append(delta)

                    # Add to local JSON contact_network
                    if patient_record:
                        patient_record.setdefault("contact_network", []).append({
                            "name":              contact["name"],
                            "age":               contact.get("age", 35),
                            "rel":               contact.get("rel", "Household"),
                            "has_comorbidity":   False,
                            "vulnerability_score": min(
                                (1.5 if contact.get("age", 35) < 10 or contact.get("age", 35) > 60 else 1.0)
                                * (1.8 if contact.get("has_symptom") else 1.0), 3.0
                            ),
                            "screened":          False,
                            "note_sourced":      True,
                            "symptom":           contact.get("symptom"),
                        })

            # 5 — Dose confirmed from note intent
            if intents.get("dose_confirmed") and action != "could_not_visit":
                if gc:
                    dose_delta = writeback_dose_confirmed(gc, producer, pid, asha)
                else:
                    dose_delta = {"change": "dose_confirmed_offline"}
                if patient_record:
                    patient_record["adherence"]["days_since_last_dose"] = 0
                    patient_record["operational"]["last_asha_visit_days_ago"] = 0
                deltas_for_note["dose_update"] = "dose_confirmed"

            elif intents.get("dose_missed") or action == "could_not_visit":
                if gc:
                    writeback_dose_missed(gc, producer, pid, asha)
                if patient_record:
                    patient_record["operational"]["last_asha_visit_days_ago"] = (
                        patient_record["operational"].get("last_asha_visit_days_ago", 0) + 1
                    )
                deltas_for_note["dose_update"] = "dose_missed"

            # 6 — Rescore patient with updated contact network and clinical data
            if patient_record:
                old_score = patient_record.get("risk_score", 0)
                old_tier  = patient_record.get("risk_level", "?")

                bbn_result  = compute_bbn_prior(patient_record)
                bbn_score   = bbn_result["score"]
                tgn_score   = patient_record.get("tgn_score", bbn_score)
                asha_load   = patient_record.get("asha_load_score", 0.3)

                # Contact network risk adjustment — applied to TGN component
                # since TGN captures graph structure. Each symptomatic household contact
                # adds transmission risk that would appear in TGN on a full rerun.
                contact_risk_delta = 0.0
                for c in deltas_for_note.get("contacts_added", []):
                    if c.get("has_symptom"):
                        contact_risk_delta += 0.08 if c.get("edge_label") == "household_contact" else 0.04
                    else:
                        contact_risk_delta += 0.02 if c.get("edge_label") == "household_contact" else 0.01
                for s in deltas_for_note.get("symptoms_flagged", []):
                    contact_risk_delta += 0.08

                adjusted_tgn = min(tgn_score + contact_risk_delta, 1.0)
                if contact_risk_delta > 0:
                    print(f"  [Overnight] Contact risk adjustment for {pid}: "
                          f"tgn {tgn_score:.3f} → {adjusted_tgn:.3f} "
                          f"(+{contact_risk_delta:.3f} from {len(deltas_for_note.get('contacts_added',[]))} new contacts, "
                          f"{len(deltas_for_note.get('symptoms_flagged',[]))} symptom flags)")

                week        = min(patient_record["clinical"]["total_treatment_days"] // 7, 26)
                regimen     = (patient_record.get("diagnosis") or {}).get("regimen") or \
                              (patient_record.get("clinical") or {}).get("regimen") or "DS-TB"
                composition = compute_risk_score_v8(
                    patient_record, adjusted_tgn, bbn_score, asha_load,
                    treatment_week=week, regimen=regimen,
                )
                new_score   = apply_urgency_multiplier(composition["composite_score"], week)
                new_tier    = assign_risk_tier(new_score, week)
                patient_record["tgn_score"] = adjusted_tgn

                patient_record["previous_risk_score"] = old_score
                patient_record["risk_score"]           = new_score
                patient_record["risk_level"]           = new_tier
                patient_record["risk_velocity"]        = round(new_score - old_score, 4)
                patient_record["score_composition"]    = composition

                # Write score back to graph node
                if gc:
                    writeback_risk_scores(gc, [patient_record])

                deltas_for_note["score_change"] = {
                    "old_score": round(old_score, 3),
                    "new_score": round(new_score, 3),
                    "old_tier":  old_tier,
                    "new_tier":  new_tier,
                    "changed":   old_tier != new_tier,
                }

            graph_deltas.append(deltas_for_note)
            entry["processed"]    = True
            entry["processed_at"] = datetime.now(timezone.utc).isoformat()
            entry["ner_result"]   = {
                "contacts_found": len(contacts),
                "intents":        intents,
            }

        except Exception as e:
            errors.append({"patient_id": pid, "error": str(e)})
            print(f"  [Overnight] ERROR processing {pid}: {e}")

    # Save updated patient JSON
    if all_patients and patients_json.exists():
        with open(patients_json, "w", encoding="utf-8") as f:
            json.dump(all_patients, f, indent=2)
        print(f"  [Overnight] Updated {patients_json} with rescored patients")

    # Mark notes as processed
    with open(PENDING_NOTES_FILE, "w") as f:
        json.dump(pending, f, indent=2)

    tier_changes    = sum(1 for d in graph_deltas if d.get("score_change", {}).get("changed"))
    contacts_added  = sum(len(d.get("contacts_added", [])) for d in graph_deltas)
    symptoms_flagged= sum(len(d.get("symptoms_flagged", [])) for d in graph_deltas)

    print(f"\n  [Overnight] Complete:")
    print(f"    Notes processed:   {len(graph_deltas)}")
    print(f"    New contacts:      {contacts_added}")
    print(f"    Symptoms flagged:  {symptoms_flagged}")
    print(f"    Tier changes:      {tier_changes}")
    print(f"    Errors:            {len(errors)}")

    return {
        "processed":        len(graph_deltas),
        "contacts_added":   contacts_added,
        "symptoms_flagged": symptoms_flagged,
        "tier_changes":     tier_changes,
        "errors":           errors,
        "graph_deltas":     graph_deltas,
        "run_at":           datetime.now(timezone.utc).isoformat(),
    }


def process_asha_reply(gc, producer, message_text: str,
                       patient_id: str, asha_id: str) -> dict:
    """Route ASHA reply to correct writeback function and publish to Event Hubs."""
    msg   = message_text.strip().lower()
    event = {"asha_id": asha_id, "patient_id": patient_id, "raw_message": message_text}

    if msg in ["done", "✓", "1"]:
        event["event_type"] = "dose_confirmed"
        if gc: writeback_dose_confirmed(gc, producer, patient_id, asha_id)
    elif msg in ["could not visit", "not home", "2"]:
        event["event_type"] = "dose_missed"
        if gc: writeback_dose_missed(gc, producer, patient_id, asha_id)
    elif msg in ["issue to report", "issue", "3"]:
        event["event_type"]   = "issue_flagged"
        event["flag_officer"] = True
        publish_event(producer, "issue_flagged", asha_id, patient_id, {})
    else:
        event["event_type"]   = "free_text_update"
        event["requires_ner"] = True

    return event


# ── Main ingestion orchestrator ───────────────────────────────────────────────

def ingest_all(gc, producer, records: list, limit: int = 1000):
    """
    Upsert-based — never calls clear_graph(). Safe to run multiple times.
    silence injection is done in main.py, not here, to avoid double injection.
    """
    records = records[:limit]
    print(f"  Building summaries for {len(records)} records...")

    asha_summaries    = build_asha_summaries(records)
    cho_summaries     = build_cho_summaries(records)
    village_summaries = build_village_summaries(records)
    district          = records[0]["location"]["district"]
    block             = records[0]["location"]["block"]

    print(f"  Upserting PHC node ({block})...")
    phc_id    = ingest_phc_node(gc, producer, district, block)

    print(f"  Upserting welfare scheme node...")
    scheme_id = ingest_welfare_scheme_node(gc, producer, district)

    print(f"  Upserting {len(asha_summaries)} ASHA nodes...")
    for s in asha_summaries.values():
        ingest_asha_node(gc, producer, s)

    print(f"  Upserting {len(cho_summaries)} CHO nodes...")
    for s in cho_summaries.values():
        ingest_cho_node(gc, producer, s)

    print(f"  Upserting {len(village_summaries)} village nodes...")
    for s in village_summaries.values():
        ingest_village_node(gc, producer, s)

    print(f"  Upserting {len(records)} patient nodes + edges...")
    contact_registry = {}
    new_c = bridges = 0

    for i, record in enumerate(records):
        try:
            aid      = record["operational"]["asha_id"]
            asha_sum = asha_summaries.get(aid, {"load_score": 0.3, "caseload": 10})

            ingest_patient_node(gc, producer, record)

            for contact in record["contact_network"]:
                cid, is_new = ingest_contact_node(gc, producer, record,
                                                   contact, contact_registry)
                ingest_contact_edge(gc, producer, record, contact, cid)
                if is_new: new_c += 1
                else:      bridges += 1

            ingest_asha_patient_edge(gc, producer, record, asha_sum)
            ingest_cho_patient_edge(gc, producer, record, cho_summaries)
            ingest_phc_patient_edge(gc, producer, record, phc_id)
            ingest_welfare_edge(gc, producer, record, scheme_id)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(records)} patients upserted...")
        except Exception as e:
            print(f"  Error on {record.get('patient_id','?')}: {e}")
            continue

    print(f"\n  Ingestion complete (upsert — existing data preserved)")
    print(f"  {len(records)} patients · {len(asha_summaries)} ASHAs · {len(cho_summaries)} CHOs · {len(village_summaries)} villages")
    print(f"  {new_c} contact nodes · {bridges} shared_contact bridges")
    print(f"  Node types (7): Patient, ASHA, CHO, PHC, WelfareScheme, Village, Contact")
    print(f"  Edge types (7): household_contact, workplace_contact, shared_contact,")
    print(f"                  assigned_to, assessed_by, attends, enrolled_in")