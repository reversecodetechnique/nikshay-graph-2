"""
cosmos_client.py
================
Azure Cosmos DB (Gremlin API) connection and query layer for Nikshay-Graph.

Centralises all Cosmos DB connection logic in one place so every stage
imports from here rather than each file managing its own client.

Provides:
  - get_client()             : returns a connected GremlinClient
  - run_query(gc, q)         : executes a Gremlin query, returns results
  - close_client(gc)         : graceful shutdown
  - upsert_vertex(...)       : insert or update any node type
  - upsert_edge(...)         : insert or update any edge type
  - get_patient(gc, pid)     : fetch a single patient node by ID
  - get_high_risk_patients() : Gremlin traversal for risk > threshold
  - get_unscreened_contacts(): contacts not yet screened, ordered by vulnerability
  - get_asha_patients(...)   : all patients assigned to one ASHA worker
  - update_memory_vector()   : write TGN memory back to patient node
  - health_check()           : verify connection before pipeline runs

Usage:
    from cosmos_client import get_client, run_query, close_client

    gc = get_client()
    results = run_query(gc, "g.V().hasLabel('patient').count()")
    print(results)
    close_client(gc)
"""

import os
import sys
import json
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Fix async event loop on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_client():
    """
    Returns a connected Gremlin client for Azure Cosmos DB.

    Reads from .env:
        COSMOS_ENDPOINT  — wss://your-account.gremlin.cosmos.azure.com:443/
        COSMOS_KEY       — your primary key
        COSMOS_DATABASE  — database name (default: NikshayDB)
        COSMOS_GRAPH     — graph/collection name (default: PatientGraph)

    Raises:
        EnvironmentError if COSMOS_ENDPOINT or COSMOS_KEY are not set.
        ConnectionError if the connection cannot be established.
    """
    try:
        from gremlin_python.driver import client as gremlin_client, serializer
    except ImportError:
        raise ImportError(
            "gremlinpython not installed. Run: pip install gremlinpython"
        )

    endpoint = os.getenv("COSMOS_ENDPOINT")
    key      = os.getenv("COSMOS_KEY")
    database = os.getenv("COSMOS_DATABASE", "NikshayDB")
    graph    = os.getenv("COSMOS_GRAPH", "PatientGraph")

    if not endpoint:
        raise EnvironmentError(
            "COSMOS_ENDPOINT not set in .env\n"
            "Expected format: wss://nikshay-graph-db.gremlin.cosmos.azure.com:443/"
        )
    if not key:
        raise EnvironmentError(
            "COSMOS_KEY not set in .env\n"
            "Get this from: Cosmos DB resource → Settings → Keys → Primary Key"
        )

    try:
        gc = gremlin_client.Client(
            endpoint, "g",
            username=f"/dbs/{database}/colls/{graph}",
            password=key,
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        logger.info(f"Cosmos DB connected: {database}/{graph}")
        return gc

    except Exception as e:
        raise ConnectionError(
            f"Could not connect to Cosmos DB at {endpoint}\n"
            f"Error: {e}\n"
            f"Check that your endpoint and key are correct, and that the "
            f"Gremlin API (not NoSQL) is enabled on your Cosmos DB account."
        )


def close_client(gc):
    """Gracefully close the Gremlin client connection."""
    try:
        if gc:
            gc.close()
            logger.info("Cosmos DB connection closed.")
    except Exception as e:
        logger.warning(f"Error closing Cosmos DB connection: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_query(gc, query: str, bindings: dict = None) -> list:
    """
    Execute a Gremlin query and return results as a list.

    Args:
        gc      : Gremlin client from get_client()
        query   : Gremlin traversal string
        bindings: optional parameter bindings dict (not used by Cosmos free tier)

    Returns:
        List of results. Empty list on error.
    """
    try:
        if bindings:
            result = gc.submit(query, bindings).all().result()
        else:
            result = gc.submit(query).all().result()
        return result
    except Exception as e:
        logger.error(f"Gremlin query failed: {e}\nQuery: {query[:200]}")
        return []


def safe(val) -> str:
    """Escape single quotes for safe Gremlin string interpolation."""
    return str(val).replace("'", "\\'")


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

def health_check(gc=None) -> bool:
    """
    Verify the Cosmos DB connection is live before running the pipeline.
    Creates its own connection if none is passed.

    Returns True if connection works, False otherwise.
    Prints a clear message either way — safe to call at pipeline startup.
    """
    owned = False
    try:
        if gc is None:
            gc     = get_client()
            owned  = True
        result = run_query(gc, "g.V().count()")
        count  = result[0] if result else 0
        print(f"✓ Cosmos DB connection healthy. Graph contains {count} vertices.")
        return True
    except (EnvironmentError, ConnectionError, ImportError) as e:
        print(f"✗ Cosmos DB connection failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Cosmos DB health check error: {e}")
        return False
    finally:
        if owned and gc:
            close_client(gc)


# ─────────────────────────────────────────────────────────────────────────────
# VERTEX (NODE) OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def upsert_vertex(gc, label: str, vertex_id: str,
                  properties: dict, partition_key: str) -> bool:
    """
    Insert a vertex if it does not exist, or update its properties if it does.
    Uses Gremlin coalesce() pattern — safe to call repeatedly.

    Args:
        gc            : Gremlin client
        label         : vertex label e.g. 'patient', 'asha_worker', 'contact'
        vertex_id     : unique string ID for this vertex
        properties    : dict of property_name → value
        partition_key : value for the /district partition key

    Returns:
        True on success, False on error.
    """
    # Build property chain
    props = ""
    for k, v in properties.items():
        if isinstance(v, bool):
            props += f".property('{safe(k)}', {str(v).lower()})"
        elif isinstance(v, (int, float)):
            props += f".property('{safe(k)}', {v})"
        else:
            props += f".property('{safe(k)}', '{safe(str(v))}')"

    query = (
        f"g.V('{safe(vertex_id)}').fold().coalesce("
        f"  unfold(){props},"
        f"  addV('{safe(label)}')"
        f"  .property('id', '{safe(vertex_id)}')"
        f"  .property('pk', '{safe(partition_key)}')"
        f"  {props}"
        f")"
    )
    result = run_query(gc, query)
    return result is not None


def upsert_edge(gc, edge_label: str,
                from_id: str, to_id: str,
                properties: dict = None) -> bool:
    """
    Add an edge between two vertices if it doesn't already exist.

    Args:
        gc          : Gremlin client
        edge_label  : edge label e.g. 'household_contact', 'assigned_to'
        from_id     : source vertex ID
        to_id       : target vertex ID
        properties  : optional dict of edge properties

    Returns:
        True on success, False on error.
    """
    props = ""
    if properties:
        for k, v in properties.items():
            if isinstance(v, bool):
                props += f".property('{safe(k)}', {str(v).lower()})"
            elif isinstance(v, (int, float)):
                props += f".property('{safe(k)}', {v})"
            else:
                props += f".property('{safe(k)}', '{safe(str(v))}')"

    query = (
        f"g.V('{safe(from_id)}')"
        f".coalesce("
        f"  outE('{safe(edge_label)}').where(inV().hasId('{safe(to_id)}')){props},"
        f"  addE('{safe(edge_label)}').to(g.V('{safe(to_id)}')){props}"
        f")"
    )
    result = run_query(gc, query)
    return result is not None


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def get_patient(gc, patient_id: str) -> Optional[dict]:
    """
    Fetch a single patient node by ID.
    Returns the property map dict, or None if not found.
    """
    result = run_query(gc,
        f"g.V('{safe(patient_id)}').hasLabel('patient').valueMap(true)"
    )
    if not result:
        return None
    # Gremlin valueMap returns lists for each property — unwrap them
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
            for k, v in result[0].items()}


def get_high_risk_patients(gc, threshold: float = 0.65,
                            district: str = "Chennai") -> list:
    """
    Return all patient nodes with risk_score above threshold.
    Ordered by risk_score descending.

    This is the query that seeds the PageRank personalization vector.
    """
    result = run_query(gc,
        f"g.V().hasLabel('patient')"
        f".has('district', '{safe(district)}')"
        f".has('risk_score', gt({threshold}))"
        f".order().by('risk_score', decr)"
        f".valueMap('id', 'risk_score', 'phase', 'days_missed', "
        f"'asha_id', 'block', 'treatment_week', 'silence')"
    )
    return [
        {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
         for k, v in node.items()}
        for node in (result or [])
    ]


def get_unscreened_contacts(gc, district: str = "Chennai",
                             min_vulnerability: float = 1.0) -> list:
    """
    Return unscreened contacts above a vulnerability threshold,
    ordered by vulnerability descending.

    These are the contacts the contact screening list is built from
    before PageRank re-ranks them by propagated network risk.
    """
    result = run_query(gc,
        f"g.V().hasLabel('contact')"
        f".has('district', '{safe(district)}')"
        f".has('screened', false)"
        f".has('vulnerability', gte({min_vulnerability}))"
        f".order().by('vulnerability', decr)"
        f".valueMap('id', 'name', 'age', 'rel', 'vulnerability')"
    )
    return [
        {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
         for k, v in node.items()}
        for node in (result or [])
    ]


def get_asha_patients(gc, asha_id: str) -> list:
    """
    Return all patients assigned to a given ASHA worker,
    ordered by risk_score descending.

    Used by Stage 5 to build the personalised morning briefing.
    """
    result = run_query(gc,
        f"g.V('{safe(asha_id)}').hasLabel('asha_worker')"
        f".outE('assigned_to').inV().hasLabel('patient')"
        f".order().by('risk_score', decr)"
        f".valueMap(true)"
    )
    return [
        {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
         for k, v in node.items()}
        for node in (result or [])
    ]


def get_silent_patients(gc, min_silence_days: int = 7,
                         district: str = "Chennai") -> list:
    """
    Return patients with silence events (no contact for min_silence_days+).
    These are patients Stage 1 has flagged as disengaging.
    """
    result = run_query(gc,
        f"g.V().hasLabel('patient')"
        f".has('district', '{safe(district)}')"
        f".has('silence', true)"
        f".has('silence_days', gte({min_silence_days}))"
        f".order().by('silence_days', decr)"
        f".valueMap('id', 'risk_score', 'silence_days', 'asha_id', 'block')"
    )
    return [
        {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
         for k, v in node.items()}
        for node in (result or [])
    ]


def get_patients_by_block(gc, block: str,
                           district: str = "Chennai") -> list:
    """Return all patients in a given block, ordered by risk descending."""
    result = run_query(gc,
        f"g.V().hasLabel('patient')"
        f".has('district', '{safe(district)}')"
        f".has('block', '{safe(block)}')"
        f".order().by('risk_score', decr)"
        f".valueMap(true)"
    )
    return [
        {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
         for k, v in node.items()}
        for node in (result or [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONTACT BRIDGE QUERY
# ─────────────────────────────────────────────────────────────────────────────

def get_shared_contact_bridges(gc, district: str = "Chennai") -> list:
    """
    Find contact nodes connected to more than one patient.
    These are the bridge nodes that create indirect patient-patient links.

    Returns list of dicts: contact_name, patient_count, connected_patients[]
    """
    result = run_query(gc,
        f"g.V().hasLabel('contact')"
        f".has('district', '{safe(district)}')"
        f".has('patient_count', gt(1))"
        f".project('contact', 'patients')"
        f".by(valueMap('id', 'name', 'vulnerability', 'patient_count'))"
        f".by(__.in('household_contact', 'workplace_contact')"
        f"   .values('id').fold())"
    )
    return result or []


def get_high_risk_contacts_live(gc, risk_threshold: float = 0.65,
                                 district: str = "Chennai") -> list:
    """
    Live Gremlin traversal: high-risk patients → their unscreened contacts.
    Used in Stage 3 as the demo query to run against real Cosmos DB.

    Returns list of (patient_id, patient_risk, contact_name, contact_vuln, edge_weight).
    """
    result = run_query(gc,
        f"g.V().hasLabel('patient')"
        f".has('district', '{safe(district)}')"
        f".has('risk_score', gt({risk_threshold}))"
        f".as('p')"
        f".outE('household_contact', 'workplace_contact').as('e')"
        f".inV().has('screened', false).as('c')"
        f".select('p', 'e', 'c')"
        f".by(valueMap('id', 'risk_score', 'block', 'asha_id'))"
        f".by(valueMap('weight', 'rel_type'))"
        f".by(valueMap('name', 'age', 'vulnerability'))"
    )
    return result or []


# ─────────────────────────────────────────────────────────────────────────────
# TGN MEMORY PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def update_memory_vector(gc, patient_id: str,
                          memory_vector: list) -> bool:
    """
    Write the updated TGN memory vector back to the patient node in Cosmos DB.
    Called after each Stage 2 inference pass so patient history survives
    across sessions.

    Args:
        gc            : Gremlin client
        patient_id    : patient node ID
        memory_vector : list of 64 floats

    Returns:
        True on success, False on error.
    """
    mem_str = json.dumps(memory_vector)
    result  = run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('memory_vector', '{safe(mem_str)}')"
    )
    return result is not None


def get_memory_vector(gc, patient_id: str) -> Optional[list]:
    """
    Load a patient's TGN memory vector from Cosmos DB.
    Returns list of 64 floats, or None if not found.
    """
    result = run_query(gc,
        f"g.V('{safe(patient_id)}').values('memory_vector')"
    )
    if not result:
        return None
    try:
        return json.loads(result[0])
    except (json.JSONDecodeError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATISTICS (for dashboard + District Officer view)
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_stats(gc, district: str = "Chennai") -> dict:
    """
    Return summary statistics for the dashboard header metrics.
    Runs 4 lightweight count queries.
    """
    stats = {}
    try:
        stats["total_patients"] = (run_query(gc,
            f"g.V().hasLabel('patient').has('district', '{safe(district)}').count()"
        ) or [0])[0]

        stats["high_risk"] = (run_query(gc,
            f"g.V().hasLabel('patient').has('district', '{safe(district)}')"
            f".has('risk_score', gt(0.65)).count()"
        ) or [0])[0]

        stats["unscreened_contacts"] = (run_query(gc,
            f"g.V().hasLabel('contact').has('district', '{safe(district)}')"
            f".has('screened', false).count()"
        ) or [0])[0]

        stats["silent_patients"] = (run_query(gc,
            f"g.V().hasLabel('patient').has('district', '{safe(district)}')"
            f".has('silence', true).count()"
        ) or [0])[0]

        stats["asha_workers"] = (run_query(gc,
            f"g.V().hasLabel('asha_worker').has('district', '{safe(district)}').count()"
        ) or [0])[0]

        stats["total_edges"] = (run_query(gc, "g.E().count()") or [0])[0]

    except Exception as e:
        logger.error(f"get_graph_stats error: {e}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clear_graph(gc):
    """
    Drop all vertices and edges. Use before a fresh ingestion run.
    Warning: irreversible. Only call this during demo setup.
    """
    print("Clearing all graph data...")
    run_query(gc, "g.V().drop()")
    print("Graph cleared.")


def get_vertex_count(gc) -> int:
    """Return total number of vertices in the graph."""
    result = run_query(gc, "g.V().count()")
    return result[0] if result else 0


def get_edge_count(gc) -> int:
    """Return total number of edges in the graph."""
    result = run_query(gc, "g.E().count()")
    return result[0] if result else 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Nikshay-Graph — Cosmos DB Connection Test")
    print("=" * 50)

    # 1. Health check
    ok = health_check()
    if not ok:
        print("\nConnection failed. Check your .env file:")
        print("  COSMOS_ENDPOINT=wss://your-account.gremlin.cosmos.azure.com:443/")
        print("  COSMOS_KEY=your_primary_key")
        print("  COSMOS_DATABASE=NikshayDB")
        print("  COSMOS_GRAPH=PatientGraph")
        exit(1)

    gc = get_client()

    # 2. Graph stats
    print("\nGraph statistics:")
    stats = get_graph_stats(gc)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 3. Sample high-risk patients
    print("\nHigh-risk patients (risk > 0.65):")
    high_risk = get_high_risk_patients(gc, threshold=0.65)
    for p in high_risk[:5]:
        print(f"  {p.get('id','?')} | risk={p.get('risk_score','?')} | "
              f"block={p.get('block','?')} | silence={p.get('silence','?')}")
    if not high_risk:
        print("  None found — run main.py to populate the graph first")

    # 4. Unscreened contacts
    print("\nUnscreened contacts (vulnerability >= 1.5):")
    contacts = get_unscreened_contacts(gc, min_vulnerability=1.5)
    for c in contacts[:5]:
        print(f"  {c.get('name','?')} | age={c.get('age','?')} | "
              f"rel={c.get('rel','?')} | vuln={c.get('vulnerability','?')}")
    if not contacts:
        print("  None found")

    # 5. Live bridge query
    print("\nLive bridge query (high-risk patients → unscreened contacts):")
    bridges = get_high_risk_contacts_live(gc)
    for row in bridges[:3]:
        p = row.get('p', {})
        c = row.get('c', {})
        e = row.get('e', {})
        print(f"  Patient {p.get('id',['?'])} → {c.get('name',['?'])} "
              f"(edge weight={e.get('weight',['?'])})")
    if not bridges:
        print("  None found")

    close_client(gc)
    print("\nConnection test complete.")