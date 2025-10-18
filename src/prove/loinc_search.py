
import requests
import pandas as pd
import time

FHIR_BASE = "https://fhir.loinc.org"
HEADERS = {"Accept": "application/fhir+json"}

def fhir_search_loinc(query: str, rows: int = 20, enrich: bool = True) -> pd.DataFrame:
    """
    Cerca LOINC via FHIR ValueSet/$expand filtrando per testo e, opzionalmente, arricchisce con $lookup.
    """
    expand_url = f"{FHIR_BASE}/ValueSet/$expand"
    params = {
        "url": "http://loinc.org/vs/loinc",
        "filter": query,
        "count": rows,
        "_format": "json",
    }
    try:
        r = requests.get(expand_url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        contains = r.json().get("expansion", {}).get("contains", [])
        if not contains:
            return pd.DataFrame()

        records = []
        for c in contains:
            rec = {
                "loinc_num": c.get("code"),
                "long_common_name": c.get("display"),
                "system_uri": c.get("system"),
            }
            records.append(rec)

        df = pd.DataFrame(records)

        if enrich and not df.empty:
            enriched_rows = []
            for _, row in df.iterrows():
                details = fhir_lookup_properties(row["loinc_num"])
                enriched_rows.append({
                    **row.to_dict(),
                    "component": details.get("COMPONENT"),
                    "property": details.get("PROPERTY"),
                    "system": details.get("SYSTEM"),
                    "time_aspct": details.get("TIME_ASPCT"),
                    "scale_typ": details.get("SCALE_TYP"),
                    "method_typ": details.get("METHOD_TYP"),
                })
                time.sleep(0.15)  # piccola pausa per non saturare il servizio
            df = pd.DataFrame(enriched_rows)

        return df

    except requests.HTTPError as e:
        print(f"❌ Errore HTTP: {e.response.status_code} - {e.response.text[:200]}")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"❌ Errore di rete: {e}")
        return pd.DataFrame()

def fhir_lookup_properties(code: str) -> dict:
    """
    Recupera le proprietà LOINC di un codice via CodeSystem/$lookup.
    Ritorna una mappa tipo {'COMPONENT': '...', 'PROPERTY': '...', ...}.
    """
    url = f"{FHIR_BASE}/CodeSystem/$lookup"
    params = {
        "system": "http://loinc.org",
        "code": code,
        "_format": "json",
    }
    props = {}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        for p in data.get("parameter", []):
            if p.get("name") == "property":
                code_key = None
                value_val = None
                for part in p.get("part", []):
                    if part.get("name") == "code":
                        code_key = part.get("valueCode")
                    elif part.get("name") == "value":
                        value_val = (
                            part.get("valueString")
                            or part.get("valueCode")
                            or (part.get("valueCoding") or {}).get("code")
                        )
                if code_key and value_val is not None:
                    props[code_key] = value_val
        return props
    except requests.RequestException:
        return props

if __name__ == "__main__":
    query = "glucose in blood"
    print(f"🔍 Ricerca LOINC (FHIR) per: '{query}'...\n")
    results = fhir_search_loinc(query, rows=20, enrich=True)
    if not results.empty:
        print(results.to_string(index=False))
        # results.to_csv("loinc_results.csv", index=False)
    else:
        print("Nessun risultato trovato.")
