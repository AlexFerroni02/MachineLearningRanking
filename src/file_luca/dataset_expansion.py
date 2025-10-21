# dataset_expansion_combined.py
"""
Generates an expanded LOINC dataset formatted like loinc_dataset-v2.xlsx.
- Keeps all original queries and their data.
- Adds up to 50 total queries (sampling new ones).
- Keeps original LOINC terms + adds new relevant ones.
- Produces one Excel file with all sheets combined.
"""

import itertools
import pandas as pd
import random
import re
import os
from collections import Counter

# ---- CONFIG ----
N_TEMPLATES_PER_COMPONENT = 3
LOINC_CORE_PATH = "../../data/LoincTableCore/LoincTableCore.csv"
ORIGINAL_DATASET_PATH = "../../data/loinc_dataset-v2.xlsx"
OUTPUT_XLSX_PATH = "../../data/expanded_loinc_dataset.xlsx"
MAX_QUERIES = 50
MAX_LOINC_TERMS_PER_QUERY = 50  # max LOINC terms per query

# ---- BASE QUERIES ----
original_queries = [
    "glucose in blood",
    "bilirubin in plasma",
    "white blood cells count"
]

# ---- COMPONENTS, SYSTEMS, TEMPLATES ----
components = [
    "glucose", "bilirubin", "cholesterol", "creatinine", "urea", "lactate",
    "sodium", "potassium", "calcium", "magnesium", "chloride",
    "albumin", "hemoglobin", "ferritin", "triglycerides",
    "aspartate aminotransferase", "alanine aminotransferase",
    "alkaline phosphatase", "protein c",
    "cortisol", "insulin", "tsh",
    "white blood cells", "platelets", "erythrocytes", "neutrophils", "lymphocytes"
]
systems = ["blood", "serum", "plasma", "urine"]
templates = [
    "{component} in {system}",
    "{component} level in {system}",
    "{component} concentration in {system}",
    "measurement of {component} in {system}",
    "test for {component} in {system}"
]

# ---- STEP 1: GENERATE CANDIDATE QUERIES ----
all_combinations = list(itertools.product(components, systems, templates))
queries = [t.format(component=comp, system=sys) for comp, sys, t in all_combinations]

# Remove duplicates and originals
queries = sorted(set(queries))
extra_queries = sorted(set(queries) - set(original_queries))

# Balanced sampling per system
sampled_extra = []
queries_needed = MAX_QUERIES - len(original_queries)
queries_per_system = max(1, queries_needed // len(systems))

for sys in systems:
    sys_queries = [q for q in extra_queries if sys in q]
    n = min(len(sys_queries), queries_per_system)
    sampled_extra.extend(random.sample(sys_queries, n))

# Fill remaining queries randomly
remaining = queries_needed - len(sampled_extra)
if remaining > 0:
    remaining_queries = list(set(extra_queries) - set(sampled_extra))
    sampled_extra.extend(random.sample(remaining_queries, remaining))

final_queries = sorted(set(original_queries + sampled_extra))
print(f"Total selected queries (including originals): {len(final_queries)}")

# Check distribution
print("Distribution of systems in final queries:")
print(Counter([q.split()[-1] for q in final_queries]))

# ---- STEP 2: LOAD ORIGINAL DATASET ----
xls = pd.ExcelFile(ORIGINAL_DATASET_PATH)
original_sheets = {sheet: xls.parse(sheet, header=None) for sheet in xls.sheet_names}
print(f"Loaded {len(original_sheets)} original sheets.")

# ---- STEP 3: LOAD LOINC DATA ----
loinc = pd.read_csv(LOINC_CORE_PATH, low_memory=False)
loinc = loinc[['LOINC_NUM', 'LONG_COMMON_NAME', 'COMPONENT', 'SYSTEM', 'PROPERTY']]
loinc = loinc[loinc['SYSTEM'].str.contains("Blood|Serum|Plasma|Urine", case=False, na=False)]

# ---- STEP 4: FILTER RELEVANT TERMS PER QUERY ----
SYSTEM_MAP = {
    "blood": ["bld", "blood"],
    "serum": ["ser", "serum"],
    "plasma": ["plas", "plasma"],
    "urine": ["ur", "urine"]
}

filtered_loinc_list = []

for query in final_queries:
    system_in_query = query.split()[-1].lower()
    system_keywords = SYSTEM_MAP.get(system_in_query, [system_in_query])

    # Filter by system
    subset = loinc[loinc['SYSTEM'].str.lower().apply(lambda x: any(k in x for k in system_keywords))]

    # Filter by component keywords
    query_keywords = set(re.findall(r"[a-zA-Z]+", query.lower()))
    mask = subset['COMPONENT'].str.lower().apply(lambda x: any(k in x for k in query_keywords))
    subset_filtered = subset[mask]

    if len(subset_filtered) > MAX_LOINC_TERMS_PER_QUERY:
        subset_filtered = subset_filtered.sample(MAX_LOINC_TERMS_PER_QUERY, random_state=42)

    filtered_loinc_list.append(subset_filtered)

filtered_loinc = pd.concat(filtered_loinc_list).drop_duplicates()
print(f"Filtered and sampled total relevant LOINC terms: {len(filtered_loinc)}")

# ---- STEP 5: MERGE ORIGINAL + NEW DATA ----
os.makedirs(os.path.dirname(OUTPUT_XLSX_PATH), exist_ok=True)
writer = pd.ExcelWriter(OUTPUT_XLSX_PATH)

for query in final_queries:
    sheet_name = query[:31]
    if any(sheet_name.lower() in s.lower() for s in original_sheets.keys()):
        continue

    df = filtered_loinc.copy()
    df.columns = [c.lower() for c in df.columns]  # convert column names to lowercase
    cols = df.columns.tolist()

    query_row = pd.DataFrame([[f"Query: {query}"] + [""]*(len(cols)-1)], columns=cols)
    empty_row = pd.DataFrame([[""]*len(cols)], columns=cols)
    column_names_row = pd.DataFrame([cols], columns=cols)

    combined = pd.concat([query_row, empty_row, column_names_row, df], ignore_index=True)
    combined.to_excel(writer, sheet_name=sheet_name, header=False, index=False)

writer.close()
print(f"✅ Merged dataset saved to {OUTPUT_XLSX_PATH}")
