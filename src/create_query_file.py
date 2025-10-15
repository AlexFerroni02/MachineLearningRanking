import pandas as pd

# Insert the queries here to create the xlsx file with all the queries
queries = [
    "glucose in blood",
    "bilirubin in plasma",
    "White blood cells count"
]
OUTPUT_FILE = "../data/queries.xlsx"  

df_queries = pd.DataFrame({"query": queries})

df_queries.to_excel(OUTPUT_FILE, index=False)
print(f"Saved {len(queries)} queries to {OUTPUT_FILE}")
