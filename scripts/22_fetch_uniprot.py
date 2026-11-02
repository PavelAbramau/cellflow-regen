"""Fetch UniProt (human, reviewed) 'Function' comment for each cytokine gene; cache to data/uniprot_function.csv"""
import json, re, sys, time, urllib.request, urllib.parse, pandas as pd, ast
GENES = json.load(open(sys.argv[1]))  # cytokine -> "G1+G2"
rows = []
for cyt, gs in GENES.items():
    texts, accs = [], []
    for g in gs.split("+"):
        q = f"(gene_exact:{g}) AND (organism_id:9606) AND (reviewed:true)"
        u = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode({"query": q, "fields": "accession,gene_names,cc_function", "format": "json", "size": 1})
        for attempt in range(3):
            try:
                d = json.load(urllib.request.urlopen(u, timeout=60)); break
            except Exception as e:
                time.sleep(2)
        res = d.get("results", [])
        if not res: continue
        r = res[0]; accs.append(r["primaryAccession"])
        for c in r.get("comments", []):
            if c.get("commentType") == "FUNCTION":
                texts += [t["value"] for t in c.get("texts", [])]
    rows.append(dict(cytokine=cyt, genes=gs, accessions=";".join(accs), function=" ".join(texts)))
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
print(pd.DataFrame(rows).function.str.len().describe())
