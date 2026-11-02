import json,sys
nb=json.load(open(sys.argv[1])); lim=int(sys.argv[2]) if len(sys.argv)>2 else 3000
for i,c in enumerate(nb["cells"]):
    if c["cell_type"]!="code": continue
    for o in c["outputs"]:
        t=""
        if o["output_type"]=="stream": t="".join(o["text"])
        elif o["output_type"]=="execute_result": t="".join(o["data"].get("text/plain",[]))
        elif o["output_type"]=="error": t="ERR "+o["ename"]+": "+o["evalue"]
        if t: print(f"--- cell {i}\n{t[:lim]}")
