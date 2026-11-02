# execute Part B alone in a fresh kernel, then append its (executed) cells to the existing executed notebook 22 (Part A outputs untouched)
import nbformat as nbf, sys
sys.path.insert(0,"scripts/build"); from nb22_partB import add_partB
from nbclient import NotebookClient
C=[]; add_partB(lambda s:C.append(nbf.v4.new_markdown_cell(s)), lambda s:C.append(nbf.v4.new_code_cell(s)))
nb=nbf.v4.new_notebook(); nb.cells=C; nb.metadata["kernelspec"]={"name":"cellflow","display_name":"cellflow","language":"python"}
NotebookClient(nb,timeout=1800,kernel_name="cellflow",resources={"metadata":{"path":"notebooks"}}).execute()
main=nbf.read("notebooks/22_representation_screen.ipynb",as_version=4)
main.cells=[c for c in main.cells if "Part B — agonist" not in "".join(c.source) ][:11]+nb.cells
nbf.write(main,"notebooks/22_representation_screen.ipynb"); print("appended",len(nb.cells),"cells")
