"""
run_notebook.py
Executes all cells in notebooks/retail_demand_forecasting.ipynb top-to-bottom
and saves the execution outputs directly into the notebook.
"""

import json
import io
import contextlib
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("notebooks/retail_demand_forecasting.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

global_env = {"display": print}
execution_count = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        print(f"Executing code cell {idx}...")
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, global_env)
            output_text = stdout_buf.getvalue()
            cell["execution_count"] = execution_count
            execution_count += 1
            cell["outputs"] = []
            if output_text:
                cell["outputs"].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True)
                })
        except Exception as e:
            err = traceback.format_exc()
            print(f"Error in cell {idx}: {err}")
            cell["outputs"].append({
                "name": "stderr",
                "output_type": "stream",
                "text": err.splitlines(keepends=True)
            })
            raise

with open("notebooks/retail_demand_forecasting.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("ALL NOTEBOOK CELLS EXECUTED CLEANLY TOP-TO-BOTTOM!")
