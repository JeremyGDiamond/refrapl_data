import os
import json
from datetime import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt

# Your lists
tools = ["No_Tools", "Perf", "PJ", "Ref_User", "Ref_Kern", "EB", "Scaphandre", "Turbostat", "CC"]
benchmarks = ["bt", "cg", "ft", "mg", "ep", "is", "mi", "sl"]

# Directory path
bdirectory = "./benches"
sdirectory = "./sleep/data"

# Regex pattern to extract run_number, tool, benchmark
pattern = re.compile(r"run_(\d+)_([A-Za-z_]+)_([a-z]+)_([\dT\.]+)\.json")

# Collect rows

def readFiles(directory):
    rows = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            match = pattern.match(filename)
            if match:
                # print (f"filename: {filename}")
                run_number = int(match.group(1))
                tool = match.group(2)
                benchmark = match.group(3)
                dt_part, frac_part = match.group(4).split(".")
                dt = datetime.strptime(dt_part, "%Y%m%dT%H%M%S")
                fts = dt.timestamp() + float("0." + frac_part)

                if tool in tools and benchmark in benchmarks:
                    filepath = os.path.join(directory, filename)
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    start = data[0]
                    end = data[1]

                    # Flatten start and end dicts with prefixes
                    row = {}
                    for key, value in start.items():
                        row[f"start_{key}"] = value
                    for key, value in end.items():
                        row[f"end_{key}"] = value

                    # Add file info
                    row["run_number"] = run_number
                    row["tool"] = tool
                    row["benchmark"] = benchmark
                    row["fts"] = fts
                    row["filename"] = filename

                    rows.append(row)
            elif (match == None):
                print (f"filename: {filename}")
                print(f"match: {match}")

    return pd.DataFrame(rows)

# Make DataFrame
df1 = readFiles(bdirectory)
df2 = readFiles(sdirectory)

df = pd.concat([df1, df2], ignore_index=True)

# print("Shape:", df.shape)
# print("Columns:", df.columns.tolist())

# # Create the pivot table
# matrix = df.groupby(["tool", "benchmark"]).size().reset_index(name="num_samples")
# pivot = matrix.pivot(index="tool", columns="benchmark", values="num_samples").fillna(0).astype(int)

# print(pivot)

runtable = df1[["run_number", "tool", "benchmark"]].sort_values(by="run_number")
# print(sortDupes)

runtable.to_csv("runtable_bench.csv", index=False)

runtable = df2[["run_number", "tool", "benchmark"]].sort_values(by="run_number")
# print(sortDupes)

runtable.to_csv("runtable_sleep_micro.csv", index=False)