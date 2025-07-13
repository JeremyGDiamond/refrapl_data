import os
import json
from datetime import datetime
import pandas as pd
import re
import matplotlib.pyplot as plt
import csv
from pprint import pprint
from scipy.stats import shapiro
import seaborn as sns
from scipy.stats import kruskal

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# Lists
tools = ["No_Tools", "Ref_User", "Ref_Kern", "Perf", "PJ", "Turbostat", "Scaphandre", "CC"]
benchmarks = ["bt", "cg", "ft", "mg", "ep", "is", "mi", "sl"]

tools1 = ["Ref_User", "Ref_Kern", "Perf", "PJ", "Turbostat", "Scaphandre", "CC"]
benchmarks1 = ["bt", "cg", "ft", "mg", "ep", "is", "sl"]


# Directory path
bdirectory = "./benches"
sdirectory = "./sleep/data"
mbdirectory = "./sleep/data/microbench"
csdirectory = "./sleep/data/cstate"

# Regex pattern to extract run_number, tool, benchmark
pattern = re.compile(r"run_(\d+)_([A-Za-z_]+)_([a-z]+)_([\dT\.]+)\.json")
pattern2 = re.compile(r"run_(\d+)_([A-Za-z_]+)_([a-z]+)_([\dT\.]+)\.csv")
pattern3 = re.compile(r"run_(\d+)_([A-Za-z_]+)_([a-z]+)_([\dT\.]+)\.log")

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
                        if(key == 'aenergy'):
                            row[f"start_{key}_total"] = value["total"]                                             
                        else:
                            row[f"start_{key}"] = value
                    for key, value in end.items():
                        if(key == 'aenergy'):
                            row[f"end_{key}_total"] = value["total"]                
                        else:
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

def read_microbench(directory):

    rows = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = pattern2.match(filename)
            if match:
                # print (f"filename: {filename}")
                run_number = int(match.group(1))
                tool = match.group(2)
                benchmark = match.group(3)
                dt_part, frac_part = match.group(4).split(".")
                dt = datetime.strptime(dt_part, "%Y%m%dT%H%M%S")
                fts = dt.timestamp() + float("0." + frac_part)

                if tool == "No_Tools" and benchmark in benchmarks:
                    filepath = os.path.join(directory, filename)
                    with open(filepath, "r") as f:
                        reader = csv.reader(f)
                        data = list(reader)
                    
                    row = {}
                    row["run_number"] = run_number
                    row["tool"] = tool
                    row["benchmark"] = benchmark
                    row["fts"] = fts
                    row["filename"] = filename

                    # add data to rows 
                    row[data[1][0]] = data[1][1]
                    row[data[2][0]] = data[2][1]
                    row[data[3][0]] = data[3][1]
                    row[data[4][0]] = data[4][1]
                    row[data[5][0]] = data[5][1]
                    row[data[6][0]] = data[6][1]
                    row[data[7][0]] = data[7][1]
                    row[data[8][0]] = data[8][1]
                    row[data[9][0]] = data[9][1]
                    row[data[10][0]] = data[10][1]
                    row[data[11][0]] = data[11][1]
                    rows.append(row)

            elif (match == None):
               print (f"filename: {filename}")
               print(f"match: {match}")
                    
    return pd.DataFrame(rows)

def read_cstate(directory):
    
    rows = []

    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            match = pattern3.match(filename)
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
                    row = {}
                    count = 0
                    with open(filepath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue  # skip empty lines

                            if line.startswith("Core"):
                                continue # skip headers

                            if line.startswith("-"):
                                sline = line.split("\t")
                                row[f"C1% {count:04d}"] = float(sline[2])
                                row[f"C3% {count:04d}"] = float(sline[3])
                                row[f"C6% {count:04d}"] = float(sline[4])
                                row[f"C8% {count:04d}"] = float(sline[5])
                                row[f"C9% {count:04d}"] = float(sline[6])
                                count = count +1
                        
                    
                    row["run_number"] = run_number
                    row["tool"] = tool
                    row["benchmark"] = benchmark
                    row["fts"] = fts
                    row["filename"] = filename

                    # pprint.pprint(row)
                    rows.append(row)

            elif (match == None):
               print (f"filename3: {filename}")
               print(f"match3: {match}")
                    
    return pd.DataFrame(rows)

def sw_Test(df, type):
    results = []

    # Loop over each (tool, benchmark) pair
    for (tool, benchmark), group in df.groupby(['tool', 'benchmark']):
        data = group[type].dropna().values
        if len(data) >= 3:  # Shapiro requires at least 3 samples
            stat, p_value = shapiro(data)
            normal = p_value > 0.05
        else:
            stat, p_value, normal = None, None, None

        results.append({
            'tool': tool,
            'benchmark': benchmark,
            'shapiro_statistic': stat,
            'p_value': p_value,
            'normal': normal,
            'n_samples': len(data)
        })
    
    return pd.DataFrame(results)


def maxtix_of_boxs(df, quant, yname, outfile):

    # Create the catplot, but DO NOT share y-axis between rows
    g = sns.catplot(
        data=df,
        x="tool",
        y=quant,
        row="benchmark",
        kind="box",
        sharey=False,   # <-- key change!
        height=3,
        aspect=10
    )

    # Remove individual axis labels
    g.set_axis_labels("", "")  

    # Remove default facet titles
    g.set_titles("")  

    # Add one global y-axis label (with unit)
    g.figure.subplots_adjust(left=0.12)
    g.figure.text(0.04, 0.5, yname, va='center', rotation='vertical', fontsize=48)

    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=20)

    # Put benchmark labels on the right
    benchmarks = df_filtered['benchmark'].unique()
    for ax, row_val in zip(g.axes[:, 0], benchmarks):
        ax.annotate(
            row_val,
            xy=(0.99, 0.5),
            xycoords='axes fraction',
            ha='left',
            va='center',
            fontsize=28
        )

    # One global x-axis label
    g.figure.text(0.55, 0.003, "Tool", ha='center', fontsize=36)

    plt.subplots_adjust(bottom=0.05)
    plt.subplots_adjust(bottom=0.05)
    
    plt.savefig(outfile, dpi=300)


# Make DataFrame
df1 = readFiles(bdirectory)
df2 = readFiles(sdirectory)

df_benches = pd.concat([df1, df2], ignore_index=True)

# print("Shape:", df.shape)
print("Columns:", df_benches.columns.tolist())

# # Create the pivot table
# matrix = df.groupby(["tool", "benchmark"]).size().reset_index(name="num_samples")
# pivot = matrix.pivot(index="tool", columns="benchmark", values="num_samples").fillna(0).astype(int)

# print(pivot)

# runtable = df1[["run_number", "tool", "benchmark"]].sort_values(by="run_number")

# runtable.to_csv("runtable_bench.csv", index=False)

# runtable = df2[["run_number", "tool", "benchmark"]].sort_values(by="run_number")

# runtable.to_csv("runtable_sleep_micro.csv", index=False)

## import micro-bench data
mbdf = read_microbench(mbdirectory)

# Create the pivot table
# matrix = mbdf.groupby(["tool", "benchmark"]).size().reset_index(name="num_samples")
# pivot = matrix.pivot(index="tool", columns="benchmark", values="num_samples").fillna(0).astype(int)

# print(pivot)

## import c-state data

csdf = read_cstate(csdirectory)

# print(csdf)

# pivot tabels if means for benches

##  RAPL energy 
overflow = 262144.0 # sudo rdmsr -f 12:8 -d 0x606 = 14
### Parse to float
df_benches['start_RAPL_fl'] = pd.to_numeric(df_benches['start_RAPL'])
df_benches['end_RAPL_fl'] = pd.to_numeric(df_benches['end_RAPL'])

### overflow
df_benches['end_RAPL_fl'] = df_benches.apply(
    lambda row: row['end_RAPL_fl'] + overflow if row['end_RAPL_fl'] < row['start_RAPL_fl'] else row['end_RAPL_fl'],
    axis=1
)

### Compute the difference
df_benches['RAPL_diff'] = df_benches['end_RAPL_fl'] - df_benches['start_RAPL_fl']

## smart plug energy

df_benches["diff_aenergy_total"] = df_benches["end_aenergy_total"] - df_benches["start_aenergy_total"]

# make dropped table

mask = (df_benches['tool'] == 'Ref_User') & (~df_benches['benchmark'].isin(['mi', 'sl']))

# For each benchmark, find the index of the 15 rows with the lowest 
to_drop = (
    df_benches[mask]
    .groupby('benchmark', group_keys=False)
    .apply(lambda g: g.nsmallest(15, 'RAPL_diff'))
    .index
)

# Drop those rows by index
df_cleaned = df_benches.drop(to_drop)

##  Duration 
### Parse to datetime
df_cleaned['start_time'] = pd.to_datetime(df_cleaned['start_timestamp'], format="%Y%m%dT%H%M%S.%f")
df_cleaned['end_time'] = pd.to_datetime(df_cleaned['end_timestamp'], format="%Y%m%dT%H%M%S.%f")

### Compute the difference
df_cleaned['duration'] = df_cleaned['end_time'] - df_cleaned['start_time']
df_cleaned['duration_seconds'] = df_cleaned['duration'].dt.total_seconds()

def pivot_fold_median():

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='RAPL_diff',
        aggfunc='median'  # Average
    )

    print("\nRAPL_diff\n")

    pivot = pivot.loc[tools]

    print(pivot)  
    pivot.to_csv("./csv/RAPL_diff_median_pivot.csv")

    # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nRAPL_diff_rel\n")

    pivot_adjusted.to_csv("./csv/RAPL_rel_median_pivot.csv")
    print(pivot_adjusted)

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='duration_seconds',
        aggfunc='median'  # Average
    )

    print("\nDurations\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/dur_median_pivot.csv")

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nDurations_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/dur_median_rel_pivot.csv")


    # smart plug energy total 
    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='diff_aenergy_total',
        aggfunc='median'  # Average
    )

    print("\nSmart_Plug_Energy\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/splug_median_pivot.csv")

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nSmart_Plug_Energy_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/splug_median_rel_pivot.csv")

def pivot_fold_mean():

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='RAPL_diff',
        aggfunc='mean'  # Average
    )

    print("\nRAPL_diff\n")

    pivot = pivot.loc[tools]

    print(pivot)  
    pivot.to_csv("./csv/RAPL_diff_mean_pivot.csv")

    # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nRAPL_diff_rel\n")

    pivot_adjusted.to_csv("./csv/RAPL_rel_mean_pivot.csv")
    print(pivot_adjusted)

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='duration_seconds',
        aggfunc='mean'  # Average
    )

    print("\nDurations\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/dur_mean_pivot.csv")

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nDurations_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/dur_mean_rel_pivot.csv")


    # smart plug energy total 
    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='diff_aenergy_total',
        aggfunc='mean'  # Average
    )

    print("\nSmart_Plug_Energy\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/splug_mean_pivot.csv")

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nSmart_Plug_Energy_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/splug_mean_rel_pivot.csv")

pivot_fold_median()

pivot_fold_mean()

#sw stuff

shapiro_results = sw_Test(df_cleaned, 'duration_seconds')
print("\nsw of duration seconds\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/dur_sw.csv")

shapiro_results = sw_Test(df_cleaned, 'RAPL_diff')
print("\nsw of RAPL_diff\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/rapl_diff_sw.csv")

shapiro_results = sw_Test(df_cleaned, 'diff_aenergy_total')
print("\nsw of diff_aenergy_total\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/diff_aenergy_total_sw.csv")

# kw stuff

def kw_fold():

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('benchmark'):
        # Get all tool groups for this benchmark
        samples = [subgroup['RAPL_diff'].values for _, subgroup in group.groupby('tool')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'benchmark': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)

    print("\nKW for benchmarks of RAPL_diff\n")
    print(kruskal_df)

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('tool'):
        # Get all tool groups for this benchmark
        samples = [subgroup['RAPL_diff'].values for _, subgroup in group.groupby('benchmark')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'tool': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)
    print("\nKW for tools of RAPL_diff\n")
    print(kruskal_df)

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('benchmark'):
        # Get all tool groups for this benchmark
        samples = [subgroup['duration_seconds'].values for _, subgroup in group.groupby('tool')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'benchmark': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)

    print("\nKW for benchmarks of duration\n")
    print(kruskal_df)

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('tool'):
        # Get all tool groups for this benchmark
        samples = [subgroup['duration_seconds'].values for _, subgroup in group.groupby('benchmark')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'tool': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)
    print("\nKW for tools of duration\n")
    print(kruskal_df)

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('benchmark'):
        # Get all tool groups for this benchmark
        samples = [subgroup['diff_aenergy_total'].values for _, subgroup in group.groupby('tool')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'benchmark': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)

    print("\nKW for benchmarks of Smart Plug Energy\n")
    print(kruskal_df)

    results = []

    # Loop over each benchmark
    for benchmark, group in df_cleaned.groupby('tool'):
        # Get all tool groups for this benchmark
        samples = [subgroup['diff_aenergy_total'].values for _, subgroup in group.groupby('benchmark')]
        
        # You need at least 2 groups with data
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
        else:
            stat, p_value = None, None

        results.append({
            'tool': benchmark,
            'H_statistic': stat,
            'p_value': p_value
        })

    kruskal_df = pd.DataFrame(results)
    print("\nKW for tools of Smart Plug Energy\n")
    print(kruskal_df)

kw_fold()

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'mi')
].copy()

# maxtix_of_boxs(df_filtered, 'RAPL_diff', 'RAPL Energy Consumption (J)', "./plots/RAPL_Energy_Consumption_box_matrix.png")

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'sl') & 
    (df_cleaned['benchmark'] != 'mi')
].copy()

# maxtix_of_boxs(df_filtered, 'duration_seconds', 'Time To Complete (s)', "./plots/duration_seconds_box_matrix.png")

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'mi')
].copy()

# maxtix_of_boxs(df_filtered, 'diff_aenergy_total', 'Smart Plug Energy Consumption (Wh)', "./plots/Splug_Energy_Consumption_box_matrix.png")
