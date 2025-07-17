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
                    row[data[1][0].strip()] = int(data[1][1].strip())
                    row[data[2][0].strip()] = int(data[2][1].strip())
                    row[data[3][0].strip()] = int(data[3][1].strip())
                    row[data[4][0].strip()] = int(data[4][1].strip())
                    row[data[5][0].strip()] = int(data[5][1].strip())
                    row[data[6][0].strip()] = int(data[6][1].strip())
                    row[data[7][0].strip()] = int(data[7][1].strip())
                    row[data[8][0].strip()] = int(data[8][1].strip())
                    row[data[9][0].strip()] = int(data[9][1].strip())
                    row[data[10][0].strip()] = int(data[10][1].strip())
                    row[data[11][0].strip()] = int(data[11][1].strip())
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
                    row[f"C0_sec"] = 0.0
                    row[f"C1_sec"] = 0.0
                    row[f"C3_sec"] = 0.0
                    row[f"C6_sec"] = 0.0
                    row[f"C8_sec"] = 0.0
                    row[f"C9_sec"] = 0.0
                    
                    with open(filepath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue  # skip empty lines

                            elif line.startswith("Core"):
                               continue
                                

                            elif line.startswith("-"):
                                continue

                            else:
                                sline = line.split("\t")
                                row[f"C1_sec"] = (float(sline[2])/100) + row[f"C1_sec"]
                                row[f"C3_sec"] = (float(sline[3])/100) + row[f"C3_sec"]
                                row[f"C6_sec"] = (float(sline[4])/100) + row[f"C6_sec"]
                                row[f"C8_sec"] = (float(sline[5])/100) + row[f"C8_sec"]
                                row[f"C9_sec"] = (float(sline[6])/100) + row[f"C9_sec"]
                                row[f"C0_sec"] = row[f"C0_sec"] + (1 - (float(sline[2])/100) - (float(sline[3])/100)- 
                                                                   (float(sline[4])/100) - (float(sline[5])/100) - (float(sline[6])/100)) 
                        
                    
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
    
    plt.savefig(outfile, dpi=150)


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

runtable = df1[["run_number", "tool", "benchmark"]].sort_values(by="run_number")

runtable.to_csv("./csv/runtable_bench.csv", index=False)
with open('./tex/runtable_bench.tex', 'w') as f:
    f.write(runtable.to_latex(index=True, escape=False))

runtable = df2[["run_number", "tool", "benchmark"]].sort_values(by="run_number")

runtable.to_csv("./csv/runtable_sleep_micro.csv", index=False)
with open('./tex/runtable_sleep_micro.tex', 'w') as f:
    f.write(runtable.to_latex(index=True, escape=False))

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

    print("\nMedien_RAPL_diff\n")

    pivot = pivot.loc[tools]

    print(pivot)  
    pivot.to_csv("./csv/RAPL_diff_median_pivot.csv")
    with open('./tex/RAPL_diff_median_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))

    # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMedien_RAPL_diff_rel\n")

    pivot_adjusted.to_csv("./csv/RAPL_rel_median_pivot.csv")
    with open('./tex/RAPL_rel_median_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))
    print(pivot_adjusted)

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='duration_seconds',
        aggfunc='median'  # Average
    )

    print("\nMedien_Durations\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/dur_median_pivot.csv")
    with open('./tex/dur_median_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))
    

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMedien_Durations_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/dur_median_rel_pivot.csv")
    with open('./tex/dur_median_rel_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))

    # smart plug energy total 
    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='diff_aenergy_total',
        aggfunc='median'  # Average
    )

    print("\nMedien_Smart_Plug_Energy\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/splug_median_pivot.csv")
    with open('./tex/splug_median_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMedien_Smart_Plug_Energy_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/splug_median_rel_pivot.csv")
    with open('./tex/splug_median_rel_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))

def pivot_fold_mean():

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='RAPL_diff',
        aggfunc='mean'  # Average
    )

    print("\nMean_RAPL_diff\n")

    pivot = pivot.loc[tools]

    print(pivot)  
    pivot.to_csv("./csv/RAPL_diff_mean_pivot.csv")
    with open('./tex/RAPL_diff_mean_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))

    # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMean_RAPL_diff_rel\n")

    pivot_adjusted.to_csv("./csv/RAPL_rel_mean_pivot.csv")
    with open('./tex/RAPL_rel_mean_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))
    print(pivot_adjusted)

    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='duration_seconds',
        aggfunc='mean'  # Average
    )

    print("\nMean_Durations\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/dur_mean_pivot.csv")
    with open('./tex/dur_mean_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMean_Durations_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/dur_mean_rel_pivot.csv")
    with open('./tex/dur_mean_rel_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))


    # smart plug energy total 
    pivot = df_cleaned.pivot_table(
        index='tool',
        columns='benchmark',
        values='diff_aenergy_total',
        aggfunc='mean'  # Average
    )

    print("\nMean_Smart_Plug_Energy\n")

    pivot = pivot.loc[tools]

    print(pivot)
    pivot.to_csv("./csv/splug_mean_pivot.csv")
    with open('./tex/splug_mean_pivot.tex', 'w') as f:
        f.write(pivot.to_latex(index=True, escape=False))

     # Subtract the 'No_Tools' row from every other row in each column
    pivot_adjusted = pivot.subtract(pivot.loc["No_Tools"], axis=1)

    print("\nMean_Smart_Plug_Energy_rel\n")

    print(pivot_adjusted)
    pivot_adjusted.to_csv("./csv/splug_mean_rel_pivot.csv")
    with open('./tex/splug_mean_rel_pivot.tex', 'w') as f:
        f.write(pivot_adjusted.to_latex(index=True, escape=False))

pivot_fold_median()

pivot_fold_mean()

#sw stuff

shapiro_results = sw_Test(df_cleaned, 'duration_seconds')
print("\nsw of duration seconds\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/dur_sw.csv")
with open('./tex/dur_sw.tex', 'w') as f:
    f.write(shapiro_results.to_latex(index=True, escape=False))

shapiro_results = sw_Test(df_cleaned, 'RAPL_diff')
print("\nsw of RAPL_diff\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/rapl_diff_sw.csv")
with open('./tex/rapl_diff_sw.tex', 'w') as f:
    f.write(shapiro_results.to_latex(index=True, escape=False))

shapiro_results = sw_Test(df_cleaned, 'diff_aenergy_total')
print("\nsw of diff_aenergy_total\n")
# pprint(shapiro_results)
shapiro_results.to_csv("./csv/diff_aenergy_total_sw.csv")
with open('./tex/diff_aenergy_total_sw.tex', 'w') as f:
    f.write(shapiro_results.to_latex(index=True, escape=False))


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
    with open('./tex/kw_bench_rapl_diff.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


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
    with open('./tex/kw_tools_rapl_diff.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


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
    with open('./tex/kw_bench_duration.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


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
    with open('./tex/kw_tool_duration.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


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
    with open('./tex/kw_bench_splug_diff.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


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
    with open('./tex/kw_tool_splug_diff.tex', 'w') as f:
        f.write(kruskal_df.to_latex(index=True, escape=False))


kw_fold()

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'mi')
].copy()

maxtix_of_boxs(df_filtered, 'RAPL_diff', 'RAPL Energy Consumption (J)', "./plots/RAPL_Energy_Consumption_box_matrix.png")

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'sl') & 
    (df_cleaned['benchmark'] != 'mi')
].copy()

maxtix_of_boxs(df_filtered, 'duration_seconds', 'Time To Complete (s)', "./plots/duration_seconds_box_matrix.png")

df_filtered = df_cleaned[
    (df_cleaned['tool'] != 'No_Tools') &
    (df_cleaned['benchmark'] != 'mi')
].copy()

# maxtix_of_boxs(df_filtered, 'diff_aenergy_total', 'Smart Plug Energy Consumption (Wh)', "./plots/Splug_Energy_Consumption_box_matrix.png")

# ref kern box plots with dist points

df_filtered = df_cleaned[
    (df_cleaned['tool'] == 'Ref_Kern') & 
    (df_cleaned['benchmark'] != 'mi') 
    ]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Energy Consumption of Benchmarks for Ref_Rapl_Kern', fontsize=16)
axes = axes.flatten()

# Pick 7 distinct colors
palette = sns.color_palette("tab10")[:7]

for i, (benchmark, color) in enumerate(zip(benchmarks1, palette)):
    ax = axes[i]
    data = df_filtered[df_filtered['benchmark'] == benchmark]
    
    sns.boxplot(
        x=[''] * len(data),
        y='RAPL_diff',
        data=data,
        ax=ax,
        color=color,
        fliersize=0,
        boxprops=dict(facecolor='none', edgecolor=color),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        medianprops=dict(color=color)
    )
    
    sns.stripplot(
        x=[''] * len(data),
        y='RAPL_diff',
        data=data,
        ax=ax,
        color=color,
        jitter=True,
        alpha=0.7
    )
    
    ax.set_title(benchmark , fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])  # remove x ticks

# Hide the 8th subplot
axes[-1].axis('off')

# Add one common y-axis label
fig.text(0.04, 0.5, 'Energy (Joules)', va='center', rotation='vertical', fontsize=20)

# fig.text(0.5, 0.04, 'Ref_Rapl_Kernel', va='center', rotation='Horizontal', fontsize=12)

plt.tight_layout(rect=[0.05, 0, 1, 1])

plt.savefig("./plots/ref_kern_benchmarks_boxplots.png", dpi=150)
plt.show()

# ref user box plots with dist points

df_filtered = df_cleaned[
    (df_cleaned['tool'] == 'Ref_User') & 
    (df_cleaned['benchmark'] != 'mi') 
    ]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Energy Consumption of Benchmarks for Ref_Rapl_User', fontsize=16)
axes = axes.flatten()

# Pick 7 distinct colors
palette = sns.color_palette("tab10")[:7]

for i, (benchmark, color) in enumerate(zip(benchmarks1, palette)):
    ax = axes[i]
    data = df_filtered[df_filtered['benchmark'] == benchmark]
    
    sns.boxplot(
        x=[''] * len(data),
        y='RAPL_diff',
        data=data,
        ax=ax,
        color=color,
        fliersize=0,
        boxprops=dict(facecolor='none', edgecolor=color),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
        medianprops=dict(color=color)
    )
    
    sns.stripplot(
        x=[''] * len(data),
        y='RAPL_diff',
        data=data,
        ax=ax,
        color=color,
        jitter=True,
        alpha=0.7
    )
    
    ax.set_title(benchmark, fontsize=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])  # remove x ticks

# Hide the 8th subplot
axes[-1].axis('off')

# Add one common y-axis label
fig.text(0.04, 0.5, 'Energy (Joules)', va='center', rotation='vertical', fontsize=20)

# fig.text(0.5, 0.04, 'Ref_Rapl_Kernel', va='center', rotation='Horizontal', fontsize=12)

plt.tight_layout(rect=[0.05, 0, 1, 1])

plt.savefig("./plots/ref_user_benchmarks_boxplots.png", dpi=150)
plt.show()

print("\n Cstate Stuff \n")

grouped = csdf[["tool","run_number","C0_sec","C1_sec","C3_sec","C6_sec","C8_sec","C9_sec"]].groupby("tool")

for tool_name, group_df in grouped:
    print(f"\n=== Tool: {tool_name} ===")
    print(group_df)


grouped_mean = csdf[["tool","C0_sec","C1_sec","C3_sec","C6_sec","C8_sec","C9_sec"]].groupby("tool").median(numeric_only=True)

print("\nC-State mean by tool\n")
print(grouped_mean)

# csdf.to_csv("./csv/csdf.csv")

print("\n Micro Bench sub raw\n")

# print("Columns:", mbdf.columns.tolist())

# print(mbdf[['nop', 'mov eax ebx']])

mbdf_subtracted = mbdf.copy()

# Columns to adjust
cols_to_subtract = [
    'mov eax ebx',
    'cpuid 0x1 0',
    'rdmsr 0x611',
    'rdmsr 0x639',
    'rdmsr 0x641',
    'rdmsr 0x619',
    'rdmsr 0x19C',
    'rdmsr 0x17'
]

# Subtract 'nop' from each specified column
mbdf_subtracted[cols_to_subtract] = mbdf_subtracted[cols_to_subtract].subtract(mbdf_subtracted['nop'], axis=0)

print(mbdf_subtracted[['mov eax ebx', 'cpuid 0x1 0','rdmsr 0x611', 'rdmsr 0x639', 'rdmsr 0x641', 'rdmsr 0x619', 'rdmsr 0x19C', 'rdmsr 0x17', 
                       'sys_call_overhead_proc_read','sys_call_overhead_sys_read']])

with open('./tex/mbdata_sub_raw.tex', 'w') as f:
    f.write(mbdf_subtracted[['mov eax ebx', 'cpuid 0x1 0','rdmsr 0x611', 'rdmsr 0x639', 'rdmsr 0x641', 'rdmsr 0x619', 'rdmsr 0x19C', 'rdmsr 0x17', 
                       'sys_call_overhead_proc_read','sys_call_overhead_sys_read']].to_latex(index=True, escape=False))


# columns_to_test = cols_to_subtract

# # Store results
# normality_results = {}

# for col in columns_to_test:
#     # Drop NaN just in case — Shapiro does not handle NaN
#     data = mbdf_subtracted[col].dropna()
#     stat, p_value = shapiro(data)
#     normality_results[col] = {'W-statistic': stat, 'p-value': p_value}

# # Display nicely
# for col, result in normality_results.items():
#     print(f"{col}: W = {result['W-statistic']:.4f}, p = {result['p-value']}")

print("\n Micro Bench per call sub mean median, max, min\n")

cols_to_sum = [
    'mov eax ebx',
    'cpuid 0x1 0',
    'rdmsr 0x611',
    'rdmsr 0x639',
    'rdmsr 0x641',
    'rdmsr 0x619',
    'rdmsr 0x19C',
    'rdmsr 0x17',
    'sys_call_overhead_proc_read',
    'sys_call_overhead_sys_read'
]

summary_stats = mbdf_subtracted[cols_to_sum].agg(['mean', 'median', 'min', 'max'])

summary_stats_div = summary_stats.div(1000)

pd.set_option('display.float_format', '{:.3e}'.format)

print(summary_stats_div)

with open('./tex/mbdata_sum_per_exec_ns.tex', 'w') as f:
    f.write(summary_stats_div.to_latex(index=True, escape=False))

cpu_freq_ghz = 3.1  # GHz of Intel NUC Kit NUC8i7HVK on powersave governer

# Multiply all values by CPU frequency in GHz
summary_stats_cycles = summary_stats_div * cpu_freq_ghz

print("\n Micro Bench per cycle sub mean median, max, min\n")

print(summary_stats_cycles)

with open('./tex/mbdata_sum_per_exec_cycles.tex', 'w') as f:
    f.write(summary_stats_cycles.to_latex(index=True, escape=False))