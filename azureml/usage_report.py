"""Azure usage report for the network-centrality project.

Cost Management bills Azure ML compute at workspace level, split only by VM series — and this
workspace is shared with other projects that run the same VM series. So this report rebuilds
the project's own usage bottom-up:

  1. every job in the project's experiments (start/end from run history)  -> node-hours
  2. the project's compute instance uptime (billed while running, jobs or not)
  3. Azure Retail Prices API (eastus2, Linux, dedicated vs Low Priority)    -> USD estimate

It also pulls Cost Management actuals for the workspace (all projects, lags 8-24 h) as a
ceiling/sanity check. Sponsorship credit balance is not exposed by any API — read it at
https://www.microsoftazuresponsorships.com.

Usage:  python azureml/usage_report.py [--ci-stopped-at "2026-10-12T22:00:00Z"]
Writes: reports/azure_usage_jobs.csv, reports/azure_usage_summary.json
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SUB = "37bac92f-7fda-49aa-a356-6384c88795a0"
RG = "tes-azure-benefit"
WS = "azureml-azure-benefit"
REGION = "eastus2"
EXPERIMENTS = {"loc-network-centrality", "loc-network-model"}
# Project computes -> (VM size, tier, idle minutes billed after the last job on a node)
COMPUTES = {
    "loc-analytic": ("Standard_E64ds_v4", "dedicated", None),     # compute instance: billed on uptime
    "loc-cpu-lp": ("Standard_E32ds_v4", "low_priority", 5),
    "loc-cpu64-lp": ("Standard_E64ds_v4", "low_priority", 5),
    "loc-ded-e16": ("Standard_E16ds_v4", "dedicated", 10),
    "gpu-a100-lp": ("Standard_NC24ads_A100_v4", "low_priority", 2),
}
CI_CREATED = "2026-10-11T06:16:45Z"  # loc-analytic creation (from the create command output)
AZ = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" if sys.platform == "win32" else "az"
OUT = Path(__file__).resolve().parent.parent / "reports"


def az(*args):
    res = subprocess.run([AZ, *args, "--subscription", SUB, "-o", "json"], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-500:])
    return json.loads(res.stdout or "null")


def prices():
    out = {}
    for sku in {v[0] for v in COMPUTES.values()}:
        r = requests.get("https://prices.azure.com/api/retail/prices", params={
            "$filter": f"armRegionName eq '{REGION}' and armSkuName eq '{sku}' and serviceName eq 'Virtual Machines' "
                       f"and priceType eq 'Consumption'"}, timeout=60)
        for it in r.json()["Items"]:
            if "Windows" in it["productName"] or "Spot" in it["meterName"]:
                continue
            tier = "low_priority" if "Low Priority" in it["meterName"] else "dedicated"
            out[(sku, tier)] = it["retailPrice"]
    return out


def ts(s):
    return pd.Timestamp(s, tz="UTC") if s else pd.NaT


def jobs():
    rows = []
    listed = az("ml", "job", "list", "-g", RG, "-w", WS, "--all-results")
    for j in listed:
        if j.get("experiment_name") not in EXPERIMENTS:
            continue
        name = j["name"]
        d = az("ml", "job", "show", "-n", name, "-g", RG, "-w", WS)
        props = d.get("properties", {})
        compute = (d.get("compute") or "").replace("azureml:", "")
        rows.append({
            "job": name, "display_name": d.get("display_name"), "experiment": d.get("experiment_name"),
            "type": d.get("type"), "status": d.get("status"), "compute": compute,
            "created": ts(d["creation_context"]["created_at"]), "start": ts(props.get("StartTimeUtc")),
            "end": ts(props.get("EndTimeUtc")),
        })
        # sweep trials run as child jobs on the cluster
        if d.get("type") == "sweep":
            for c in az("ml", "job", "list", "-g", RG, "-w", WS, "--parent-job-name", name):
                cp = az("ml", "job", "show", "-n", c["name"], "-g", RG, "-w", WS).get("properties", {})
                rows.append({"job": c["name"], "display_name": f"{d.get('display_name')}/trial", "experiment": d.get("experiment_name"),
                             "type": "sweep_trial", "status": c.get("status"), "compute": compute,
                             "created": ts(c["creation_context"]["created_at"]), "start": ts(cp.get("StartTimeUtc")),
                             "end": ts(cp.get("EndTimeUtc"))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci-stopped-at", default=None, help="UTC time loc-analytic was stopped (default: now)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    now = pd.Timestamp(datetime.now(timezone.utc))
    price = prices()

    df = jobs()
    df = df[df.type != "sweep"]  # parent sweep holds no node; its trials do
    df["end_eff"] = df.end.fillna(now)  # still running -> count to now
    df["run_hours"] = ((df.end_eff - df.start).dt.total_seconds() / 3600).clip(lower=0).fillna(0)
    df["queue_minutes"] = ((df.start - df.created).dt.total_seconds() / 60).round(1)
    df["vm_size"] = df.compute.map(lambda c: COMPUTES.get(c, ("?", "?", 0))[0])
    df["tier"] = df.compute.map(lambda c: COMPUTES.get(c, ("?", "?", 0))[1])
    idle = df.compute.map(lambda c: (COMPUTES.get(c, (0, 0, 0))[2] or 0) / 60)
    df["usd_per_hour"] = [price.get((s, t), float("nan")) for s, t in zip(df.vm_size, df.tier)]
    # Cluster jobs: node runtime + idle tail before scale-down. Jobs on the compute instance cost
    # nothing extra — the instance is billed on uptime below.
    on_ci = df.compute == "loc-analytic"
    df["billed_hours"] = (df.run_hours + idle).where(~on_ci, 0.0)
    df["usd_est"] = (df.billed_hours * df.usd_per_hour).round(2)

    ci_end = pd.Timestamp(a.ci_stopped_at) if a.ci_stopped_at else now
    ci_hours = (ci_end - pd.Timestamp(CI_CREATED)).total_seconds() / 3600
    ci_price = price[("Standard_E64ds_v4", "dedicated")]

    by_compute = df.groupby(["compute", "vm_size", "tier"]).agg(
        jobs=("job", "size"), run_hours=("run_hours", "sum"), billed_hours=("billed_hours", "sum"),
        usd_est=("usd_est", "sum")).round(2).reset_index()
    ci_row = {"compute": "loc-analytic (uptime)", "vm_size": "Standard_E64ds_v4", "tier": "dedicated",
              "jobs": int(on_ci.sum()), "run_hours": round(df[on_ci].run_hours.sum(), 2),
              "billed_hours": round(ci_hours, 2), "usd_est": round(ci_hours * ci_price, 2)}
    by_compute = pd.concat([by_compute[by_compute.compute != "loc-analytic"], pd.DataFrame([ci_row])], ignore_index=True)

    vcpus = {"Standard_E64ds_v4": 64, "Standard_E32ds_v4": 32, "Standard_E16ds_v4": 16, "Standard_NC24ads_A100_v4": 24}
    df["vcpu_hours"] = df.run_hours * df.vm_size.map(vcpus).fillna(0)
    summary = {
        "generated_utc": now.isoformat(),
        "jobs_total": int(len(df)), "jobs_completed": int((df.status == "Completed").sum()),
        "jobs_failed": int((df.status == "Failed").sum()), "jobs_running": int(df.end.isna().sum()),
        "first_job_utc": str(df.created.min()), "last_job_end_utc": str(df.end.max()),
        "node_run_hours": round(df.run_hours.sum(), 2), "vcpu_hours": round(df.vcpu_hours.sum(), 1),
        "gpu_hours": round(df[df.vm_size.str.contains("A100")].run_hours.sum(), 2),
        "compute_instance_uptime_hours": round(ci_hours, 2),
        "usd_estimate_total": round(by_compute.usd_est.sum(), 2),
        "usd_estimate_by_compute": by_compute.to_dict(orient="records"),
        "prices_usd_per_hour": {f"{k[0]}|{k[1]}": v for k, v in price.items()},
        "notes": [
            "Estimate = retail list price (eastus2, Linux) x node-hours; Sponsorship credit is drawn at these list prices.",
            "Cluster idle tail before scale-down is included; node provisioning time before StartTimeUtc is not (minutes).",
            "Storage (blob ~ few GB) and container registry are negligible and not included.",
            "Cost Management actuals cover the whole shared workspace and lag 8-24 h; see cost_management_workspace_usd.",
        ],
    }

    try:  # Cost Management actuals (whole resource group, all projects)
        body = {"type": "ActualCost", "timeframe": "Custom",
                "timePeriod": {"from": "2026-10-11T00:00:00Z", "to": now.strftime("%Y-%m-%dT23:59:59Z")},
                "dataset": {"granularity": "Daily", "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
                            "grouping": [{"type": "Dimension", "name": "MeterSubCategory"}],
                            "filter": {"dimensions": {"name": "ResourceGroupName", "operator": "In", "values": [RG]}}}}
        bf = OUT / "_cost_query.json"
        bf.write_text(json.dumps(body))
        r = az("rest", "--method", "post", "--url",
               f"https://management.azure.com/subscriptions/{SUB}/providers/Microsoft.CostManagement/query?api-version=2023-11-01",
               "--body", f"@{bf}")
        bf.unlink()
        cols = [c["name"] for c in r["properties"]["columns"]]
        summary["cost_management_workspace_usd"] = [dict(zip(cols, row)) for row in r["properties"]["rows"]]
    except Exception as e:  # throttling (429) is common; the estimate stands on its own
        summary["cost_management_workspace_usd"] = f"unavailable: {e}"

    cols = ["job", "display_name", "experiment", "type", "status", "compute", "vm_size", "tier", "created", "start", "end",
            "queue_minutes", "run_hours", "billed_hours", "usd_per_hour", "usd_est"]
    df.sort_values("created")[cols].to_csv(OUT / "azure_usage_jobs.csv", index=False)
    (OUT / "azure_usage_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(by_compute.to_string(index=False))
    print(f"\njobs={summary['jobs_total']} node-hours={summary['node_run_hours']} vCPU-hours={summary['vcpu_hours']} "
          f"GPU-hours={summary['gpu_hours']} -> est. USD {summary['usd_estimate_total']}")


if __name__ == "__main__":
    main()
