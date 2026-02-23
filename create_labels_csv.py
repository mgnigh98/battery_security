#!/usr/bin/env python3
import os, glob, re
import pandas as pd

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_label_sheet(xl: pd.ExcelFile):
    # common names you used
    for cand in ["cycle_labels", "labels", "cycle_label", "cyclelabels"]:
        if cand in xl.sheet_names:
            return cand
    # fallback: pick the sheet that contains both "Cycle" and "cycle_label_3name" like columns
    for s in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=s, nrows=5)
        cols = {norm(c): c for c in df.columns}
        if "cycle" in cols and ("cyclelabel3name" in cols or "cyclelabel3class" in cols or "label" in cols):
            return s
    return None

def main():
    labels_dir = "drone_labels_out"
    out_csv = os.path.join(labels_dir, "all_cycle_labels_3class.csv")

    files = sorted(glob.glob(os.path.join(labels_dir, "**", "*_good_labeled_3class.xlsx"), recursive=True))
    if not files:
        raise SystemExit("No *_good_labeled_3class.xlsx files found under drone_labels_out/")

    rows = []
    for fp in files:
        base = os.path.basename(fp)
        xl = pd.ExcelFile(fp)
        sheet = find_label_sheet(xl)
        if sheet is None:
            print(f"[SKIP] {base}: cannot find label sheet")
            continue

        df = pd.read_excel(xl, sheet_name=sheet)

        # try to locate columns robustly
        cols = {norm(c): c for c in df.columns}
        cycle_col = cols.get("cycle") or cols.get("cycleindex") or cols.get("cycle_index")
        name_col  = cols.get("cyclelabel3name") or cols.get("cycle_label_3name")
        if name_col is None:
            # sometimes you only stored numeric class; map it
            cls_col = cols.get("cyclelabel3class") or cols.get("cycle_label_3class") or cols.get("label")
            if cls_col is not None:
                # adjust mapping if your encoding differs
                map3 = {0: "bad", 1: "good_no_drone", 2: "good_drone"}
                df["_name"] = pd.to_numeric(df[cls_col], errors="coerce").map(map3)
                name_col = "_name"

        if cycle_col is None or name_col is None:
            print(f"[SKIP] {base}: missing cycle/name columns")
            continue

        tmp = df[[cycle_col, name_col]].copy()
        tmp.columns = ["Cycle", "cycle_label_3name"]
        tmp["Cycle"] = pd.to_numeric(tmp["Cycle"], errors="coerce")
        tmp = tmp.dropna(subset=["Cycle"])
        tmp["Cycle"] = tmp["Cycle"].astype(int)

        # IMPORTANT: 'file' must match the ORIGINAL data xlsx filename
        # If your labeled files keep the same base name, we can recover it.
        # Example: original "ABC.xlsx" -> labeled "ABC_good_labeled_3class.xlsx"
        orig_name = base.replace("_good_labeled_3class.xlsx", ".xlsx")
        tmp["file"] = orig_name

        rows.append(tmp[["file", "Cycle", "cycle_label_3name"]])

    all_df = pd.concat(rows, ignore_index=True).drop_duplicates()
    os.makedirs(labels_dir, exist_ok=True)
    all_df.to_csv(out_csv, index=False)
    print(f"[DONE] wrote {out_csv} rows={len(all_df)}")

if __name__ == "__main__":
    main()
