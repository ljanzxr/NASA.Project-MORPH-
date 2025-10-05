import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from astropy.io import fits
import os
from tqdm import tqdm
from multiprocessing import Pool


os.makedirs("results", exist_ok=True)

# =========================
# 1) قراءة ملفات FITS و Kepler CSV
# =========================
fits_file = "fits_results.csv"
kepler_file = "Kepler Data.csv"

# قراءة نتائج FITS
df_results = pd.read_csv(fits_file)
print(f"✅ تم تحميل نتائج FITS: {len(df_results)}")
print("📂 أعمدة FITS قبل التوحيد:", df_results.columns.tolist())

# توحيد الأعمدة
if "file" in df_results.columns:
    df_results.rename(columns={"file": "filename"}, inplace=True)
if "best_period_days" in df_results.columns:
    df_results.rename(columns={"best_period_days": "period"}, inplace=True)
if "kepid" not in df_results.columns:
    df_results["kepid"] = np.nan  # لو ما موجود أصلاً

# numeric conversion
df_results["kepid"] = pd.to_numeric(df_results["kepid"], errors="coerce")
df_results["period"] = pd.to_numeric(df_results["period"], errors="coerce")

print("📂 أعمدة FITS بعد التوحيد:", df_results.columns.tolist())

# قراءة Kepler CSV مع محاولة اكتشاف الفاصل
try:
    kepler_df = pd.read_csv(kepler_file, sep=None, engine="python")
except Exception:
    # إذا فشل autodetect separator، نجرب semicolon
    kepler_df = pd.read_csv(kepler_file, sep=";", engine="python")

# تنظيف الأعمدة المكررة أو الفارغة
kepler_df = kepler_df.loc[:, ~kepler_df.columns.str.contains('^Unnamed')]
# توحيد الأعمدة
if "kepid" not in kepler_df.columns:
    if "koi_period" in kepler_df.columns:
        kepler_df.rename(columns={"koi_period": "period"}, inplace=True)

kepler_df["kepid"] = pd.to_numeric(kepler_df.get("kepid", np.nan), errors="coerce")
kepler_df["period"] = pd.to_numeric(kepler_df.get("period", np.nan), errors="coerce")

print("📂 الأعمدة في Kepler بعد التنظيف:", kepler_df.columns.tolist())

# =========================
# 2) محاولة الربط
# =========================
merged = pd.DataFrame()
# حاول الربط أولاً بـ kepid
if "kepid" in df_results.columns and "kepid" in kepler_df.columns:
    merged = pd.merge(df_results, kepler_df, on="kepid", how="inner", suffixes=("_res", "_koi"))

# لو فارغ، حاول الربط بـ period الأقرب
if merged.empty:
    print("⚠️ الربط بـ kepid فشل، محاولة الربط بـ period الأقرب")
    matches = []
    for _, r in df_results.iterrows():
        if pd.isna(r["period"]):
            continue
        diffs = abs(kepler_df["period"] - r["period"])
        idx = diffs.idxmin()
        matched = kepler_df.loc[idx].copy()
        for col in df_results.columns:
            matched[col] = r.get(col)
        matched["period_diff"] = abs(matched["period"] - r["period"])
        matches.append(matched)
    merged = pd.DataFrame(matches)

print(f"✅ تم تحميل {len(merged)} نتائج صالحة للتحقق.")

# =========================
# 3) إنشاء true_label و predicted
# =========================
if "koi_disposition" in merged.columns:
    merged["true_label"] = merged["koi_disposition"].apply(lambda x: 1 if str(x).upper() == "CONFIRMED" else 0)
else:
    merged["true_label"] = 0

# Predicted by power
if "max_power" in merged.columns:
    threshold = merged["max_power"].median()
    merged["pred_by_power"] = merged["max_power"].apply(lambda x: 1 if pd.notna(x) and x > threshold else 0)
else:
    merged["pred_by_power"] = 0

# Predicted by period match
if "period_diff" in merged.columns:
    merged["pred_by_period"] = merged["period_diff"].apply(lambda x: 1 if pd.notna(x) and x <= 0.01 * merged["period"] else 0)
else:
    merged["pred_by_period"] = 0

# =========================
# 4) Metrics
# =========================
metrics = {}
metrics["by_power"] = {
    "accuracy": float(accuracy_score(merged["true_label"], merged["pred_by_power"])),
    "precision": float(precision_score(merged["true_label"], merged["pred_by_power"], zero_division=0)),
    "recall": float(recall_score(merged["true_label"], merged["pred_by_power"], zero_division=0))
}
metrics["by_period"] = {
    "accuracy": float(accuracy_score(merged["true_label"], merged["pred_by_period"])),
    "precision": float(precision_score(merged["true_label"], merged["pred_by_period"], zero_division=0)),
    "recall": float(recall_score(merged["true_label"], merged["pred_by_period"], zero_division=0))
}

print("📊 Metrics:", metrics)

import time


os.makedirs("results", exist_ok=True)

# =========================
# قراءة نتائج FITS فقط
# =========================
fits_file = "fits_results.csv"
df_results = pd.read_csv(fits_file)
print(f"✅ تم تحميل نتائج FITS: {len(df_results)}")
df_results.rename(columns={"file": "filename", "best_period_days": "period"}, inplace=True)
df_results["kepid"] = pd.to_numeric(df_results.get("kepid", np.nan), errors="coerce")
df_results["period"] = pd.to_numeric(df_results["period"], errors="coerce")
print("📂 أعمدة FITS بعد التوحيد:", df_results.columns.tolist())

# =========================
# Feature Extraction
# =========================
def extract_features(lightcurve_file):
    path = f"lightcurves/{lightcurve_file}"
    try:
        if lightcurve_file.endswith(".fits"):
            with fits.open(path) as hdul:
                data = hdul[1].data
                time_arr = data['TIME']
                flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['SAP_FLUX']
        else:
            lc = pd.read_csv(path)
            time_arr = lc['time'].values
            flux = lc['flux'].values

        mask = ~np.isnan(time_arr) & ~np.isnan(flux)
        time_arr, flux = time_arr[mask], flux[mask]

        if len(flux) == 0:
            return pd.Series({'depth': np.nan, 'duration': np.nan, 'snr': np.nan})

        median_flux = np.median(flux)
        min_flux = np.min(flux)
        depth = median_flux - min_flux

        below_median = time_arr[flux < median_flux]
        duration = (np.max(below_median) - np.min(below_median)) if below_median.size > 0 else 0
        snr = depth / np.std(flux)

        return pd.Series({'depth': depth, 'duration': duration, 'snr': snr})

    except Exception as e:
        print(f"⚠️ خطأ مع {lightcurve_file}: {e}")
        return pd.Series({'depth': np.nan, 'duration': np.nan, 'snr': np.nan})

# =========================
# تطبيق على كل الملفات وحساب الوقت
# =========================
start_time = time.time()
features = df_results['filename'].apply(extract_features)
df_results = pd.concat([df_results, features], axis=1)
end_time = time.time()

# =========================
# حفظ النتائج النهائية
# =========================
df_results.to_csv("results/validation_with_features.csv", index=False)
print("✅ تم حفظ Validation + Features في: results/validation_with_features.csv")
print(f"⏱️ الوقت المستغرق: {end_time - start_time:.2f} ثانية")