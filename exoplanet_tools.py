# exoplanet_tools_ready.py
import os
import logging
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ExoplanetAnalyzer:
    """
    Analyzer لتحقق وفلترة نتائج FITS وربطها بملف Kepler Data.
    Usage:
        analyzer = ExoplanetAnalyzer("lightcurves/")
        analyzer.ingest_results(results_list)  # results_list جاهزة
        metrics = analyzer.validate_against_csv("Kepler Data.csv", period_tolerance=0.01)
    """

    def __init__(self, lightcurves_dir: str = "lightcurves/"):
        self.lightcurves_dir = lightcurves_dir
        self.results: List[Dict] = []
        os.makedirs(self.lightcurves_dir, exist_ok=True)
        os.makedirs("results", exist_ok=True)

    @staticmethod
    def _extract_kepid_from_filename(filename: str) -> Optional[int]:
        import re
        m = re.search(r"(\d{5,9})", filename)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def ingest_results(self, results_list: List[Dict]):
        df = pd.DataFrame(results_list).copy()
        # توحيد الأعمدة
        if "file" in df.columns and "filename" not in df.columns:
            df.rename(columns={"file": "filename"}, inplace=True)
        if "best_period_days" in df.columns and "period" not in df.columns:
            df.rename(columns={"best_period_days": "period"}, inplace=True)
        if "max_power" in df.columns and "power" not in df.columns:
            df.rename(columns={"max_power": "power"}, inplace=True)
        if "kepid" not in df.columns:
            df["kepid"] = df["filename"].apply(lambda x: self._extract_kepid_from_filename(x) if isinstance(x, str) else None)
        self.results = df.to_dict(orient="records")
        logger.info(f"✅ تم تحميل {len(self.results)} نتائج جاهزة للتحقق.")

    def validate_against_csv(
        self,
        kepler_csv: str,
        csv_sep: str = ";",
        period_tolerance: float = 0.01,
        power_threshold: Optional[float] = None,
        save_path: str = "results/validation_results.csv"
    ) -> Optional[Dict]:
        if not self.results:
            logger.warning("⚠️ لا توجد نتائج منشورة في analyzer. استعمل ingest_results(results_list) أولاً.")
            return None

        try:
            kepler_df = pd.read_csv(kepler_csv, sep=csv_sep)
        except Exception as e:
            logger.error(f"خطأ في قراءة ملف Kepler CSV: {e}")
            return None

        results_df = pd.DataFrame(self.results).copy()
        # توحيد الأسماء
        if "best_period_days" in results_df.columns and "period" not in results_df.columns:
            results_df.rename(columns={"best_period_days": "period"}, inplace=True)
        # Ensure numeric
        results_df["period"] = pd.to_numeric(results_df.get("period", pd.Series(dtype=float)), errors="coerce")
        results_df["power"] = pd.to_numeric(results_df.get("power", pd.Series(dtype=float)), errors="coerce")
        if "kepid" in kepler_df.columns:
            kepler_df["kepid"] = pd.to_numeric(kepler_df["kepid"], errors="coerce")

        # Merge by kepid if possible
        merged = None
        if "kepid" in results_df.columns and "kepid" in kepler_df.columns and results_df["kepid"].notna().any():
            merged = pd.merge(results_df, kepler_df, on="kepid", how="inner", suffixes=("_res", "_koi"))

        # إذا لم يوجد kepid صالح، نربط بالفترة الأقرب
        if merged is None or merged.empty:
            if "period" not in results_df.columns or "koi_period" not in kepler_df.columns:
                logger.warning("لا يوجد طريقة ربط: لا 'kepid' ولا 'period' متاحة للمطابقة.")
                return None
            matches = []
            koi_periods = pd.to_numeric(kepler_df["koi_period"], errors="coerce")
            for _, r in results_df.iterrows():
                p = r.get("period", np.nan)
                if np.isnan(p):
                    matches.append(None)
                    continue
                diffs = abs(koi_periods - p)
                idx = diffs.idxmin()
                matched = kepler_df.loc[idx].copy()
                matched = matched.add_prefix("koi_")
                for col in results_df.columns:
                    matched[col] = r.get(col)
                matched["period_diff"] = abs(matched["koi_koi_period"] - p)
                matches.append(matched)
            merged = pd.DataFrame(matches).reset_index(drop=True)

        if merged.empty:
            logger.warning("⚠️ لم يتم إنشاء دمج بين النتائج وبيانات Kepler.")
            return None

        # true_label
        if "koi_disposition" in merged.columns:
            merged["true_label"] = merged["koi_disposition"].apply(lambda x: 1 if str(x).upper() == "CONFIRMED" else 0)

