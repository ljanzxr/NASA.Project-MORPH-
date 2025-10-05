# kepler_full_pipeline.py
"""
واحد سكربت شامل:
1) يقرأ Kepler full CSV (يتعامل مع header-comment)
2) ينظف البيانات، يعوّض القيم المفقودة، يحفظ نسخة نظيفة
3) (اختياري) يستخرج features من ملفات lightcurves/FITS إذا متوفرة ويربطها
4) يبني ميزات engineering إضافية
5) يدرب RandomForest, XGBoost, NeuralNetwork (أو MLP fallback)
6) يعمل ensemble (weighted average of predict_proba)
7) يحفظ النماذج، الرسومات، والتوقعات
"""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer

import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Optional FITS support
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

# Optional tensorflow
USE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    USE_TF = True
except Exception:
    USE_TF = False

# ---------------- CONFIG ----------------
KEPLER_FILE = "Kepler_Full.csv"   # أو اسم الملف الي عندك (أو مسار كامل)
OUTPUT_DIR = "results"
MODELS_DIR = "models"
CLEANED_DIR = "data_cleaned"
LIGHTCURVES_FOLDER = "light_curves"  # مسار مجلد الملفات FITS/CSV لو عندك
FITS_RESULTS_CSV = "fits_results.csv" # لو عندك ملف يربط أسماء الملفات بنتائج البحث

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
QUICK = False    # لو تحبين تجربة سريعة: True
QUICK_FRAC = 0.15

# Hyperparams
RF_N_ESTIMATORS = 300
XGB_N_ESTIMATORS = 200
NN_EPOCHS = 80
NN_BATCH = 64
NN_PATIENCE = 8
NN_LR = 1e-3

# ---------------- Helpers ----------------
def load_kepler(path):
    """حاول قراءة الملف بعدة طرق (يتجاهل التعليقات #)"""
    print(f"🔭 Loading Kepler CSV from: {path}")
    try:
        df = pd.read_csv(path, comment='#', low_memory=False)
        print("✅ Loaded with comment='#'")
        return df
    except Exception as e:
        print("⚠ read with comment failed, trying default...")
    # try other separators
    for sep in ['\t', ';']:
        try:
            df = pd.read_csv(path, sep=sep, comment='#', low_memory=False)
            print(f"✅ Loaded with sep='{sep}'")
            return df
        except Exception:
            pass
    # last try plain
    df = pd.read_csv(path, low_memory=False)
    print("✅ Loaded with default read_csv")
    return df

def safe_median_fill(df, columns=None):
    """Fill numeric NaNs with column medians for specific columns or all numeric"""
    if columns is None:
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerics = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    if len(numerics) > 0:
        for col in numerics:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"   Filled {col} with median: {median_val:.4f}")
    return df

def robust_feature_engineering(df):
    """هندسة الميزات بطريقة أكثر مرونة"""
    df = df.copy()
    
    # الميزات الأساسية المطلوبة
    base_features = []
    
    # 1. الميزات الأساسية من كبلر
    if "koi_period" in df.columns:
        base_features.append("koi_period")
    if "koi_prad" in df.columns:
        base_features.append("koi_prad") 
    if "koi_depth" in df.columns:
        base_features.append("koi_depth")
    if "koi_duration" in df.columns:
        base_features.append("koi_duration")
    if "koi_model_snr" in df.columns:
        base_features.append("koi_model_snr")
        df["snr_log"] = np.log1p(df["koi_model_snr"])
        base_features.append("snr_log")
    
    # 2. الميزات المشتقة (مع معالجة الأخطاء)
    try:
        if "koi_depth" in df.columns and "koi_prad" in df.columns:
            df["depth_to_radius"] = df["koi_depth"] / (df["koi_prad"].replace(0, 1e-8) + 1e-8)
            base_features.append("depth_to_radius")
    except Exception as e:
        print(f"⚠ Could not create depth_to_radius: {e}")
    
    try:
        if "koi_duration" in df.columns and "koi_period" in df.columns:
            df["duration_to_period"] = df["koi_duration"] / (df["koi_period"].replace(0, 1e-8) + 1e-8)
            base_features.append("duration_to_period")
    except Exception as e:
        print(f"⚠ Could not create duration_to_period: {e}")
    
    # 3. الميزات الإضافية إذا كانت متوفرة
    additional_features = []
    if "koi_teq" in df.columns:
        df["teq_scaled"] = df["koi_teq"] / 1000.0
        additional_features.append("teq_scaled")
        additional_features.append("koi_teq")
    
    if "koi_steff" in df.columns:
        additional_features.append("koi_steff")
    if "koi_slogg" in df.columns:
        additional_features.append("koi_slogg") 
    if "koi_srad" in df.columns:
        additional_features.append("koi_srad")
    if "koi_insol" in df.columns:
        additional_features.append("koi_insol")
    if "koi_score" in df.columns:
        additional_features.append("koi_score")
    
    # 4. الميزات المعقدة (إذا كانت البيانات كافية)
    try:
        if all(col in df.columns for col in ["koi_steff", "koi_srad", "koi_prad"]):
            df["flux_index"] = (df["koi_steff"] / (df["koi_srad"].replace(0, 1e-8) + 1e-8)) * np.log1p(df["koi_prad"])
            additional_features.append("flux_index")
    except Exception as e:
        print(f"⚠ Could not create flux_index: {e}")
    
    all_features = base_features + additional_features
    print(f"✅ Engineered {len(all_features)} features total")
    
    return df, all_features

def extract_lightcurve_features_from_fits(filename):
    """مستخرج بسيط من ملفات FITS: depth, duration (in time units), snr"""
    path = os.path.join(LIGHTCURVES_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if filename.lower().endswith('.fits') and ASTROPY_AVAILABLE:
        with fits.open(path, memmap=True) as hdul:
            # غالباً data في HDU 1
            data = hdul[1].data
            # try PDCSAP_FLUX then SAP_FLUX then 'flux'
            colnames = [c.lower() for c in data.columns.names]
            if 'pdcsap_flux' in colnames:
                flux = data['PDCSAP_FLUX']
            elif 'sap_flux' in colnames:
                flux = data['SAP_FLUX']
            else:
                # try lowercase access
                try:
                    flux = data['flux']
                except Exception:
                    flux = None
            try:
                time = data['TIME']
            except Exception:
                time = None
            if flux is None:
                raise ValueError("no flux column in FITS")
            flux = np.array(flux, dtype=float)
            if time is None:
                # if no time, use index as proxy
                time = np.arange(len(flux), dtype=float)
            mask = ~np.isnan(flux)
            if mask.sum() == 0:
                return {'depth': np.nan, 'duration': np.nan, 'snr': np.nan}
            flux = flux[mask]
            time = np.array(time)[mask]
            median_flux = np.median(flux)
            min_flux = np.min(flux)
            depth = median_flux - min_flux
            below = time[flux < median_flux]
            duration = (np.max(below) - np.min(below)) if below.size>0 else 0.0
            snr = depth / (np.std(flux) + 1e-12)
            return {'depth': float(depth), 'duration': float(duration), 'snr': float(snr)}
    else:
        # try CSV read
        lc = pd.read_csv(path)
        if 'flux' in lc.columns and 'time' in lc.columns:
            flux = lc['flux'].astype(float).values
            time = lc['time'].astype(float).values
            mask = ~np.isnan(flux)
            if mask.sum() == 0:
                return {'depth': np.nan, 'duration': np.nan, 'snr': np.nan}
            flux = flux[mask]; time = time[mask]
            median_flux = np.median(flux); min_flux = np.min(flux)
            depth = median_flux - min_flux
            below = time[flux < median_flux]
            duration = (np.max(below) - np.min(below)) if below.size>0 else 0.0
            snr = depth / (np.std(flux) + 1e-12)
            return {'depth': float(depth), 'duration': float(duration), 'snr': float(snr)}
        else:
            raise ValueError("unsupported lightcurve format")

def build_keras_nn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=NN_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def check_data_sufficiency(X, y, min_samples=10):
    """تحقق إذا البيانات كافية للتدريب"""
    if len(X) < min_samples:
        print(f"❌ Not enough data: only {len(X)} samples available, need at least {min_samples}")
        return False
    
    # تحقق إذا هناك عينات من كل class
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")
    
    if len(unique_classes) < 2:
        print("❌ Need at least 2 classes for classification")
        return False
        
    if min(counts) < 2:
        print("❌ Some classes have less than 2 samples")
        return False
        
    return True

def create_fallback_features(df):
    """إنشاء ميزات بديلة من أي بيانات متوفرة"""
    print("🔄 Creating fallback features from available data...")
    
    # استخدم أي أعمدة رقمية متوفرة
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # إزالة الأعمدة غير المرغوبة
    exclude_cols = ['kepid', 'label', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    feature_candidates = [col for col in numeric_cols if col not in exclude_cols]
    
    # إعطاء أولوية للميزات الأساسية
    priority_features = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration', 'koi_score']
    final_features = []
    
    # أضف الميزات ذات الأولوية أولاً
    for feature in priority_features:
        if feature in feature_candidates:
            final_features.append(feature)
            feature_candidates.remove(feature)
    
    # أضف باقي الميزات (حد أقصى 15 ميزة)
    remaining_slots = 15 - len(final_features)
    final_features.extend(feature_candidates[:remaining_slots])
    
    print(f"✅ Selected {len(final_features)} fallback features: {final_features}")
    return final_features

def main():
    t0 = time.time()
    try:
        df = load_kepler(KEPLER_FILE)
        if df is None or len(df) == 0:
            print("❌ Empty dataset loaded")
            return
    except Exception as e:
        print("❌ Could not load Kepler file:", e)
        traceback.print_exc()
        return

    print("Columns available (first 40):", df.columns.tolist()[:40])
    print("Total rows:", len(df))

    # Select scientific columns we need (keep only existings)
    desired = [
        'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
        'koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
        'koi_model_snr', 'koi_teq', 'koi_steff', 'koi_slogg', 'koi_srad',
        'koi_insol', 'koi_score'
    ]
    available = [c for c in desired if c in df.columns]
    print("Available desired columns:", available)
    
    # إذا لم توجد الأعمدة المطلوبة، استخدم كل الأعمدة الرقمية
    if len(available) < 5:
        print("⚠ Few desired columns available, using all numeric columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available = numeric_cols
        print(f"Using {len(available)} numeric columns")
    
    df = df[available].copy()
    
    # Map dispositions -> label (0 FP, 1 Candidate, 2 Confirmed)
    if 'koi_disposition' in df.columns:
        print("Mapping dispositions to labels...")
        df['label'] = df['koi_disposition'].map({
            'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2
        })
        # إزالة الصفوف التي لا تحتوي على تسميات صالحة
        initial_count = len(df)
        df = df.dropna(subset=['label'])
        final_count = len(df)
        print(f"Labels mapped. Kept {final_count}/{initial_count} rows with valid labels")
    else:
        print("⚠ No koi_disposition column - trying to create labels from available data")
        # إذا لم يكن هناك عمود تسميات، حاول إنشاء تسميات بسيطة
        if 'koi_score' in df.columns:
            df['label'] = np.where(df['koi_score'] >= 0.9, 2, 
                                  np.where(df['koi_score'] >= 0.5, 1, 0))
            print("Created labels from koi_score")
        else:
            # إذا لا يوجد أي عمود للتسميات، استخدم تجميع K-means بسيط
            from sklearn.cluster import KMeans
            numeric_data = df.select_dtypes(include=[np.number]).fillna(0)
            if len(numeric_data) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
                df['label'] = kmeans.fit_predict(numeric_data)
                print("Created labels using K-means clustering")
            else:
                print("❌ Cannot create labels - insufficient data")
                return

    # drop rows without label
    df = df.dropna(subset=['label'])
    if len(df) == 0:
        print("❌ No data after label processing")
        return
    # هندسة الميزات
    print("🛠 Engineering features...")
    df, engineered_features = robust_feature_engineering(df)
    
    # إذا لم تكن هناك ميزات كافية، استخدم الميزات البديلة
    if len(engineered_features) < 3:
        print("⚠ Not enough engineered features, using fallback features")
        features = create_fallback_features(df)
    else:
        features = engineered_features
    
    print(f"🎯 Final features to use ({len(features)}): {features}")

    # 🔥🔥🔥 الإضافة الجديدة المحسنة 🔥🔥🔥
    # إضافة الميزات الجديدة المقترحة من البيانات الأصلية
    print("🔍 Adding recommended FP features from original data...")
    
    # نحتاج نرجع نحمل البيانات عشان نجيب الأعمدة الجديدة
    original_df = load_kepler(KEPLER_FILE)
    recommended_features = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
    
    features_added = 0
    for feature in recommended_features:
        if feature in original_df.columns and original_df[feature].notna().sum() > 0:
            # نضيف العمود للـ df الحالي
            df[feature] = original_df[feature].fillna(0)
            if feature not in features:
                features.append(feature)
                features_added += 1
                print(f"✅ Added recommended feature: {feature}")
    
    if features_added > 0:
        print(f"🎯 Final features after additions: {len(features)} features")
    else:
        print("ℹ️ No new features were added")
    # 🔥🔥🔥 نهاية الإضافة 🔥🔥🔥

    # تنظيف البيانات - إستراتيجية أكثر مرونة
    print("🧹 Cleaning data...")
    
    # أولاً: احسب نسبة البيانات المفقودة لكل ميزة
    missing_info = {}
    for feature in features:
        if feature in df.columns:
            missing_count = df[feature].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_info[feature] = missing_pct
            print(f"   {feature}: {missing_pct:.1f}% missing")
    
    # استخدم إستراتيجيات مختلفة بناءً على نسبة البيانات المفقودة
    df_clean = df.copy()
    
    for feature in features:
        if feature in df_clean.columns and df_clean[feature].isna().any():
            missing_pct = missing_info.get(feature, 0)
            
            if missing_pct < 50:  # إذا أقل من 50% مفقود، املأ بالقيم الوسيطة
                median_val = df_clean[feature].median()
                df_clean[feature] = df_clean[feature].fillna(median_val)
                print(f"   Filled {feature} ({missing_pct:.1f}% missing) with median: {median_val:.4f}")
            else:  # إذا أكثر من 50% مفقود، استخدم 0 أو قيم افتراضية
                df_clean[feature] = df_clean[feature].fillna(0)
                print(f"   Filled {feature} ({missing_pct:.1f}% missing) with 0")
    
    # إزالة الصفوف التي لا تزال تحتوي على قيم NaN (إذا بقي أي)
    initial_clean_count = len(df_clean)
    df_clean = df_clean.dropna(subset=features + ['label'])
    final_clean_count = len(df_clean)
    
    print(f"✅ Data cleaning complete: {final_clean_count}/{initial_clean_count} rows remaining")
    
    if final_clean_count == 0:
        print("❌ No data remaining after cleaning. Trying alternative approach...")
        
        # محاولة بديلة: استخدام imputer متقدم
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        df_alt = df.copy()
        numeric_data = df_alt[features].select_dtypes(include=[np.number])
        
        if len(numeric_data) > 10:  # إذا كان هناك بيانات كافية للimputation
            imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
            imputed_data = imputer.fit_transform(numeric_data)
            df_imputed = pd.DataFrame(imputed_data, columns=features, index=df_alt.index)
            
            # استبدال البيانات في DataFrame الأصلي
            for feature in features:
                df_alt[feature] = df_imputed[feature]
            
            df_clean = df_alt.dropna(subset=features + ['label'])
            print(f"✅ Alternative imputation: {len(df_clean)} rows recovered")
        
        if len(df_clean) == 0:
            print("❌ All data cleaning attempts failed")
            return

    if not check_data_sufficiency(df_clean[features], df_clean['label']):
        print("❌ Insufficient data for training")
        return

    # حفظ البيانات النظيفة
    df_clean.to_csv(os.path.join(CLEANED_DIR, "kepler_cleaned.csv"), index=False)
    print("💾 Saved cleaned data to", os.path.join(CLEANED_DIR, "kepler_cleaned.csv"))

    if QUICK:
        df_clean = df_clean.sample(frac=QUICK_FRAC, random_state=RANDOM_STATE).reset_index(drop=True)
        print("QUICK MODE: using", len(df_clean), "rows")

    X = df_clean[features].values
    y = df_clean['label'].astype(int).values

    # train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        print("⚠ Stratification failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
    print(f"📊 Train/test sizes: {X_train.shape[0]}/{X_test.shape[0]}")

    # scale numeric features (for NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # compute class weights
    try:
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {c: w for c, w in zip(classes, cw)}
        print("Class weights:", class_weight)
    except Exception as e:
        print("⚠ Could not compute class weights:", e)
        class_weight = None

    models = {}
    val_accuracies = {}

    # ---------- Random Forest ----------
    try:
        print("🌲 Training RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=min(RF_N_ESTIMATORS, len(X_train)), 
            max_depth=18, n_jobs=-1, random_state=RANDOM_STATE
        )
        rf.fit(X_train_scaled, y_train)
        rf_val_acc = rf.score(X_test_scaled, y_test)
        print("RF val acc:", rf_val_acc)
        models['rf'] = rf
        val_accuracies['rf'] = rf_val_acc
        joblib.dump(rf, os.path.join(MODELS_DIR, "rf.joblib"))
    except Exception as e:
        print("❌ RandomForest training failed:", e)

    # ---------- XGBoost ----------
    try:
        print("🚀 Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob', 
            num_class=len(np.unique(y_train)),
            n_estimators=min(XGB_N_ESTIMATORS, len(X_train)), 
            max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric='mlogloss', 
            n_jobs=-1, random_state=RANDOM_STATE
        )
        xgb_clf.fit(X_train_scaled, y_train)
        xgb_val_acc = xgb_clf.score(X_test_scaled, y_test)
        print("XGB val acc:", xgb_val_acc)
        models['xgb'] = xgb_clf
        val_accuracies['xgb'] = xgb_val_acc
        joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "xgb.joblib"))
    except Exception as e:
        print("❌ XGBoost training failed:", e)

    # ---------- Neural Network ----------
    try:
        if USE_TF and len(X_train) > 50:  # فقط إذا كانت البيانات كافية
            print("🧠 Training Keras Neural Network...")
            nn_model = build_keras_nn(X_train_scaled.shape[1])
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=NN_PATIENCE, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
            ]
            ckpt = os.path.join(MODELS_DIR, "nn_best.h5")
            callbacks.append(ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True, verbose=1))
            
            # استخدام validation split صغير إذا البيانات قليلة
            val_split = 0.15 if len(X_train) > 50 else 0.3
            history = nn_model.fit(
                X_train_scaled, y_train, 
                validation_split=val_split, 
                epochs=NN_EPOCHS, 
                batch_size=min(NN_BATCH, len(X_train)),
                class_weight=class_weight, 
                callbacks=callbacks, 
                verbose=2
            )
            nn_val_acc = nn_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
            print("NN val acc:", nn_val_acc)
            models['nn'] = nn_model
            val_accuracies['nn'] = nn_val_acc
            nn_model.save(os.path.join(MODELS_DIR, "nn_tf.h5"))
        else:
            print("📊 Training sklearn MLPClassifier...")
            mlp = MLPClassifier(
                hidden_layer_sizes=(128,64), 
                max_iter=500, 
                random_state=RANDOM_STATE
            )
            mlp.fit(X_train_scaled, y_train)
            nn_val_acc = mlp.score(X_test_scaled, y_test)
            print("MLP val acc:", nn_val_acc)
            models['mlp'] = mlp
            val_accuracies['mlp'] = nn_val_acc
            joblib.dump(mlp, os.path.join(MODELS_DIR, "mlp_sklearn.joblib"))
    except Exception as e:
        print("❌ Neural Network training failed:", e)

    # حفظ السكالر
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

    # ---------- Ensemble ----------
    if len(models) >= 2:
        print(f"🤝 Building ensemble from {len(models)} models...")
        
        # جمع احتمالات التنبؤ من جميع النماذج الناجحة
        all_probs = []
        model_weights = []
        
        for name, model in models.items():
            try:
                if name == 'nn' and USE_TF:
                    probs = model.predict(X_test_scaled)
                else:
                    probs = model.predict_proba(X_test_scaled)
                all_probs.append(probs)
                model_weights.append(val_accuracies[name])
            except Exception as e:
                print(f"⚠ Could not get predictions from {name}: {e}")
        
        if len(all_probs) >= 2:
            # تطبيق الأوزان بناءً على دقة النماذج
            weights = np.array(model_weights)
            weights = weights / weights.sum()  # تطبيع الأوزان
            
            probs_ensemble = np.zeros_like(all_probs[0])
            for i, probs in enumerate(all_probs):
                probs_ensemble += weights[i] * probs
                
            y_pred_ensemble = np.argmax(probs_ensemble, axis=1)

            # ---------- Evaluation ----------
            print("\nEnsemble classification report:")
            print(classification_report(y_test, y_pred_ensemble, digits=4))
            ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
            print("Ensemble accuracy:", ensemble_acc)

            # Confusion matrix plot
            cm = confusion_matrix(y_test, y_pred_ensemble)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["FP","CAND","CONF"], yticklabels=["FP","CAND","CONF"])
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Ensemble Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_confusion_matrix.png"))
            plt.close()

            # Save predictions on test set
            out_df = pd.DataFrame(probs_ensemble, columns=["prob_fp","prob_candidate","prob_confirmed"])
            out_df["y_true"] = y_test
            out_df["y_pred"] = y_pred_ensemble
            out_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_test_predictions.csv"), index=False)
        else:
            print("❌ Not enough models for ensemble")
    else:
        print("❌ Need at least 2 successful models for ensemble")

    # ---------- Feature Importance ----------
    for name, model in models.items():
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                plt.figure(figsize=(8,6))
                idx = np.argsort(importance)[::-1]
                # عرض أهم 10 ميزات فقط إذا كان هناك أكثر من 10
                n_features = min(10, len(features))
                feat_names = np.array(features)[idx][:n_features]
                plt.barh(feat_names, importance[idx][:n_features])
                plt.title(f"{name.upper()} Feature Importances")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.png"))
                plt.close()
                print(f"💡 Top 3 features for {name}: {feat_names[:3]}")
        except Exception as e:
            print(f"Could not plot {name} importance:", e)

    elapsed = time.time() - t0
    print(f"\n🎉 Pipeline completed. Time elapsed: {elapsed:.1f} seconds")
    print(f"📁 Results saved to {OUTPUT_DIR}, models saved to {MODELS_DIR}")
    print(f"✅ Successful models: {list(models.keys())}")
    # تحليل إضافي للبيانات
def analyze_data_quality():
    """تحليل جودة البيانات واقتراح تحسينات"""
    df_clean = pd.read_csv(os.path.join(CLEANED_DIR, "kepler_cleaned.csv"))
    
    print("\n" + "="*50)
    print("📈 DATA QUALITY ANALYSIS")
    print("="*50)
    
    # تحليل الميزات المتوفرة
    available_features = []
    for col in df_clean.columns:
        if col != 'label' and df_clean[col].nunique() > 1:  # ميزات ذات تباين
            available_features.append(col)
    
    print(f"✅ Features with variance: {available_features}")
    
    # اقتراح ميزات إضافية من البيانات الأصلية
    original_df = load_kepler(KEPLER_FILE)
    potential_features = [
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'koi_impact', 'koi_ror', 'koi_eccen'
    ]
    
    new_features = []
    for feature in potential_features:
        if feature in original_df.columns and original_df[feature].notna().sum() > 0:
            new_features.append(feature)
    
    print(f"💡 Suggested additional features: {new_features}")
    
    return new_features

# تشغيل التحليل بعد الانتهاء
if __name__ == "__main__":
    main()
    
    new_features = analyze_data_quality()
    
    if new_features:
        print(f"\n🎯 RECOMMENDATION: Add these features for better performance:")
        for feat in new_features:
            print(f"   - {feat}")