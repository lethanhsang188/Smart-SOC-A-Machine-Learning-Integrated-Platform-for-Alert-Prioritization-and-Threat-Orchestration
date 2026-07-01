import gc
import os
import json
import multiprocessing
from pathlib import Path
from typing import Iterator

import lightgbm as lgb
import numpy as np
import polars as pl
import tqdm
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from .features import PEFeatureExtractor


ORDERED_COLUMNS = [
    "sha256",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "label",
    "file_type",
    "family",
    "family_confidence",
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
]


def raw_feature_iterator(file_paths: list[Path]) -> Iterator[str]:
    """Yield raw feature strings from the inputed file paths"""
    for path in file_paths:
        with path.open("r") as fin:
            for line in fin:
                yield line


def gather_feature_paths(data_dir: Path | str, subset: str, filetype: str = None, week: str = None) -> list[Path]:
    """
    Gather paths to raw metadata .jsonl files in the given data_dir.
    Supports filtering by train/test/challenge subset, file type, and/or week.
    """
    feature_paths = []
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(Path(os.path.join(data_dir, file_name)))

    if not len(feature_paths):
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths


def read_label(raw_features_string: str, label_type: str) -> str:
    """Read the label or tag from raw features and return it"""
    raw_features = json.loads(raw_features_string)
    label = raw_features[label_type]
    return label


def read_label_unpack(args):
    """Pass through function for unpacking read_label arguments"""
    return read_label(*args)


def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> set:
    """Read the unique labels/tags in the subset"""
    pool = multiprocessing.Pool()
    argument_iterator = (
        (raw_features_string, label_type)
        for _, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    label_counts = {}
    for labels in tqdm.tqdm(pool.imap_unordered(read_label_unpack, argument_iterator), total=nrows):
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            if label_counts.get(label) is None:
                label_counts[label] = 0
            label_counts[label] += 1
    return label_counts


def vectorize(irow, raw_features_string, X_path, y_path, extractor, nrows, label_type="label", label_map={}):
    """Vectorize a single sample of raw features and write to a large numpy file"""
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError("Invalid label_type!")
    label = raw_features[label_type]

    if label is None and (label_type == "label" or label_type == "family"):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = -1
    elif isinstance(label, int):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = label
    elif isinstance(label, str):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        if label_map.get(label) is not None:
            y[irow] = label_map[label]
        else:
            y[irow] = -1
    elif isinstance(label, list):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows, len(label_map.keys())))
        for l in label:
            if label_map.get(l) is not None:
                y[irow, label_map[l]] = 1
    else:
        raise ValueError("Unable to parse label format")

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """Pass through function for unpacking vectorize arguments"""
    return vectorize(*args)


def vectorize_subset(X_path, y_path, raw_feature_paths, extractor, nrows, label_type="label", label_map={}):
    """Vectorize a subset of data and write it to disk"""
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    if label_type == "label" or label_type == "family":
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    else:
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(nrows, len(label_map.keys())))
    del X, y

    pool = multiprocessing.Pool()
    argument_iterator = (
        (irow, raw_features_string, X_path, y_path, extractor, nrows, label_type, label_map)
        for irow, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(data_dir, label_type="label", class_min=10):
    """Create feature vectors from raw features and write them to disk"""
    ignore_tags = set(["", "win32", "win64", "elf", "linux", "pdf", "apk", "android"])

    extractor = PEFeatureExtractor()
    data_path = Path(data_dir)

    print("Preparing to vectorize raw features")
    X_train_path = data_path / "X_train.dat"
    y_train_path = data_path / "y_train.dat"
    train_feature_paths = gather_feature_paths(data_path, "train")
    train_nrows = sum([1 for fp in train_feature_paths for _ in fp.open()])

    X_test_path = data_path / "X_test.dat"
    y_test_path = data_path / "y_test.dat"
    test_feature_paths = gather_feature_paths(data_path, "test")
    test_nrows = sum([1 for fp in test_feature_paths for _ in fp.open()])

    label_map = {}
    i = 0
    if label_type != "label":
        train_label_counts = read_label_subset(train_feature_paths, train_nrows, label_type)
        for l, count in train_label_counts.items():
            if l in ignore_tags:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing training set")
    vectorize_subset(X_train_path, y_train_path, train_feature_paths, extractor, train_nrows, label_type, label_map)

    if label_type != "label":
        test_label_counts = read_label_subset(test_feature_paths, test_nrows, label_type)
        for l, count in test_label_counts.items():
            if l in ignore_tags:
                continue
            if label_map.get(l) is not None:
                continue
            if count >= class_min:
                label_map[l] = i
                i += 1

    print("Vectorizing test set")
    vectorize_subset(X_test_path, y_test_path, test_feature_paths, extractor, test_nrows, label_type, label_map)

    print("Vectorizing challenge set")
    X_challenge_path = data_path / "X_challenge.dat"
    y_challenge_path = data_path / "y_challenge.dat"
    try:
        raw_feature_paths = gather_feature_paths(data_path, "challenge")
        nrows = sum([1 for fp in raw_feature_paths for _ in fp.open()])
        vectorize_subset(X_challenge_path, y_challenge_path, raw_feature_paths, extractor, nrows)
    except ValueError:
        print("  No challenge set found, skipping.")


def read_vectorized_features(data_dir, subset="train"):
    """
    Read vectorized features into memory.
    WARNING: Loads ALL data into RAM. Use train_model_memmap() on 32GB machines.
    """
    data_path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    extractor = PEFeatureExtractor()
    ndim = extractor.dim  # Luôn suy ra từ extractor, không mã hóa cứng
    X = np.memmap(X_path, dtype=np.float32, mode="r")
    X = np.array(X).reshape(-1, ndim)
    N = X.shape[0]
    y = np.memmap(y_path, dtype=np.int32, mode="r")
    y = np.array(y)
    if y.shape[0] > N:
        y = y.reshape(N, -1)

    return X, y


def read_vectorized_features_memmap(data_dir, subset="train"):
    """
    Read vectorized features as memory-mapped arrays (zero RAM copy).
    Data stays on disk, only accessed when explicitly indexed.
    """
    data_path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    extractor = PEFeatureExtractor()
    ndim = extractor.dim  # Luôn suy ra từ extractor, không mã hóa cứng

    X = np.memmap(X_path, dtype=np.float32, mode="r")
    y = np.memmap(y_path, dtype=np.int32, mode="r")
    n_samples = len(X) // ndim

    return X, y, ndim, n_samples


def _temporal_stratified_sample(y_mm, n_samples_total, sample_size, n_windows, seed):
    """
    Sample indices using temporal-stratified strategy.

    Divides data into n_windows equal time windows and samples proportionally
    from each window while maintaining the malware/benign ratio per window.
    This preserves temporal structure and avoids concept-drift leakage.

    Args:
        y_mm            : Full label memmap array (flat)
        n_samples_total : Total number of samples
        sample_size     : Target number of samples to select
        n_windows       : Number of temporal windows
        seed            : Random seed

    Returns:
        Sorted numpy array of selected indices
    """
    rng = np.random.default_rng(seed)
    window_size = n_samples_total // n_windows
    samples_per_window = sample_size // n_windows
    all_selected = []

    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size if w < n_windows - 1 else n_samples_total

        # Chỉ nạp nhãn của cửa sổ này (nhẹ: 4 byte x kích thước cửa sổ)
        week_labels = np.array(y_mm[start:end])
        labeled_local = np.where(week_labels != -1)[0]

        if len(labeled_local) == 0:
            continue

        n_take = min(samples_per_window, len(labeled_local))

        malware_local = labeled_local[week_labels[labeled_local] == 1]
        benign_local  = labeled_local[week_labels[labeled_local] == 0]

        if len(malware_local) == 0 or len(benign_local) == 0:
            chosen_local = rng.choice(labeled_local, n_take, replace=False)
        else:
            malware_ratio = len(malware_local) / len(labeled_local)
            n_malware = max(1, int(n_take * malware_ratio))
            n_benign  = n_take - n_malware
            n_malware = min(n_malware, len(malware_local))
            n_benign  = min(n_benign,  len(benign_local))

            chosen_malware = rng.choice(malware_local, n_malware, replace=False)
            chosen_benign  = rng.choice(benign_local,  n_benign,  replace=False)
            chosen_local   = np.concatenate([chosen_malware, chosen_benign])

        all_selected.append(chosen_local + start)

    selected = np.concatenate(all_selected)
    selected.sort()
    return selected


def train_model_memmap(
    data_dir,
    params={},
    sample_size=2_000_000,
    num_threads=16,
    seed=42,
    n_temporal_windows=52,
):
    """
    Train LightGBM model using memory-mapped data with temporal-stratified sampling.

    Designed for machines with limited RAM (32GB) while preserving the
    temporal structure of EMBER2024 for accurate malware detection.

    Strategy:
    - Data divided into n_temporal_windows equal time windows (default: 52 weeks)
    - Samples drawn proportionally from each window, stratified by label
    - Validation uses the most recent 10 percent of sampled data (temporal holdout)
    - Test set evaluated in chunks to avoid OOM

    Args:
        data_dir            : Path to directory containing X_train.dat / y_train.dat
        params              : LightGBM parameters (merged with defaults)
        sample_size         : Total labeled samples to use for training (default: 2M)
        num_threads         : CPU threads for LightGBM
        seed                : Random seed for reproducibility
        n_temporal_windows  : Number of time windows for stratified sampling (default: 52)

    Returns:
        (lgb.Booster, metrics_dict)
    """
    data_path = Path(data_dir)

    # Luôn suy ra ndim từ extractor, không mã hóa cứng
    extractor = PEFeatureExtractor()
    ndim = extractor.dim

    print("=" * 60)
    print("EMBER2024 Training -- Temporal-Stratified Memmap Mode")
    print("=" * 60)
    print(f"  Feature dim (ndim)  : {ndim}")
    print(f"  Data dir            : {data_path}")
    print(f"  Target sample size  : {sample_size:,}")
    print(f"  Temporal windows    : {n_temporal_windows}")
    print(f"  Threads             : {num_threads}")
    print(f"  Seed                : {seed}")
    print()

    # Bước 1: Mở memmap, chưa cấp phát RAM
    print("[1/6] Opening memmap files (no RAM copy)...")
    X_mm = np.memmap(data_path / "X_train.dat", dtype=np.float32, mode="r")
    y_mm = np.memmap(data_path / "y_train.dat", dtype=np.int32,   mode="r")

    n_total = len(X_mm) // ndim
    print(f"  Total samples in X_train.dat : {n_total:,}")

    if len(y_mm) != n_total:
        print(f"  WARNING: y_train length {len(y_mm)} != {n_total}. Using min.")
        n_total = min(n_total, len(y_mm))

    # Bước 2: Lấy mẫu phân tầng theo thời gian
    print("[2/6] Temporal-stratified sampling...")
    sample_idx = _temporal_stratified_sample(
        y_mm, n_total, sample_size, n_temporal_windows, seed
    )
    actual_sample = len(sample_idx)
    print(f"  Actual samples selected : {actual_sample:,}")

    # Bước 3: Nạp các dòng đã lấy mẫu vào RAM
    # Ước tính RAM: actual_sample x ndim x 4 byte
    # Với 2M mẫu x 2568 chiều: khoảng 20 GB, phù hợp với máy 32 GB
    estimated_gb = actual_sample * ndim * 4 / 1024**3
    print(f"[3/6] Loading {actual_sample:,} samples into RAM (~{estimated_gb:.1f} GB)...")

    X_sample = np.zeros((actual_sample, ndim), dtype=np.float32)
    y_sample = np.zeros(actual_sample, dtype=np.int32)

    report_step = 500_000
    for i, idx in enumerate(sample_idx):
        X_sample[i] = X_mm[idx * ndim : (idx + 1) * ndim]
        y_sample[i] = y_mm[idx]
        if (i + 1) % report_step == 0 or (i + 1) == actual_sample:
            print(f"    Loaded {i+1:,} / {actual_sample:,} ...")

    del X_mm, y_mm
    gc.collect()

    malware_count = int((y_sample == 1).sum())
    benign_count  = int((y_sample == 0).sum())
    print(f"  Malware : {malware_count:,}  |  Benign : {benign_count:,}")

    # Bước 4: Chia tập xác thực theo holdout thời gian
    # sample_idx đã được sắp xếp nên 10% cuối là các mẫu mới nhất,
    # mô phỏng quy trình xác thực theo thời gian trong thực tế.
    print("[4/6] Splitting train / validation (temporal holdout)...")
    split_point = int(actual_sample * 0.9)
    X_train_arr = X_sample[:split_point]
    y_train_arr = y_sample[:split_point]
    X_val_arr   = X_sample[split_point:]
    y_val_arr   = y_sample[split_point:]

    del X_sample, y_sample
    gc.collect()

    print(f"  Train : {len(X_train_arr):,}  |  Val : {len(X_val_arr):,}")

    # Bước 5: Huấn luyện LightGBM
    print("[5/6] Training LightGBM...")
    print("=" * 60)

    # Tham số mặc định được tinh chỉnh cho RAM 32 GB / 16 CPU / phân loại malware nhị phân
    default_params = {
        "objective"        : "binary",
        "metric"           : "auc",
        "boosting_type"    : "gbdt",
        "num_leaves"       : 255,
        "max_depth"        : 10,
        "learning_rate"    : 0.05,
        "feature_fraction" : 0.8,
        "bagging_fraction" : 0.8,
        "bagging_freq"     : 5,
        "min_data_in_leaf" : 50,
        "lambda_l1"        : 0.1,
        "lambda_l2"        : 0.1,
        "verbose"          : -1,
        "num_threads"      : num_threads,
        "seed"             : seed,
    }
    # Tham số người dùng ghi đè tham số mặc định
    default_params.update(params)

    # Chỉ số categorical_feature được cố định theo thứ tự đặc trưng của PEFeatureExtractor:
    # chỉ số 2     = is_pe        (GeneralFileInfo)
    # chỉ số 3-6   = start_bytes  (GeneralFileInfo)
    # chỉ số 701   = machine_type (HeaderFileInfo)
    # chỉ số 702   = subsystem    (HeaderFileInfo)
    train_set = lgb.Dataset(
        X_train_arr, y_train_arr,
        categorical_feature=[2, 3, 4, 5, 6, 701, 702],
        free_raw_data=True,
    )
    val_set = lgb.Dataset(
        X_val_arr, y_val_arr,
        reference=train_set,
        categorical_feature=[2, 3, 4, 5, 6, 701, 702],
        free_raw_data=True,
    )

    del X_train_arr
    gc.collect()

    model = lgb.train(
        default_params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    del train_set, val_set, X_val_arr, y_val_arr
    gc.collect()

    # Bước 6: Đánh giá trên toàn bộ tập test theo từng khối để tránh tràn bộ nhớ
    print("\n[6/6] Evaluating on full test set (chunked)...")

    X_test_mm = np.memmap(data_path / "X_test.dat", dtype=np.float32, mode="r")
    y_test_mm = np.memmap(data_path / "y_test.dat", dtype=np.int32,   mode="r")
    n_test_total = len(X_test_mm) // ndim

    chunk_size = 100_000
    all_preds  = []
    all_labels = []

    for start in range(0, n_test_total, chunk_size):
        end     = min(start + chunk_size, n_test_total)
        n_chunk = end - start

        X_chunk = np.array(X_test_mm[start * ndim : end * ndim]).reshape(n_chunk, ndim)
        y_chunk = np.array(y_test_mm[start:end])

        mask = y_chunk != -1
        if mask.sum() > 0:
            all_preds.extend(model.predict(X_chunk[mask]))
            all_labels.extend(y_chunk[mask].tolist())

        del X_chunk, y_chunk
        if (start // chunk_size) % 10 == 0 or end == n_test_total:
            print(f"    Processed {end:,} / {n_test_total:,} test samples...")

    del X_test_mm, y_test_mm
    gc.collect()

    from sklearn.metrics import classification_report, confusion_matrix

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_auc   = roc_auc_score(all_labels, all_preds)
    y_pred_bin = (all_preds > 0.5).astype(int)

    print(f"\n{'=' * 60}")
    print(f"  TEST AUC-ROC      : {test_auc:.6f}")
    print(f"  Test samples used : {len(all_labels):,}")
    print(f"  ndim verified     : {ndim}")
    print(f"{'=' * 60}")

    print("\nClassification Report:")
    print(classification_report(
        all_labels, y_pred_bin,
        target_names=["Benign", "Malware"],
        digits=4,
    ))

    cm = confusion_matrix(all_labels, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"               Benign   Malware")
    print(f"Actual Benign  {tn:7d}  {fp:7d}   FPR={fpr:.4f}")
    print(f"       Malware {fn:7d}  {tp:7d}   FNR={fnr:.4f}")

    metrics = {
        "test_auc"     : float(test_auc),
        "test_samples" : int(len(all_labels)),
        "fpr"          : float(fpr),
        "fnr"          : float(fnr),
        "ndim"         : ndim,
    }

    return model, metrics


def read_metadata_record(raw_features_string):
    """Decode a raw features string and return the metadata fields"""
    all_data = json.loads(raw_features_string)
    metadata_keys = set(ORDERED_COLUMNS)
    return {k: all_data[k] for k in all_data.keys() & metadata_keys}


def read_metadata(data_dir):
    """
    Read metadata from raw feature files and return as Polars DataFrames.
    Challenge set is optional -- returns empty DataFrame if not found.
    """
    pool = multiprocessing.Pool()
    data_path = Path(data_dir)

    train_feature_paths = gather_feature_paths(data_path, "train")
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
    train_metadf = pl.DataFrame(train_records).with_columns(subset=pl.lit("train")).select(ORDERED_COLUMNS)

    test_feature_paths = gather_feature_paths(data_path, "test")
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
    test_metadf = pl.DataFrame(test_records).with_columns(subset=pl.lit("test")).select(ORDERED_COLUMNS)

    try:
        challenge_feature_paths = gather_feature_paths(data_path, "challenge")
        challenge_records = list(pool.imap(read_metadata_record, raw_feature_iterator(challenge_feature_paths)))
        challenge_metadf = pl.DataFrame(challenge_records).with_columns(subset=pl.lit("challenge")).select(ORDERED_COLUMNS)
    except ValueError:
        challenge_metadf = pl.DataFrame(schema={col: pl.Utf8 for col in ORDERED_COLUMNS})

    return train_metadf, test_metadf, challenge_metadf


def optimize_model(data_dir):
    """
    Run a grid search to find the best LightGBM parameters.
    WARNING: Requires sufficient RAM to load all training data.
    """
    X_train, y_train = read_vectorized_features(data_dir, "train")
    train_rows = y_train != -1
    X_train_labeled = X_train[train_rows]
    y_train_labeled = y_train[train_rows]

    score = make_scorer(roc_auc_score, max_fpr=5e-3)
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train_labeled)

    fit_params = {"categorical_feature": [2, 3, 4, 5, 6, 701, 702]}
    param_grid = {
        "boosting_type"   : ["gbdt"],
        "objective"       : ["binary"],
        "num_iterations"  : [500, 1000],
        "learning_rate"   : [0.005, 0.05],
        "num_leaves"      : [512, 1024, 2048],
        "feature_fraction": [0.5, 0.8, 1.0],
        "bagging_fraction": [0.5, 0.8, 1.0],
    }
    grid = GridSearchCV(
        estimator=lgb.LGBMClassifier(n_jobs=-1, verbose=-1),
        cv=progressive_cv,
        param_grid=param_grid,
        scoring=score,
        n_jobs=1,
        verbose=3,
    )
    grid.fit(X_train_labeled, y_train_labeled, **fit_params)
    return grid.best_params_


def train_model(data_dir, params={}):
    """
    Train LightGBM model on the vectorized features.
    WARNING: Loads ALL data into RAM. Use train_model_memmap() on 32GB machines.
    """
    X, y = read_vectorized_features(data_dir, "train")

    if len(y.shape) != 1:
        raise ValueError("Encountered y_train with invalid shape. Use train_ovr_model() instead.")

    num_classes = np.max(y) + 1
    X = X[y != -1, :]
    y = y[y != -1]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
    val_set   = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=[2, 3, 4, 5, 6, 701, 702])

    if num_classes == 2:
        return lgb.train(params, train_set, valid_sets=val_set)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric"   : "multi_logloss",
    }
    params.update(lgbm_params)
    return lgb.train(params, train_set, valid_sets=val_set)


def train_ovr_model(data_dir, params={}):
    """
    Returns a list of One-vs-Rest (OvR) LightGBM classifiers trained on the vectorized features.
    """
    X, y = read_vectorized_features(data_dir, "train")

    if len(y.shape) != 2:
        raise ValueError("Encountered y_train with invalid shape. Use train_model() instead.")

    lgbm_models = []
    for i in range(y.shape[1]):
        lgbm_params = {
            "objective"   : "binary",
            "is_unbalance": True,
        }
        params.update(lgbm_params)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_i, test_size=0.1, stratify=y_i)
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        val_set   = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        lgbm_models.append(lgb.train(params, train_set, valid_sets=val_set))
    return lgbm_models


def predict_sample(lgbm_model, file_data):
    """
    Predict a PE file with a LightGBM model.
    Returns probability of being malware (0.0 = benign, 1.0 = malware).
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    predict_result = lgbm_model.predict([features])
    return float(predict_result[0])
