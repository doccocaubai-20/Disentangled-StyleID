"""Row4 CPU-only LEACE fit/evaluation on row3 frozen benchmark caches. No GPU training."""
import numpy as np
"""Deterministic verification metrics, using only the Python standard library.

Scores are similarities: a pair is accepted when ``score >= threshold``.
``None`` represents reject-all, so JSON never needs an infinite threshold.
An oracle threshold is fitted on the evaluated labels and is diagnostic only;
it must never be described as a deployable or held-out accuracy estimate.
"""

from collections.abc import Mapping
from fractions import Fraction
from hashlib import sha256
from itertools import groupby
import json
import math
from numbers import Real


METRIC_PROTOCOL = {
    "version": "styleid-research-v2-metrics-1",
    "acceptance": "score >= threshold; null threshold rejects all",
    "auc": "pairwise rank probability with half credit for ties",
    "tpr": "max empirical TPR with FPR <= 1/100; no interpolation",
    "operating_ties": "prefer fewer false positives then higher threshold",
    "oracle": "exact score tie blocks; prefer higher threshold on accuracy ties",
    "fixed_thresholds": [0.3, 0.4, 0.5],
    "quantiles": "linear interpolation at (n-1)*q",
}


def hash_protocol(protocol):
    """Hash a JSON-compatible protocol; callers must include pair/input details.

    Reject NaN and infinity rather than emitting nonstandard JSON. Key ordering
    is ignored; list ordering remains meaningful (including pair order).
    """
    encoded = json.dumps(
        protocol, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


METRIC_PROTOCOL_HASH = hash_protocol(METRIC_PROTOCOL)
_UNSET = object()
_DIRECTIONS = {
    "tpr_at_fpr_0_01": "higher", "auc": "higher",
    "acc_at_0_3": "higher", "acc_at_0_4": "higher", "acc_at_0_5": "higher",
    "oracle_best_accuracy": "higher", "positive_mean": "higher",
    "negative_mean": "lower", "margin": "higher",
    "accuracy_at_validation_threshold": "higher",
}


def _validated(labels, scores):
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores):
        raise ValueError("labels and scores must have the same nonzero length")
    clean_labels, clean_scores = [], []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if not isinstance(label, Real) or label not in (0, 1):
            raise ValueError(f"label at index {index} must be binary (0 or 1)")
        if not isinstance(score, Real) or isinstance(score, bool):
            raise ValueError(f"score at index {index} must be a finite real number")
        try:
            score = float(score)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"score at index {index} is not representable") from exc
        if not math.isfinite(score):
            raise ValueError(f"score at index {index} must be finite")
        clean_labels.append(int(label))
        clean_scores.append(score)
    n_positive = sum(clean_labels)
    if n_positive == 0 or n_positive == len(clean_labels):
        raise ValueError("both positive and negative pairs are required")
    return clean_labels, clean_scores


def _blocks(labels, scores):
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    result = []
    for score, pairs in groupby(ordered, key=lambda pair: pair[0]):
        group_labels = [label for _, label in pairs]
        positive = sum(group_labels)
        result.append((score, positive, len(group_labels) - positive))
    return result


def _operating_point(blocks, n_positive, n_negative, max_fpr):
    limit = Fraction(str(max_fpr))
    tp = fp = best_tp = best_fp = 0
    threshold = None
    for score, positive, negative in blocks:
        tp += positive
        fp += negative
        # Integer comparison keeps the 1% boundary exact, including n_neg < 100.
        if fp * limit.denominator > n_negative * limit.numerator:
            break
        if tp > best_tp:
            best_tp, best_fp, threshold = tp, fp, score
    return {
        "threshold": threshold, "tpr": best_tp / n_positive,
        "fpr": best_fp / n_negative,
    }


def _oracle(blocks, n_positive, n_negative):
    tp = fp = 0
    best_correct, threshold = n_negative, None
    for score, positive, negative in blocks:
        tp += positive
        fp += negative
        correct = tp + n_negative - fp
        if correct > best_correct:
            best_correct, threshold = correct, score
    return best_correct / (n_positive + n_negative), threshold


def _threshold_counts(labels, scores, threshold):
    if threshold is not None:
        if not isinstance(threshold, Real) or not math.isfinite(threshold):
            raise ValueError("threshold must be finite or None (reject all)")
        threshold = float(threshold)
    tp = sum(y == 1 and threshold is not None and s >= threshold
             for y, s in zip(labels, scores))
    fp = sum(y == 0 and threshold is not None and s >= threshold
             for y, s in zip(labels, scores))
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "threshold": threshold, "accuracy": (tp + negatives - fp) / len(labels),
        "tpr": tp / positives, "fpr": fp / negatives,
    }


def evaluate_at_threshold(labels, scores, threshold):
    """Evaluate a caller-frozen threshold; None explicitly rejects every pair."""
    labels, scores = _validated(labels, scores)
    return _threshold_counts(labels, scores, threshold)


def select_validation_threshold(labels, scores, *, objective="accuracy", max_fpr=0.01):
    """Fit a threshold on validation ONLY, then freeze it for separate test data.

    ``objective`` is ``accuracy`` or ``tpr_at_fpr``. This function cannot establish
    that the caller's validation identities are disjoint from test identities.
    Accuracy ties retain the most conservative decision rule, possibly None.
    """
    labels, scores = _validated(labels, scores)
    blocks = _blocks(labels, scores)
    positive, negative = sum(labels), len(labels) - sum(labels)
    if objective == "accuracy":
        _, threshold = _oracle(blocks, positive, negative)
    elif objective == "tpr_at_fpr":
        if (not isinstance(max_fpr, Real) or not math.isfinite(max_fpr)
                or not 0 <= max_fpr <= 1):
            raise ValueError("max_fpr must be finite and between zero and one")
        threshold = _operating_point(blocks, positive, negative, max_fpr)["threshold"]
    else:
        raise ValueError("objective must be 'accuracy' or 'tpr_at_fpr'")
    return {
        **_threshold_counts(labels, scores, threshold), "objective": objective,
        "source": "validation", "n_validation_pairs": len(labels),
    }


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    # Weighted terms avoid overflow from subtracting extreme finite scores.
    return values[lower] * (1 - weight) + values[upper] * weight


def evaluate_scores(labels, scores, *, validation_threshold=_UNSET):
    """Return exact rank/fixed-threshold metrics and score-tail diagnostics.

    Optional ``validation_threshold`` must already be frozen on separate data;
    explicitly passing None evaluates a reject-all validation decision rule.
    ``oracle_best_accuracy``/``oracle_threshold`` are fitted on THESE scores.
    For fair comparisons the caller must add ``protocol_hash`` (input protocol)
    and ``pair_manifest_hash`` (ordered pair identities and labels).
    """
    labels, scores = _validated(labels, scores)
    positive_scores = [s for y, s in zip(labels, scores) if y == 1]
    negative_scores = [s for y, s in zip(labels, scores) if y == 0]
    n_positive, n_negative = len(positive_scores), len(negative_scores)
    blocks = _blocks(labels, scores)
    negative_above = twice_concordant = 0
    for _, positive, negative in blocks:
        negative_below = n_negative - negative_above - negative
        twice_concordant += positive * (2 * negative_below + negative)
        negative_above += negative
    operating = _operating_point(blocks, n_positive, n_negative, 0.01)
    oracle_accuracy, oracle_threshold = _oracle(blocks, n_positive, n_negative)
    positive_mean = math.fsum(s / n_positive for s in positive_scores)
    negative_mean = math.fsum(s / n_negative for s in negative_scores)
    margin = positive_mean - negative_mean
    if not math.isfinite(margin):
        raise ValueError("finite scores produce an unrepresentable margin")
    result = {
        "tpr_at_fpr_0_01": operating["tpr"],
        "auc": twice_concordant / (2 * n_positive * n_negative),
        "acc_at_0_3": _threshold_counts(labels, scores, 0.3)["accuracy"],
        "acc_at_0_4": _threshold_counts(labels, scores, 0.4)["accuracy"],
        "acc_at_0_5": _threshold_counts(labels, scores, 0.5)["accuracy"],
        "oracle_best_accuracy": oracle_accuracy, "oracle_threshold": oracle_threshold,
        "oracle_is_deployable": False,
        "positive_mean": positive_mean, "negative_mean": negative_mean,
        "margin": margin, "n_pairs": len(labels),
        "n_positive": n_positive, "n_negative": n_negative,
        "fpr_at_operating_point": operating["fpr"],
        "threshold_at_fpr_0_01": operating["threshold"],
        "negative_quantile_0_99": _quantile(negative_scores, 0.99),
        "positive_quantile_0_01": _quantile(positive_scores, 0.01),
        "positive_quantile_0_05": _quantile(positive_scores, 0.05),
        "metric_protocol_hash": METRIC_PROTOCOL_HASH,
    }
    if validation_threshold is not _UNSET:
        validation = _threshold_counts(labels, scores, validation_threshold)
        result.update({
            "validation_threshold": validation["threshold"],
            "accuracy_at_validation_threshold": validation["accuracy"],
            "tpr_at_validation_threshold": validation["tpr"],
            "fpr_at_validation_threshold": validation["fpr"],
        })
    return result


def _finite_number(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def compare_to_baseline(candidate_by_dataset, baseline_by_dataset, required_metrics_by_dataset):
    """Require a strict numeric improvement in EVERY declared metric/dataset.

    Each dataset maps to an evaluate_scores result with caller-supplied protocol
    and pair manifest hashes. Hashes and pair counts must match the live baseline.
    Missing/nonfinite metrics or missing/mismatched provenance make the result
    not_evaluable, never a win. No tolerance, rounding or averaging grants wins.
    A win here is historical numeric dominance, not statistical significance or
    evidence of independent generalization after repeated test-set inspection.
    """
    if not isinstance(required_metrics_by_dataset, Mapping) or not required_metrics_by_dataset:
        raise ValueError("at least one dataset and required metric must be declared")
    dataset_results = {}
    evaluable = dominant = True
    for dataset, required in required_metrics_by_dataset.items():
        if isinstance(required, (str, bytes)):
            raise ValueError("required metrics must be a sequence of names")
        required = list(required)
        if not required or len(set(required)) != len(required):
            raise ValueError("each dataset needs nonempty, unique required metrics")
        if any(metric not in _DIRECTIONS for metric in required):
            raise ValueError("required metrics contain an unknown or nondirectional field")
        candidate = candidate_by_dataset.get(dataset)
        baseline = baseline_by_dataset.get(dataset)
        errors = []
        if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
            errors.append("candidate or baseline dataset is missing")
        candidate = candidate if isinstance(candidate, Mapping) else {}
        baseline = baseline if isinstance(baseline, Mapping) else {}
        for key in ("metric_protocol_hash", "protocol_hash", "pair_manifest_hash"):
            c, b = candidate.get(key), baseline.get(key)
            if not isinstance(c, str) or not c or not isinstance(b, str) or not b:
                errors.append(f"{key} missing")
            elif c != b:
                errors.append(f"{key} mismatch")
        for key in ("n_pairs", "n_positive", "n_negative"):
            c, b = candidate.get(key), baseline.get(key)
            if (not isinstance(c, int) or isinstance(c, bool) or c <= 0
                    or not isinstance(b, int) or isinstance(b, bool) or b <= 0):
                errors.append(f"{key} missing or invalid")
            elif c != b:
                errors.append(f"{key} mismatch")
        for name, record in (("candidate", candidate), ("baseline", baseline)):
            counts = [record.get(key) for key in ("n_pairs", "n_positive", "n_negative")]
            if all(isinstance(value, int) and not isinstance(value, bool) for value in counts):
                if counts[0] != counts[1] + counts[2]:
                    errors.append(f"{name} pair counts are inconsistent")
        provenance_invalid = bool(errors)
        metric_results = {}
        for metric in required:
            c, b = candidate.get(metric), baseline.get(metric)
            valid = _finite_number(c) and _finite_number(b)
            if not valid:
                errors.append(f"{metric} missing or nonfinite")
            if provenance_invalid or not valid:
                status = "not_evaluable"
            elif c == b:
                status = "tie"
            else:
                improved = c > b if _DIRECTIONS[metric] == "higher" else c < b
                status = "win" if improved else "loss"
            metric_results[metric] = {
                "candidate": c if _finite_number(c) else None,
                "baseline": b if _finite_number(b) else None,
                "direction": _DIRECTIONS[metric], "status": status,
            }
        dataset_evaluable = not errors
        dataset_dominant = dataset_evaluable and all(
            value["status"] == "win" for value in metric_results.values()
        )
        evaluable = evaluable and dataset_evaluable
        dominant = dominant and dataset_dominant
        dataset_results[dataset] = {
            "evaluable": dataset_evaluable,
            "historical_numeric_dominance": dataset_dominant,
            "gate_errors": errors, "metrics": metric_results,
        }
    return {
        "status": ("not_evaluable" if not evaluable else
                   "historical_numeric_dominance" if dominant else "not_dominant"),
        "evaluable": evaluable, "historical_numeric_dominance": dominant,
        "datasets": dataset_results,
    }

def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm < 1e-12) or not np.isfinite(x).all():
        raise ValueError('Invalid/zero transformed embedding')
    return x/norm

def fit_leace(x, labels, relative_tol=1e-10):
    """Return low-rank factors for row-vector affine erasure.

    Covariance eigensystem obtained by thin SVD of centered X, so singular
    high-dimensional covariance is handled without ridge/tuning on evaluation.
    """
    x = np.asarray(x, dtype=np.float64)
    classes, index = np.unique(labels, return_inverse=True)
    z = np.eye(len(classes))[index]
    mean = x.mean(0)
    xc, zc = x-mean, z-z.mean(0)
    _, singular, vt = np.linalg.svd(xc, full_matrices=False)
    eigenvalues = singular**2/(len(x)-1)
    keep = eigenvalues > eigenvalues.max()*relative_tol
    basis = vt[keep].T
    roots = np.sqrt(eigenvalues[keep])
    cross = xc.T@zc/(len(x)-1)
    whitened_cross = (basis/roots)@(basis.T@cross)
    u, s, _ = np.linalg.svd(whitened_cross, full_matrices=False)
    keep_concept = s > s.max()*relative_tol if s.max() > 0 else np.zeros_like(s, dtype=bool)
    u = u[:, keep_concept]
    left = (basis*roots)@(basis.T@u)
    right = ((u.T@basis)/roots)@basis.T
    result = {'mean': mean, 'left': left, 'right': right}
    after = transform(x, result, renormalize=False)
    residual = (after-after.mean(0)).T@zc/(len(x)-1)
    result['audit'] = {'erased_rank': int(u.shape[1]), 'covariance_rank': int(keep.sum()),
                       'fit_cross_covariance_before': float(np.linalg.norm(cross)),
                       'fit_cross_covariance_after': float(np.linalg.norm(residual))}
    return result

def transform(x, operator, renormalize=True):
    if operator is None:
        y = np.asarray(x, dtype=np.float64).copy()
    else:
        y = x-((x-operator['mean'])@operator['right'].T)@operator['left'].T
    return normalize(y) if renormalize else y

def lfw_metrics(pairs, scores):
    labels = [p['label'] for p in pairs]
    result = evaluate_scores(labels, scores)
    folds = sorted({p['fold'] for p in pairs})
    rows, exact_rows = [], []
    y, s = np.array(labels), np.array(scores)
    for fold in folds:
        fit = [i for i,p in enumerate(pairs) if p['fold'] != fold]
        test = [i for i,p in enumerate(pairs) if p['fold'] == fold]
        chosen = select_validation_threshold([labels[i] for i in fit], [scores[i] for i in fit], objective='accuracy')
        evaluated = evaluate_at_threshold([labels[i] for i in test], [scores[i] for i in test], chosen['threshold'])
        exact_rows.append(dict(fold=fold,calibration_folds=[j for j in folds if j != fold],
                               calibration_pairs=len(fit),test_pairs=len(test),**evaluated))
        # Preserve the exact algorithm in the user-supplied 6000-pair script.
        best_accuracy, threshold = 0.0, 0.0
        for value in np.linspace(-0.2,0.95,200):
            accuracy = np.mean((s[fit] >= value).astype(int) == y[fit])
            if accuracy > best_accuracy:
                best_accuracy, threshold = accuracy, float(value)
        grid_evaluated = evaluate_at_threshold([labels[i] for i in test], [scores[i] for i in test], threshold)
        rows.append(dict(fold=fold,calibration_folds=[j for j in folds if j != fold],
                         calibration_pairs=len(fit),test_pairs=len(test),**grid_evaluated))
    result.update(cv_accuracy=float(np.mean([r['accuracy'] for r in rows])),
                  cv_std=float(np.std([r['accuracy'] for r in rows])),cv_folds=rows,
                  cv_exact_accuracy=float(np.mean([r['accuracy'] for r in exact_rows])),
                  cv_exact_std=float(np.std([r['accuracy'] for r in exact_rows])),cv_exact_folds=exact_rows,
                  cv_note='Primary:200 threshold grid [-0.2,0.95] and first optimum, same as user reference; exact-search CV separate. Both fit other9 folds only; shared identities can span folds')
    return result

DEFAULT_FIT_IDS = [4, 5, 7, 12, 26, 27, 28, 29, 30, 36, 54, 55, 58, 63, 65, 70, 71, 72, 76, 78, 84, 87, 90, 92, 95, 115, 119, 122, 123, 124, 130, 134]

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from datetime import datetime, timezone


def run_leace(source, model, output, fit_ids, expected_backbone=None):
    """Accept a completed benchmark directory or the exported cache ZIP."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Choose a NEW output directory: ' + str(output))
    archive = zipfile.ZipFile(source) if source.is_file() else None
    inputs = {}

    def read(name):
        raw = archive.read(name) if archive else (source/name).read_bytes()
        inputs[name] = hashlib.sha256(raw).hexdigest()
        return raw

    try:
        manifest = json.loads(read('LEACE_EXPORT_MANIFEST.json')) if archive else None
        if manifest:
            for name, meta in manifest['files'].items():
                raw = read(name)
                if len(raw) != meta['bytes'] or inputs[name] != meta['sha256']:
                    raise ValueError('Corrupt exported cache: ' + name)
        report, preflight = [json.loads(read(n)) for n in ('results.json','preflight.json')]
        binding = report['binding']
        if report.get('status') != 'complete' or preflight['binding'] != binding:
            raise ValueError('Benchmark incomplete or binding mismatch')
        if manifest and manifest['binding'] != binding:
            raise ValueError('Manifest binding mismatch')
        checkpoint = preflight['checkpoints'][model]
        if expected_backbone and checkpoint['backbone_sha256'] != expected_backbone:
            raise ValueError('This cache is not from the requested row3 backbone')
        if model == 'soup':
            print('NOTICE: historical Soup cache. This is row3->row4 post-filter evidence;')
            print('it is NOT a matched comparison to a newly trained row2.')
        pairs, keys, vectors = {}, {}, {}
        for ds, count in [('sksf',1876),('lfw',6000)]:
            pairs[ds] = list(csv.DictReader(io.StringIO(read(ds+'_pairs.csv').decode('utf-8-sig'))))
            if len(pairs[ds]) != count:
                raise ValueError('Wrong pair count for ' + ds)
            for p in pairs[ds]:
                p['label'] = int(p['label'])
                if ds == 'lfw': p['fold'] = int(p['fold'])
            if ds == 'lfw':
                folds = sorted({p['fold'] for p in pairs[ds]})
                if len(folds) != 10 or any(sum(p['fold']==f and p['label']==y for p in pairs[ds])!=300 for f in folds for y in (0,1)):
                    raise ValueError('Expected official 10 folds, 300 positive + 300 negative per fold')
            with np.load(io.BytesIO(read(model+'__'+ds+'__embeddings.npz')), allow_pickle=False) as data:
                keys[ds], vectors[ds] = data['keys'].tolist(), data['vectors'].copy()
                if data['binding'].item() != binding+'|'+model+'|'+ds or int(data['done'].item()) != len(keys[ds]):
                    raise ValueError('Incomplete cache/binding mismatch: ' + ds)
            if vectors[ds].shape != (len(keys[ds]),512) or len(set(keys[ds])) != len(keys[ds]):
                raise ValueError('Expected unique keys and ID512 embeddings')
            if set(keys[ds]) != {p[k] for p in pairs[ds] for k in ('left','right')}:
                raise ValueError('Pair/cache key mismatch')
            if not np.isfinite(vectors[ds]).all() or not np.allclose(np.linalg.norm(vectors[ds],axis=1),1,atol=1e-5):
                raise ValueError('Expected finite normalized cache')
        ids = np.array([int(Path(k).stem) for k in keys['sksf']])
        styles = np.array([k.split('/')[0] for k in keys['sksf']])
        fit_ids = sorted(set(map(int,fit_ids)))
        if not fit_ids or not set(fit_ids) < set(ids.tolist()):
            raise ValueError('Fit IDs must be a nonempty strict subset of available SKSF identities')
        fit = np.isin(ids,fit_ids)
        if len(np.unique(styles[fit])) < 2:
            raise ValueError('Need at least 2 style labels to fit LEACE')
        filtered = [p for p in pairs['sksf'] if int(p['left_id']) not in fit_ids and int(p['right_id']) not in fit_ids]
        if not filtered or len({p['label'] for p in filtered})!=2:
            raise ValueError('Need held-out positive AND negative verification pairs')
        operator = fit_leace(vectors['sksf'][fit],styles[fit])
        changed = {ds:transform(v,operator) for ds,v in vectors.items()}
        results, summaries, all_scores = {}, [], {}
        for method, data in [('row3_raw',vectors),('row4_leace',changed)]:
            results[method] = {}
            for ds, selected in [('sksf',filtered),('lfw',pairs['lfw'])]:
                lookup = {k:i for i,k in enumerate(keys[ds])}
                scores = [float(np.dot(data[ds][lookup[p['left']]],data[ds][lookup[p['right']]])) for p in selected]
                metrics = lfw_metrics(selected,scores) if ds=='lfw' else evaluate_scores([p['label'] for p in selected],scores)
                results[method][ds] = metrics
                summaries.append(dict(method=method,dataset='sksf_filtered' if ds=='sksf' else 'lfw6000',
                    **{k:metrics.get(k,'') for k in ('n_pairs','tpr_at_fpr_0_01','auc','acc_at_0_3','acc_at_0_4','acc_at_0_5','cv_accuracy','cv_std')}))
                all_scores[method+'_'+ds] = [dict(**p,score=s) for p,s in zip(selected,scores)]
            # Validate that raw vectors reproduce the completed full benchmark.
            if method=='row3_raw':
                for ds in ('sksf','lfw'):
                    lookup = {k:i for i,k in enumerate(keys[ds])}
                    scores = [float(np.dot(data[ds][lookup[p['left']]],data[ds][lookup[p['right']]])) for p in pairs[ds]]
                    metrics = lfw_metrics(pairs[ds],scores) if ds=='lfw' else evaluate_scores([p['label'] for p in pairs[ds]],scores)
                    for k in ('auc','tpr_at_fpr_0_01','acc_at_0_3'):
                        if abs(metrics[k]-report['models'][model]['metrics'][ds][k]) > 1e-10:
                            raise ValueError('Cache does not reproduce benchmark: '+ds+'/'+k)
        output.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(output/'row4_leace_operator.npz', **{k:operator[k] for k in ('mean','left','right')})
        for name, rows in dict(summary=summaries, **all_scores).items():
            with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        provenance = dict(source=str(source),model=model,binding=binding,checkpoint=checkpoint,
            fit_ids=fit_ids,evaluation_ids=sorted(set(ids.tolist())-set(fit_ids)),
            sksf_primary_pairs=len(filtered),lfw_pairs=len(pairs['lfw']),
            operator_audit=operator['audit'],input_sha256=inputs,
            operator_sha256=hashlib.sha256((output/'row4_leace_operator.npz').read_bytes()).hexdigest(),
            limits=['Do NOT compare filtered SKSF against full1876 metrics of other rows.',
                    'Re-evaluate row1/row2/row3 on the same exported filtered pairs.',
                    'Fit identities held out only from LEACE evaluation, encoder exposure unknown.',
                    'New row2 vs historical Soup does NOT isolate the orthogonality loss.',
                    'Float64 empirical LEACE without shrinkage/trace cap; L2 after filtering.'],
            results=results)
        (output/'RESULTS.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(summaries,indent=2),flush=True)
        return provenance
    finally:
        if archive: archive.close()


if __name__ == '__main__':
    if Path('/content').is_dir() and not Path('/content/drive/MyDrive').is_dir():
        from google.colab import drive
        drive.mount('/content/drive')
    CACHE_SOURCE = Path(globals().get('CACHE_SOURCE','/content/drive/MyDrive/StyleID_A100_Champion/benchmark_3models_full_lfw_v2'))
    CACHE_MODEL = str(globals().get('CACHE_MODEL','soup'))
    OUTPUT = Path(globals().get('OUTPUT',str(CACHE_SOURCE.parent/('row4_leace_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))))
    run_leace(CACHE_SOURCE,CACHE_MODEL,OUTPUT,globals().get('FIT_IDS',DEFAULT_FIT_IDS),globals().get('EXPECTED_BACKBONE_SHA'))
    import shutil
    result_zip=shutil.make_archive(str(OUTPUT),'zip',OUTPUT)
    print('COMPLETE:',result_zip,flush=True)
    if Path('/content').is_dir():
        from google.colab import files
        files.download(result_zip)
