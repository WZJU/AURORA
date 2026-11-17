"""Test the V2 whole-geometry conditional flow-matching model.

The V2 experiment is self-contained: a 125-D lossless GGS geometry vector, a
57-D context (normalized AR + padded aerodynamic rows + validity mask), and a
joint-condition MLP flow. This script mirrors the four stages of
ablation/test_fm.py but uses only the V2 dataset/model code, so no VAE or
reference-chord rescaling is involved.

Stages:
    sample      Integrate the learned flow from Gaussian noise conditioned on
                each held-out test wing's AR and aerodynamic rows.
    reconstruct Decode each sampled vector straight back to a GGS wing dict
                (geometry.decode_wing) and save v2_wing_{i}.json.
    stl         Render each wing dict with AircraftScriptGenerator and run
                OpenVSP to export v2_wing_{i}.stl.
    evaluate    Geometry AR adherence plus the hybrid aerodynamic error
                (surrogate and physics-integral curves against target CL/CD/CM),
                written incrementally to v2_aero_test_results.csv.

Process files are written to <exp_dir>/test_results. Every stage is resumable.
"""

import argparse
import json
import sys
import traceback
from copy import deepcopy
from math import atan2, degrees
from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = V2_DIR.parent
SRC_DIR = PROJECT_ROOT / "main"
sys.path.insert(0, str(V2_DIR))        # geometry.py, Airfoil_FM.py, prepare_flow_dataset.py
sys.path.insert(0, str(SRC_DIR))       # AeroTransformerPredict.py
sys.path.insert(0, str(PROJECT_ROOT / "toolkit"))

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from Airfoil_FM import GeometryFlow
from geometry import decode_wing, projected_aspect_ratio
from prepare_flow_dataset import load_dataset, restore_geometry
from WingScriptGenerator import calculate_wing_mac

WORKING_DIR = V2_DIR / "experiments/fm_geometry/test_results"
VSP_EXE = Path("D:/SW/OpenVSP-3.45.4-win64/vsp.exe")
SAMPLES_PATH = WORKING_DIR / "v2_samples.pt"
FAILURES_PATH = WORKING_DIR / "v2_failures.json"
RESULTS_CSV = WORKING_DIR / "v2_aero_test_results.csv"
CL_CSV = WORKING_DIR / "v2_cl_error.csv"
VSP_SCRIPT = WORKING_DIR / "script_v2.vspscript"

COLUMNS = ["id", "wing_id", "aoa", "mach", "target_ar",
           "target_cl", "target_cd", "target_cm", "generated_ar", "ar_rel_error",
           "cd_loss_surr", "cm_loss_surr", "cd_loss_intg", "cm_loss_intg",
           "cl_loss_surr", "cl_loss_intg"]


# ---------------------------------------------------------------- utilities

def record_failure(stage: str, key: str, error: str) -> None:
    failures = {}
    if FAILURES_PATH.exists():
        try:
            failures = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures = {}
    failures.setdefault(stage, {})[str(key)] = error
    FAILURES_PATH.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_checkpoint(exp_dir: Path, ckpt: str) -> Path:
    if ckpt in ("best", "last"):
        matches = sorted((exp_dir / "checkpoints").glob(f"*{ckpt}*.ckpt"))
        if not matches:
            raise FileNotFoundError(f"No {ckpt} checkpoint in {exp_dir / 'checkpoints'}")
        return matches[-1]
    path = Path(ckpt).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_v2_dataset(path: Path) -> dict:
    """load_dataset already validates shapes, finite values and the split partition."""
    return load_dataset(path.resolve())


def test_indices(dataset: dict) -> list[int]:
    return sorted(dataset["splits"]["test"])


def generated_aspect_ratio(wing: dict) -> tuple[float, dict]:
    geometry = calculate_wing_mac(deepcopy(wing))
    return 2 * geometry["span"] ** 2 / geometry["area"], geometry


# ---------------------------------------------------------------- stages

def stage_sample(args, dataset: dict, indices: list[int]) -> None:
    device = torch.device(args.device)
    ckpt = resolve_checkpoint(args.exp_dir, args.ckpt)
    print(f"[sample] checkpoint: {ckpt}")
    model = GeometryFlow.load_from_checkpoint(str(ckpt), map_location="cpu").to(device).eval()
    context = dataset["context"][indices].to(device)

    torch.manual_seed(args.seed)
    with torch.inference_mode():
        sampled = model.sample(context, args.steps)

    # Sampling lives in normalized space; store raw physical coordinates so
    # reconstruction only needs the lossless geometry decoder.
    raw = restore_geometry(sampled, dataset["normalization"]).cpu()
    torch.save({"x": raw, "test_indices": indices,
                "wing_ids": [dataset["wing_ids"][row] for row in indices],
                "checkpoint": str(ckpt), "seed_base": args.seed, "steps": args.steps}, SAMPLES_PATH)
    reference = dataset["x_raw"][indices].cpu()
    scale_ratio = (raw.abs().mean() / reference.abs().mean()).item()
    print(f"[sample] wrote {raw.shape[0]} vectors to {SAMPLES_PATH}")
    print(f"[sample] raw/GT mean-|coord| ratio={scale_ratio:.3f}")


def stage_reconstruct(args, dataset: dict, indices: list[int]) -> None:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"{SAMPLES_PATH} missing; run the sample stage first.")
    payload = torch.load(SAMPLES_PATH, map_location="cpu", weights_only=False)
    selected = payload["test_indices"]

    limit = args.limit if args.limit else len(selected)
    failures = 0
    for position in range(limit):
        wing_path = WORKING_DIR / f"v2_wing_{position}.json"
        if wing_path.exists():
            continue
        try:
            wing = decode_wing(payload["x"][position].tolist())
            wing_path.write_text(json.dumps(wing, indent=2), encoding="utf-8")
        except Exception:
            failures += 1
            record_failure("reconstruct", position, traceback.format_exc(limit=2))
            print(f"[reconstruct] wing {position} FAILED (see v2_failures.json)")
        if (position + 1) % 25 == 0:
            print(f"[reconstruct] {position + 1}/{limit}")
    print(f"[reconstruct] done: {limit - failures}/{limit} wings written to {WORKING_DIR}")


def stage_stl(args, dataset: dict, indices: list[int]) -> None:
    from general_generation_system import AircraftScriptGenerator

    limit = args.limit if args.limit else len(indices)
    generator = AircraftScriptGenerator()
    failures = 0
    for position in range(limit):
        wing_path = WORKING_DIR / f"v2_wing_{position}.json"
        stl_path = WORKING_DIR / f"v2_wing_{position}.stl"
        if not wing_path.exists() or stl_path.exists():
            continue
        try:
            wing = json.loads(wing_path.read_text(encoding="utf-8"))
            generator.generate_script(
                script_file_path=str(VSP_SCRIPT),
                global_params=None, fuselage=None, wings=[wing],
                save_stl=True, save_file_path=str(WORKING_DIR).replace("\\", "/"),
                id_str=f"v2_wing_{position}",
            )
            import subprocess
            result = subprocess.run([str(VSP_EXE), "-script", str(VSP_SCRIPT)],
                                    capture_output=True, text=True, timeout=300)
            if not stl_path.exists():
                raise RuntimeError(f"OpenVSP returned {result.returncode}; no STL produced.")
        except Exception:
            failures += 1
            record_failure("stl", position, traceback.format_exc(limit=2))
            print(f"[stl] wing {position} FAILED (see v2_failures.json)")
        if (position + 1) % 25 == 0:
            print(f"[stl] {position + 1}/{limit}")
    print(f"[stl] done for {limit} wings ({failures} new failures)")


# ------------------------------------------------- aerodynamic error (notebook)

def find_nearest_two_numbers(lst, target):
    nearest1 = nearest2 = None
    min_diff1 = min_diff2 = float("inf")
    pos1 = pos2 = None
    for index, num in enumerate(lst):
        diff = abs(num - target)
        if num < target:
            if diff < min_diff1:
                min_diff2, nearest2, pos2 = min_diff1, nearest1, pos1
                min_diff1, nearest1, pos1 = diff, num, index
            elif diff < min_diff2 and num != nearest1:
                min_diff2, nearest2, pos2 = diff, num, index
        else:
            if diff < min_diff2:
                min_diff2, nearest2, pos2 = diff, num, index
            elif diff < min_diff1 and num != nearest2:
                min_diff1, nearest1, pos1 = diff, num, index
    return nearest1, pos1, nearest2, pos2


def calculate_aerodynamic_error(coeff_target: np.ndarray, coeff_pred: np.ndarray):
    pred_cl, pred_cd, pred_cm = coeff_pred[:, 0], coeff_pred[:, 1], coeff_pred[:, 2]
    target_cl, target_cd, target_cm = coeff_target[:, 0], coeff_target[:, 1], coeff_target[:, 2]

    total_cd = total_cm = total_rlt_cd = total_rlt_cm = 0.0
    for index in range(len(target_cl)):
        cl_1, pos_1, cl_2, pos_2 = find_nearest_two_numbers(pred_cl, target_cl[index])
        abs_error_cd = target_cd[index] - pred_cd[pos_1] - \
            (pred_cd[pos_2] - pred_cd[pos_1]) / (cl_2 - cl_1) * (target_cl[index] - cl_1)
        abs_error_cm = target_cm[index] - pred_cm[pos_1] - \
            (pred_cm[pos_2] - pred_cm[pos_1]) / (cl_2 - cl_1) * (target_cl[index] - cl_1)
        total_cd += abs_error_cd
        total_cm += abs_error_cm
        total_rlt_cd += abs_error_cd / abs(target_cd[index])
        total_rlt_cm += abs_error_cm / abs(target_cm[index])
    count = len(target_cl)
    return total_cd / count, total_cm / count, total_rlt_cd / count, total_rlt_cm / count


def cal_ac_hybrid_loss(surrogate, stl_path, root_chord, area, mac, ac_target):
    """Surrogate + physics-integral error curves, identical to the notebook."""
    from AeroTransformerPredict import integrate_wing_aerodynamics

    ac_target = np.asarray(ac_target)
    angle = torch.rad2deg(torch.atan2(torch.tensor(ac_target[:, 0]), torch.tensor(ac_target[:, 1])))
    working_condition = torch.stack([angle, torch.tensor(ac_target[:, 2])], dim=1)

    surrogate.build_grid(str(stl_path), root_chord=root_chord, area=area, mac=mac)
    s_ref = area / (root_chord ** 2)
    c_ref = mac / root_chord

    result = {
        "surrogate": {"cd": [], "cm": [], "cl": []},
        "integral": {"cd": [], "cm": [], "cl": []},
        "working_condition": working_condition,
    }
    for wc_i, wc in enumerate(working_condition):
        angle_range = torch.linspace(wc[0] - 2, wc[0] + 2, 40)
        mach_range = torch.full_like(angle_range, wc[1])
        coeff_target = ac_target[wc_i, 3:6].reshape(-1, 3)
        target_cl = float(ac_target[wc_i, 3])

        forces = surrogate.predict_coeffs(angle_range.numpy(), mach_range.numpy())
        s_cd, s_cm, _, _ = calculate_aerodynamic_error(coeff_target, forces[:, :3])
        # CL itself is used as the interpolation anchor for CD/CM, so it can
        # never show an error there. Measure it directly at the working point.
        wc_forces = surrogate.predict_coeffs(np.array([float(wc[0])]), np.array([float(wc[1])]))
        s_cl = float(wc_forces[0, 0]) - target_cl
        result["surrogate"]["cd"].append(float(s_cd))
        result["surrogate"]["cm"].append(float(s_cm))
        result["surrogate"]["cl"].append(s_cl)

        integral_rows = []
        for pt_idx in range(len(angle_range)):
            flowfield, origeom, _ = surrogate.predict_flow_field(aoa=float(angle_range[pt_idx]),
                                                                 mach=float(mach_range[pt_idx]))
            inp = torch.tensor(origeom, dtype=torch.float32).unsqueeze(0)
            outputs = torch.tensor(flowfield, dtype=torch.float32)
            res = integrate_wing_aerodynamics(inp, outputs, s_ref=s_ref, c_ref=c_ref,
                                              x_ref=0, y_ref=0, alpha_deg=float(angle_range[pt_idx]),
                                              include_friction=False)
            integral_rows.append([res["CL"], res["CD"], res["CM"]])
        i_cd, i_cm, _, _ = calculate_aerodynamic_error(coeff_target, np.asarray(integral_rows))

        flowfield, origeom, _ = surrogate.predict_flow_field(aoa=float(wc[0]), mach=float(wc[1]))
        inp = torch.tensor(origeom, dtype=torch.float32).unsqueeze(0)
        outputs = torch.tensor(flowfield, dtype=torch.float32)
        wc_res = integrate_wing_aerodynamics(inp, outputs, s_ref=s_ref, c_ref=c_ref,
                                             x_ref=0, y_ref=0, alpha_deg=float(wc[0]),
                                             include_friction=False)
        i_cl = float(wc_res["CL"]) - target_cl

        result["integral"]["cd"].append(float(i_cd))
        result["integral"]["cm"].append(float(i_cm))
        result["integral"]["cl"].append(i_cl)
        print(f"  condition {wc_i}: aoa={wc[0]:.2f} mach={wc[1]:.2f} | "
              f"surr cd={s_cd:.4f} cm={s_cm:.4f} cl={s_cl:+.4f} | "
              f"intg cd={i_cd:.4f} cm={i_cm:.4f} cl={i_cl:+.4f}")
    return result


def stage_evaluate(args, dataset: dict, indices: list[int]) -> None:
    from AeroTransformerPredict import AeroTransformer

    conditions = dataset["conditions"]
    wing_ids = dataset["wing_ids"]
    ar_target, ac, ac_mask = conditions["ar_target"], conditions["ac"], conditions["ac_mask"]

    done_ids: set[int] = set()
    if RESULTS_CSV.exists():
        done_ids = set(pd.read_csv(RESULTS_CSV)["id"].astype(int).tolist())
        print(f"[evaluate] resuming: {len(done_ids)} wings already in {RESULTS_CSV.name}")

    surrogate = AeroTransformer()
    limit = args.limit if args.limit else len(indices)
    rows = []
    for position in range(limit):
        if position in done_ids:
            continue
        stl_path = WORKING_DIR / f"v2_wing_{position}.stl"
        wing_path = WORKING_DIR / f"v2_wing_{position}.json"
        if not (stl_path.exists() and wing_path.exists()):
            continue
        row_index = indices[position]
        try:
            wing = json.loads(wing_path.read_text(encoding="utf-8"))
            generated_ar, geometry = generated_aspect_ratio(wing)
            target_ar = float(ar_target[row_index, 0])
            valid = ac_mask[row_index]
            ac_target = ac[row_index][valid]
            ret = cal_ac_hybrid_loss(surrogate, stl_path, root_chord=wing["init"]["chord_root"],
                                     area=geometry["area"], mac=geometry["mac_length"], ac_target=ac_target)
            for j in range(ret["working_condition"].shape[0]):
                rows.append({
                    "id": position, "wing_id": wing_ids[row_index],
                    "aoa": float(ret["working_condition"][j, 0]),
                    "mach": float(ret["working_condition"][j, 1]),
                    "target_ar": target_ar, "generated_ar": generated_ar,
                    "ar_rel_error": (generated_ar - target_ar) / target_ar,
                    "target_cl": float(ac_target[j, 3]), "target_cd": float(ac_target[j, 4]),
                    "target_cm": float(ac_target[j, 5]),
                    "cd_loss_surr": ret["surrogate"]["cd"][j],
                    "cm_loss_surr": ret["surrogate"]["cm"][j],
                    "cd_loss_intg": ret["integral"]["cd"][j],
                    "cm_loss_intg": ret["integral"]["cm"][j],
                    "cl_loss_surr": ret["surrogate"]["cl"][j],
                    "cl_loss_intg": ret["integral"]["cl"][j],
                })
        except Exception:
            record_failure("evaluate", position, traceback.format_exc(limit=2))
            print(f"[evaluate] wing {position} FAILED (see v2_failures.json)")
            continue
        finally:
            if rows:
                frame = pd.DataFrame(rows, columns=COLUMNS)
                existing = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else None
                merged = pd.concat([existing, frame], ignore_index=True) if existing is not None else frame
                merged.to_csv(RESULTS_CSV, index=False)
                rows.clear()
        if (position + 1) % 10 == 0:
            print(f"[evaluate] {position + 1}/{limit}")

    if RESULTS_CSV.exists():
        frame = pd.read_csv(RESULTS_CSV)
        print("\n===== V2 FM test summary =====")
        print(f"wings evaluated: {frame['id'].nunique()}, condition rows: {len(frame)}")
        for column in ("ar_rel_error", "cd_loss_surr", "cm_loss_surr",
                       "cd_loss_intg", "cm_loss_intg", "cl_loss_surr", "cl_loss_intg"):
            if column not in frame.columns:
                continue
            values = frame[column].abs().dropna()
            print(f"|{column}|: mean={values.mean():.4f}  median={values.median():.4f}  "
                  f"p90={values.quantile(0.9):.4f}")
    else:
        print("[evaluate] no results produced")


def stage_cl(args, dataset: dict, indices: list[int]) -> None:
    """Back-fill CL errors for results produced before CL was measured.

    CL is the interpolation anchor of calculate_aerodynamic_error, so CD/CM
    errors say nothing about whether the wing reaches the target lift at all.
    This stage re-uses the existing STLs and only evaluates the working
    (aoa, mach) point: one batched surrogate call plus one flow field per
    condition, roughly 1/40 of a full evaluate pass.
    """
    from AeroTransformerPredict import AeroTransformer, integrate_wing_aerodynamics

    conditions = dataset["conditions"]
    wing_ids = dataset["wing_ids"]
    ac, ac_mask = conditions["ac"], conditions["ac_mask"]

    done_ids: set[int] = set()
    if CL_CSV.exists():
        done_ids = set(pd.read_csv(CL_CSV)["id"].astype(int).tolist())
        print(f"[cl] resuming: {len(done_ids)} wings already in {CL_CSV.name}")

    surrogate = AeroTransformer()
    limit = args.limit if args.limit else len(indices)
    rows, failures = [], 0
    for position in range(limit):
        if position in done_ids:
            continue
        stl_path = WORKING_DIR / f"v2_wing_{position}.stl"
        wing_path = WORKING_DIR / f"v2_wing_{position}.json"
        if not (stl_path.exists() and wing_path.exists()):
            continue
        row_index = indices[position]
        try:
            wing = json.loads(wing_path.read_text(encoding="utf-8"))
            geometry = calculate_wing_mac(deepcopy(wing))
            ac_target = ac[row_index][ac_mask[row_index]].numpy()
            angle = np.degrees(np.arctan2(ac_target[:, 0], ac_target[:, 1]))
            mach = ac_target[:, 2]

            surrogate.build_grid(str(stl_path), root_chord=wing["init"]["chord_root"],
                                 area=geometry["area"], mac=geometry["mac_length"])
            s_ref = geometry["area"] / (wing["init"]["chord_root"] ** 2)
            c_ref = geometry["mac_length"] / wing["init"]["chord_root"]

            predicted = surrogate.predict_coeffs(angle, mach)
            for j in range(len(angle)):
                flowfield, origeom, _ = surrogate.predict_flow_field(aoa=float(angle[j]),
                                                                     mach=float(mach[j]))
                inp = torch.tensor(origeom, dtype=torch.float32).unsqueeze(0)
                outputs = torch.tensor(flowfield, dtype=torch.float32)
                res = integrate_wing_aerodynamics(inp, outputs, s_ref=s_ref, c_ref=c_ref,
                                                  x_ref=0, y_ref=0, alpha_deg=float(angle[j]),
                                                  include_friction=False)
                target_cl = float(ac_target[j, 3])
                rows.append({
                    "id": position, "wing_id": wing_ids[row_index],
                    "aoa": float(angle[j]), "mach": float(mach[j]),
                    "target_cl": target_cl,
                    "pred_cl_surr": float(predicted[j, 0]),
                    "pred_cl_intg": float(res["CL"]),
                    "cl_loss_surr": float(predicted[j, 0]) - target_cl,
                    "cl_loss_intg": float(res["CL"]) - target_cl,
                })
        except Exception:
            failures += 1
            record_failure("cl", position, traceback.format_exc(limit=2))
            print(f"[cl] wing {position} FAILED (see v2_failures.json)")
            continue
        finally:
            if rows:
                existing = pd.read_csv(CL_CSV) if CL_CSV.exists() else None
                frame = pd.DataFrame(rows)
                merged = pd.concat([existing, frame], ignore_index=True) if existing is not None else frame
                merged.to_csv(CL_CSV, index=False)
                rows.clear()
        if (position + 1) % 25 == 0:
            print(f"[cl] {position + 1}/{limit}")

    print(f"[cl] done ({failures} failures); results in {CL_CSV.name}")
    if RESULTS_CSV.exists() and CL_CSV.exists():
        main_frame, cl_frame = pd.read_csv(RESULTS_CSV), pd.read_csv(CL_CSV)
        main_frame = main_frame.drop(columns=[c for c in ("cl_loss_surr", "cl_loss_intg")
                                              if c in main_frame.columns])
        key = lambda frame: frame["id"].astype(str) + "|" + frame["aoa"].round(4).astype(str)
        main_frame["_k"], cl_frame["_k"] = key(main_frame), key(cl_frame)
        joined = main_frame.merge(cl_frame[["_k", "cl_loss_surr", "cl_loss_intg"]], on="_k", how="left")
        coverage = joined["cl_loss_surr"].notna().mean()
        joined.drop(columns=["_k"]).to_csv(RESULTS_CSV, index=False)
        print(f"[cl] merged CL columns into {RESULTS_CSV.name} (coverage {coverage*100:.1f}% of rows)")


# -------------------------------------------------------------------- main

def main() -> None:
    global WORKING_DIR, SAMPLES_PATH, FAILURES_PATH, RESULTS_CSV, CL_CSV, VSP_SCRIPT
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "sample", "reconstruct", "stl", "evaluate", "cl"),
                        default="all")
    parser.add_argument("--exp_dir", "-e", type=Path, default=V2_DIR / "experiments/fm_geometry")
    parser.add_argument("--ckpt", type=str, default="best",
                        help="'best', 'last', or an explicit checkpoint path.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/geometry_dataset.pt")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N test wings (per stage).")
    parser.add_argument("--steps", type=int, default=100,
                        help="Euler integration steps for FM sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=62)
    args = parser.parse_args()

    exp_dir = args.exp_dir.expanduser().resolve()
    WORKING_DIR = exp_dir / "test_results"
    SAMPLES_PATH = WORKING_DIR / "v2_samples.pt"
    FAILURES_PATH = WORKING_DIR / "v2_failures.json"
    RESULTS_CSV = WORKING_DIR / "v2_aero_test_results.csv"
    CL_CSV = WORKING_DIR / "v2_cl_error.csv"
    VSP_SCRIPT = WORKING_DIR / "script_v2.vspscript"
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_v2_dataset(args.dataset)
    indices = test_indices(dataset)
    print(f"test split: {len(indices)} wings (limit={args.limit or 'all'}), device={args.device}")

    stages = ("sample", "reconstruct", "stl", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n========== stage: {stage} ==========")
        {"sample": stage_sample, "reconstruct": stage_reconstruct,
         "stl": stage_stl, "evaluate": stage_evaluate, "cl": stage_cl}[stage](args, dataset, indices)


if __name__ == "__main__":
    main()
