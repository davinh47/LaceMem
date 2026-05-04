import json
import argparse
import re
from pathlib import Path

def collect_dia_ids(conv_obj: dict, keep_sessions: set[str]) -> set[str]:
    dia_ids = set()
    for k, v in conv_obj.items():
        if k.startswith("session_") and k.endswith("_date_time"):
            continue
        if k.startswith("session_") and isinstance(v, list):
            if k in keep_sessions:
                for turn in v:
                    if "dia_id" in turn:
                        dia_ids.add(turn["dia_id"])
    return dia_ids

def evidence_ok(evidence_list, allowed_dia_ids: set[str]) -> bool:
    if not evidence_list:
        return True  # 你也可以改成 False，只保留有证据的题
    # evidence 里可能是 "D8:6; D9:17" 这种
    for item in evidence_list:
        for dia in re.findall(r"D\d+:\d+", str(item)):
            if dia not in allowed_dia_ids:
                return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sessions", required=True, help="comma sep, e.g. session_1,session_2")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    keep_sessions = set(s.strip() for s in args.sessions.split(",") if s.strip())

    data = json.loads(src.read_text(encoding="utf-8"))

    # LOCOMO files can be either a single dict sample or a list of samples.
    if isinstance(data, dict):
        samples = [data]
        input_is_list = False
    elif isinstance(data, list):
        samples = data
        input_is_list = True
    else:
        raise TypeError(f"Unsupported JSON top-level type: {type(data)}")

    mini_samples = []
    kept_total_qa = 0
    kept_total_allowed = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if "conversation" not in sample or "qa" not in sample:
            continue

        conv = sample["conversation"]
        allowed_dia_ids = collect_dia_ids(conv_obj=conv, keep_sessions=keep_sessions)

        # Skip samples that don't contain the requested sessions.
        if not allowed_dia_ids:
            continue

        # 裁 conversation：只保留选中的 sessions + speaker + date_time
        mini_conv = {}
        for k, v in conv.items():
            if k in ("speaker_a", "speaker_b"):
                mini_conv[k] = v
            elif k.endswith("_date_time") and k.replace("_date_time", "") in keep_sessions:
                mini_conv[k] = v
            elif k in keep_sessions:
                mini_conv[k] = v

        # 裁 qa：只保留 evidence 都落在选中 dia_id 里的题
        mini_qa = [q for q in sample["qa"] if evidence_ok(q.get("evidence", []), allowed_dia_ids)]

        mini = dict(sample)
        mini["conversation"] = mini_conv
        mini["qa"] = mini_qa

        mini_samples.append(mini)
        kept_total_qa += len(mini_qa)
        kept_total_allowed += len(allowed_dia_ids)

    # Write output in the same top-level shape as input
    if input_is_list:
        out_obj = mini_samples
    else:
        out_obj = mini_samples[0] if mini_samples else {}

    dst.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"kept sessions={sorted(keep_sessions)}")
    print(f"samples kept={len(mini_samples)} / {len(samples)}")
    print(f"total allowed dia_ids kept={kept_total_allowed}")
    print(f"total qa kept={kept_total_qa}")
    print(f"wrote: {dst}")

if __name__ == "__main__":
    main()