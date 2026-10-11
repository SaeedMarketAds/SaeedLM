import json
import sys
from datetime import datetime, timezone
from pathlib import Path

JOURNAL = Path(__file__).with_name("journal.jsonl")

KINDS = {
    "interaction",
    "knowledge",
    "correction",
    "benchmark",
    "training",
}


def record(kind, title, details):
    if kind not in KINDS:
        raise ValueError(f"نوع غير معروف: {kind}")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "title": title,
        "details": details,
    }

    with JOURNAL.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("تم تسجيل الحدث:", title)


def report():
    counts = {kind: 0 for kind in KINDS}

    if JOURNAL.exists():
        with JOURNAL.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    kind = entry.get("kind")
                    if kind in counts:
                        counts[kind] += 1
                except json.JSONDecodeError:
                    print("تحذير: يوجد سطر غير صالح في السجل.")

    print("=== سجل تعلّم SaeedLM ===")
    for kind, count in counts.items():
        print(f"{kind}: {count}")

    print("هذا التقرير يحصي السجلات فقط، ولا يثبت تحسن النموذج.")
    print("لا يُعدّ التدريب منجزًا إلا بوجود دليل تدريب واختبار فعلي.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "report":
        report()
    elif len(sys.argv) == 5 and sys.argv[1] == "add":
        record(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("عرض التقرير:")
        print("python -m learning.journal report")
        print()
        print("إضافة سجل:")
        print('python -m learning.journal add knowledge "العنوان" "التفاصيل"')
