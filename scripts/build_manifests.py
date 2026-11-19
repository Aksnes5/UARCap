#!/usr/bin/env python3
"""Build frozen, nested VideoUFO corpus manifests.

Ordering is deterministic and source-balanced:
  * sources are ordered by sha256(source_id)
  * within a source, clips are ordered by sha256(clip_id)
  * within each topic, clips are taken round-robin across sources
Levels are prefixes of one fixed per-topic order, so they are strictly nested.
Topic counts are allocated proportionally with largest-remainder rounding.
"""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

CSV_PATH = "/home/bn/datasets/VideoUFO/metadata/VideoUFO.csv"
OUT = "/home/bn/videoufo_research/manifests"
LEVELS = [10000, 50000, 100000, 250000, 500000, None]  # None = FULL


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load():
    csv.field_size_limit(10 ** 9)
    rows = []  # (clip_id, source_id, topic)
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        header = next(r)
        idx_id = header.index("ID")
        idx_topic = header.index("Topic")
        for row in r:
            if not row:
                continue
            clip_id = row[idx_id]
            topic = row[idx_topic].strip().lower() or "unknown"
            source_id = clip_id.rsplit(".", 1)[0]
            rows.append((clip_id, source_id, topic))
    return rows


def build_orders(rows):
    """Per-topic deterministic source-balanced order."""
    by_topic = defaultdict(list)
    for clip_id, source_id, topic in rows:
        by_topic[topic].append((clip_id, source_id))
    orders = {}
    for topic, items in by_topic.items():
        by_source = defaultdict(list)
        for clip_id, source_id in items:
            by_source[source_id].append(clip_id)
        sources = sorted(by_source, key=h)
        for s in sources:
            by_source[s].sort(key=h)
        order = []
        max_len = max(len(by_source[s]) for s in sources)
        for k in range(max_len):
            for s in sources:
                if k < len(by_source[s]):
                    order.append((by_source[s][k], s))
        orders[topic] = order
        assert len(order) == len(items)
    return orders


def allocate(total, topic_sizes):
    n_total = sum(topic_sizes.values())
    if total is None:
        return dict(topic_sizes)
    raw = {t: total * n / n_total for t, n in topic_sizes.items()}
    alloc = {t: int(v) for t, v in raw.items()}
    rem = total - sum(alloc.values())
    order = sorted(raw, key=lambda t: (-(raw[t] - alloc[t]), t))
    for t in order[:rem]:
        alloc[t] += 1
    for t in alloc:
        alloc[t] = min(alloc[t], topic_sizes[t])
    # if capping removed some, redistribute deterministically
    while sum(alloc.values()) < total:
        for t in sorted(topic_sizes, key=lambda x: (-(topic_sizes[x] - alloc[x]), x)):
            if alloc[t] < topic_sizes[t]:
                alloc[t] += 1
                if sum(alloc.values()) == total:
                    break
    assert sum(alloc.values()) == total
    return alloc


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    print("clips", len(rows), "unique sources", len({r[1] for r in rows}))
    orders = build_orders(rows)
    topic_sizes = {t: len(o) for t, o in orders.items()}
    print("topics", len(topic_sizes))
    stats = {
        "source_csv": CSV_PATH,
        "total_clips": len(rows),
        "total_sources": len({r[1] for r in rows}),
        "levels": {},
    }
    previous = set()
    for level in LEVELS:
        name = "full" if level is None else f"{level // 1000}k"
        alloc = allocate(level, topic_sizes)
        selected = []
        for topic in sorted(orders):
            selected.extend(orders[topic][: alloc[topic]])
        ids = [c for c, _ in selected]
        assert len(ids) == (len(rows) if level is None else level), (name, len(ids))
        idset = set(ids)
        assert previous.issubset(idset), f"{name} is not nested"
        previous = idset
        path = os.path.join(OUT, f"clips_{name}.tsv")
        with open(path, "w") as f:
            f.write("clip_id\tsource_id\ttopic\n")
            for topic in sorted(orders):
                for clip_id, source_id in orders[topic][: alloc[topic]]:
                    f.write(f"{clip_id}\t{source_id}\t{topic}\n")
        srcs = {s for _, s in selected}
        hh = hashlib.sha256(("\n".join(ids)).encode()).hexdigest()
        stats["levels"][name] = {
            "clips": len(ids),
            "unique_sources": len(srcs),
            "avg_clips_per_source": round(len(ids) / max(1, len(srcs)), 4),
            "topic_counts": {t: alloc[t] for t in sorted(alloc) if alloc[t]},
            "sha256_clip_ids": hh,
        }
        print(name, len(ids), "sources", len(srcs), "sha256", hh[:16])
    with open(os.path.join(OUT, "manifest_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
