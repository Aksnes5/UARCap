"""Lightweight zero-dependency local web annotation server (works on Windows & Ubuntu).

Features:
  - HTML5 muted video player (play/pause/seek/replay/mute toggle) + sampled 16-frame strip viewer.
  - Strict Phase-1 video-only core-event entry BEFORE revealing Phase-2 anonymous captions.
  - 1-click 4-class video truth buttons (Video-Supported, Video-Contradicted, Video-Unverifiable, Non-factual),
    plus add/edit/split/merge claims & notes.
  - Optional within-video normalized-claim reuse (strictly blind; never reveals which other caption shares it,
    with per-caption override allowed).
  - E3 dual-target independent checkboxes (Target A & Target B allowing both) + negation/hedged/partial/generic.
  - Atomic auto-save per annotator, instant resume, active timer, and one-click CSV/JSONL export.
"""

from __future__ import annotations

import csv
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..common.io_utils import atomic_write_json, atomic_write_jsonl, read_json

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>UARCap Q2 Blind Video Factuality Annotator</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background: #f4f6f9; color: #1b2631; }
  header { background: #1b4f72; color: #fff; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
  .wrap { max-width: 1280px; margin: 16px auto; padding: 0 16px; display: grid; grid-template-columns: 460px 1fr; gap: 18px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 14px; }
  video { width: 100%; border-radius: 6px; background: #000; max-height: 300px; }
  .frames { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 8px; }
  .frames img { width: 100%; border-radius: 3px; border: 1px solid #d5dbdb; }
  .btn { border: none; border-radius: 5px; padding: 7px 12px; cursor: pointer; font-size: 13px; margin: 2px; background: #e5e8e8; color: #1b2631; }
  .btn-primary { background: #1b4f72; color: #fff; }
  .btn-ok { background: #1e8449; color: #fff; }
  .btn-err { background: #b03a2e; color: #fff; }
  .btn-warn { background: #b7950b; color: #fff; }
  .btn-muted { background: #7f8c8d; color: #fff; }
  .btn-active { outline: 3px solid #1b4f72; font-weight: bold; }
  .claim-box { border: 1px solid #d5dbdb; border-radius: 6px; padding: 10px; margin: 8px 0; background: #fafbfc; }
  .cap-card { border-left: 4px solid #1b4f72; padding: 10px 12px; margin: 12px 0; background: #f8fbfe; border-radius: 4px; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; background: #d6eaf8; color: #1b4f72; }
  textarea, input[type=text] { width: 96%; padding: 7px; border: 1px solid #bdc3c7; border-radius: 4px; font-size: 13px; }
  .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .locked { opacity: 0.45; pointer-events: none; }
</style>
</head>
<body>
<header>
  <div><strong>UARCap Q2 Blind Factuality Annotator</strong> <span id="annBadge" class="pill"></span></div>
  <div class="toolbar">
    <span id="progressText">0 / 0</span>
    <span id="timerText">00:00</span>
    <button class="btn" onclick="exportFile('json')">Export JSON</button>
    <button class="btn" onclick="exportFile('jsonl')">Export JSONL</button>
    <button class="btn" onclick="exportFile('csv')">Export CSV</button>
  </div>
</header>
<div class="wrap">
  <div>
    <div class="card">
      <div class="toolbar" style="justify-content:space-between;margin-bottom:8px;">
        <strong id="taskTitle">Task</strong>
        <div>
          <button class="btn" onclick="nav(-1)">&larr; Prev</button>
          <button class="btn btn-primary" onclick="nav(1)">Next &rarr;</button>
        </div>
      </div>
      <video id="vidPlayer" controls muted playsinline></video>
      <div class="toolbar" style="margin-top:8px;">
        <button class="btn" onclick="replayVideo()">Replay</button>
        <button class="btn" onclick="toggleMute()">Toggle Mute</button>
        <button class="btn" onclick="toggleFrames()">Show/Hide Sampled Frames</button>
      </div>
      <div id="framesGrid" class="frames" style="display:none;"></div>
      <div class="toolbar" style="margin-top:8px;">
        <span style="font-size:12px;color:#566573;">16-frame visibility check:</span>
        <button id="fvSufficient" class="btn btn-ok" onclick="setFramesVisibility('sufficient')">Sufficient</button>
        <button id="fvInsufficient" class="btn btn-err" onclick="setFramesVisibility('insufficient')">Insufficient / misaligned</button>
      </div>
      <div style="margin-top:8px;">
        <textarea id="videoNotes" rows="2" placeholder="Optional video-level notes..." onchange="setVideoNotes(this.value)"></textarea>
      </div>
    </div>
    <div class="card">
      <strong>Phase 1: Independent Core Events (Video-Only)</strong>
      <p style="font-size:12px;color:#566573;">Watch the muted video first and write 1–2 main core events before unlocking anonymous captions.</p>
      <textarea id="coreEventsInput" rows="3" placeholder="One core event per line..."></textarea>
      <div style="margin-top:6px;">
        <button class="btn btn-primary" onclick="lockCoreEvents()">Confirm Core Events & Unlock Captions</button>
      </div>
    </div>
  </div>
  <div>
    <div id="workArea" class="card">Loading tasks...</div>
  </div>
</div>
<script>
let STATE = { bundle: null, saved: {}, idx: 0, secOnTask: 0, claimCache: {} };
const LABELS = [
  ['Video-Supported', 'btn-ok'],
  ['Video-Contradicted', 'btn-err'],
  ['Video-Unverifiable', 'btn-warn'],
  ['Non-factual', 'btn-muted']
];

async function init() {
  const res = await fetch('/api/state');
  const data = await res.json();
  STATE.bundle = data.bundle;
  STATE.saved = data.saved || {};
  document.getElementById('annBadge').textContent = 'Annotator: ' + STATE.bundle.annotator_id;
  // Resume at first incomplete task
  const firstUndone = STATE.bundle.tasks.findIndex((t, i) => !isTaskDone(t));
  STATE.idx = firstUndone >= 0 ? firstUndone : 0;
  render();
  setInterval(() => { STATE.secOnTask += 1; updateTimer(); }, 1000);
}

function taskKey(t) { return t.blind_video_id || t.blind_task_id; }

function isTaskDone(t) {
  const rec = STATE.saved[taskKey(t)];
  return Boolean(rec && rec.completed);
}

function updateTimer() {
  const m = String(Math.floor(STATE.secOnTask / 60)).padStart(2, '0');
  const s = String(STATE.secOnTask % 60).padStart(2, '0');
  document.getElementById('timerText').textContent = `${m}:${s}`;
}

function replayVideo() { const v = document.getElementById('vidPlayer'); v.currentTime = 0; v.play(); }
function toggleMute() { const v = document.getElementById('vidPlayer'); v.muted = !v.muted; }
function toggleFrames() {
  const g = document.getElementById('framesGrid');
  g.style.display = (g.style.display === 'none') ? 'grid' : 'none';
}

function setFramesVisibility(value) {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  rec.frames_visibility = value;
  saveCurrent(false);
  render();
}

function setVideoNotes(value) {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  rec.notes = value;
  saveCurrent(false);
}

function nav(delta) {
  saveCurrent(false);
  const n = STATE.bundle.tasks.length;
  STATE.idx = Math.max(0, Math.min(n - 1, STATE.idx + delta));
  STATE.secOnTask = 0;
  render();
}

function ensureRec(t) {
  const k = taskKey(t);
  if (!STATE.saved[k]) {
    STATE.saved[k] = {
      task_key: k, task_type: t.task_type, core_events: [], phase1_confirmed: (t.task_type !== 'E2_FACT_AUDIT'),
      frames_visibility: 'not_checked', notes: '',
      captions: {}, e3: { target_a_expressed: false, target_b_expressed: false, negated_a: false, negated_b: false, hedged: false, partial: false, generic_only: false, notes: '' },
      elapsed_sec: 0, completed: false
    };
  }
  return STATE.saved[k];
}

function lockCoreEvents() {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  const lines = document.getElementById('coreEventsInput').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (lines.length === 0) { alert('Please enter at least 1 core event before unlocking captions.'); return; }
  rec.core_events = lines;
  rec.phase1_confirmed = true;
  saveCurrent(false);
  render();
}

function setClaimLabel(bcid, cid, normKey, lbl) {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  if (!rec.captions[bcid]) rec.captions[bcid] = { claims: {}, added_claims: [], core_covered: false, notes: '' };
  rec.captions[bcid].claims[cid] = lbl;
  // Blind within-video normalized-claim reuse cache (never reveals which caption/arm had it)
  if (normKey) {
    const cacheKey = `${taskKey(t)}::${normKey}`;
    STATE.claimCache[cacheKey] = lbl;
  }
  saveCurrent(false);
  render();
}

function addMissingClaim(bcid) {
  const txt = prompt('Enter omitted factual claim from this caption:');
  if (!txt || !txt.trim()) return;
  const lbl = prompt('Enter label (Video-Supported / Video-Contradicted / Video-Unverifiable / Non-factual):', 'Video-Contradicted');
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  if (!rec.captions[bcid]) rec.captions[bcid] = { claims: {}, added_claims: [], claim_revisions: [], core_covered: false, notes: '' };
  rec.captions[bcid].added_claims.push({ text: txt.trim(), label: (lbl && lbl.trim()) || 'Video-Contradicted' });
  saveCurrent(false);
  render();
}

function splitClaim(bcid, cid) {
  const part1 = prompt('Split claim — Part 1 text:');
  if (!part1 || !part1.trim()) return;
  const lbl1 = prompt('Part 1 label (Video-Supported / Video-Contradicted / Video-Unverifiable / Non-factual):', 'Video-Supported');
  const part2 = prompt('Split claim — Part 2 text:');
  if (!part2 || !part2.trim()) return;
  const lbl2 = prompt('Part 2 label (Video-Supported / Video-Contradicted / Video-Unverifiable / Non-factual):', 'Video-Contradicted');
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  if (!rec.captions[bcid]) rec.captions[bcid] = { claims: {}, added_claims: [], claim_revisions: [], core_covered: false, notes: '' };
  if (!rec.captions[bcid].claim_revisions) rec.captions[bcid].claim_revisions = [];
  rec.captions[bcid].claim_revisions.push({
    revision_type: 'split',
    source_claim_ids: [cid],
    resulting_claims: [
      { claim_id: cid + '_S1', text: part1.trim(), label: (lbl1 && lbl1.trim()) || 'Video-Supported' },
      { claim_id: cid + '_S2', text: part2.trim(), label: (lbl2 && lbl2.trim()) || 'Video-Contradicted' }
    ]
  });
  saveCurrent(false);
  render();
}

function mergeClaims(bcid, claimIdsCsv) {
  const cids = prompt('Enter comma-separated claim IDs to merge:', claimIdsCsv);
  if (!cids || !cids.trim()) return;
  const mergedTxt = prompt('Enter merged claim text:');
  if (!mergedTxt || !mergedTxt.trim()) return;
  const lbl = prompt('Merged claim label (Video-Supported / Video-Contradicted / Video-Unverifiable / Non-factual):', 'Video-Supported');
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  if (!rec.captions[bcid]) rec.captions[bcid] = { claims: {}, added_claims: [], claim_revisions: [], core_covered: false, notes: '' };
  if (!rec.captions[bcid].claim_revisions) rec.captions[bcid].claim_revisions = [];
  const srcList = cids.split(',').map(x => x.trim()).filter(Boolean);
  rec.captions[bcid].claim_revisions.push({
    revision_type: 'merge',
    source_claim_ids: srcList,
    resulting_claims: [{ claim_id: srcList.join('+'), text: mergedTxt.trim(), label: (lbl && lbl.trim()) || 'Video-Supported' }]
  });
  saveCurrent(false);
  render();
}

async function saveCurrent(markComplete) {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  rec.elapsed_sec = (rec.elapsed_sec || 0) + STATE.secOnTask;
  STATE.secOnTask = 0;
  if (markComplete) rec.completed = true;
  await fetch('/api/save', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ task_key: taskKey(t), record: rec })
  });
  updateProgress();
}

function updateProgress() {
  const total = STATE.bundle.tasks.length;
  const done = STATE.bundle.tasks.filter(isTaskDone).length;
  document.getElementById('progressText').textContent = `Done: ${done} / ${total}`;
}

function render() {
  const t = STATE.bundle.tasks[STATE.idx];
  const rec = ensureRec(t);
  updateProgress();
  document.getElementById('taskTitle').textContent = `${t.task_type} — ${taskKey(t)} (${STATE.idx + 1}/${STATE.bundle.tasks.length})`;
  const v = document.getElementById('vidPlayer');
  v.src = '/media?path=' + encodeURIComponent(t.video_path);
  v.muted = true;
  const fg = document.getElementById('framesGrid');
  fg.innerHTML = (t.sampled_frame_paths || []).map(p => `<img src="/media?path=${encodeURIComponent(p)}" />`).join('');
  document.getElementById('coreEventsInput').value = (rec.core_events || []).join('\n');
  document.getElementById('videoNotes').value = rec.notes || '';
  document.getElementById('fvSufficient').classList.toggle('btn-active', rec.frames_visibility === 'sufficient');
  document.getElementById('fvInsufficient').classList.toggle('btn-active', rec.frames_visibility === 'insufficient');

  const work = document.getElementById('workArea');
  if (t.task_type === 'E2_FACT_AUDIT') {
    const cls = rec.phase1_confirmed ? '' : 'locked';
    let html = `<div class="${cls}"><h3>Phase 2: Blind Caption Fact Verification</h3>`;
    for (const c of t.captions) {
      const crec = rec.captions[c.blind_caption_id] || { claims: {}, added_claims: [], core_covered: false, notes: '' };
      html += `<div class="cap-card"><div><span class="pill">${c.blind_caption_id}</span> <strong>${c.caption}</strong></div>`;
      for (const cl of c.claims) {
        const cacheKey = `${taskKey(t)}::${cl.norm_key}`;
        const cur = crec.claims[cl.claim_id] || STATE.claimCache[cacheKey] || '';
        const reusedTag = (!crec.claims[cl.claim_id] && STATE.claimCache[cacheKey]) ? ' <span class="pill">reused suggestion</span>' : '';
        html += `<div class="claim-box"><div>${cl.text}${reusedTag}</div><div>`;
        for (const [lbl, bcls] of LABELS) {
          const act = (cur === lbl) ? 'btn-active' : '';
          html += `<button class="btn ${bcls} ${act}" onclick="setClaimLabel('${c.blind_caption_id}','${cl.claim_id}','${cl.norm_key}','${lbl}')">${lbl}</button>`;
        }
        html += `<button class="btn" onclick="splitClaim('${c.blind_caption_id}','${cl.claim_id}')">Split</button>`;
        html += `</div></div>`;
      }
      if (crec.added_claims && crec.added_claims.length) {
        html += `<div><em>Added claims:</em> ` + crec.added_claims.map(x => `${x.text} [${x.label}]`).join('; ') + `</div>`;
      }
      if (crec.claim_revisions && crec.claim_revisions.length) {
        html += `<div><em>Revisions (split/merge):</em> ` + crec.claim_revisions.map(r => `${r.revision_type}(${r.source_claim_ids.join(',')})`).join('; ') + `</div>`;
      }
      const allCids = c.claims.map(x => x.claim_id).join(',');
      html += `<div style="margin-top:6px;"><button class="btn" onclick="addMissingClaim('${c.blind_caption_id}')">+ Add Omitted Claim</button> <button class="btn" onclick="mergeClaims('${c.blind_caption_id}','${allCids}')">Merge Claims</button></div></div>`;
    }
    html += `<button class="btn btn-ok" onclick="saveCurrent(true); nav(1);">Mark Video Complete & Next &rarr;</button></div>`;
    work.innerHTML = html;
  } else {
    const e = rec.e3;
    work.innerHTML = `
      <h3>E3 Blind Dual-Target Verification (${t.blind_task_id})</h3>
      <div class="cap-card"><strong>Anonymous Caption:</strong><br/>${t.caption}</div>
      <div class="claim-box">
        <label><input type="checkbox" ${e.target_a_expressed ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.target_a_expressed=this.checked;saveCurrent(false);" /> <strong>Target A expressed:</strong> ${t.target_a}</label><br/>
        <label><input type="checkbox" ${e.negated_a ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.negated_a=this.checked;saveCurrent(false);" /> Target A explicitly negated</label>
      </div>
      <div class="claim-box">
        <label><input type="checkbox" ${e.target_b_expressed ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.target_b_expressed=this.checked;saveCurrent(false);" /> <strong>Target B expressed:</strong> ${t.target_b}</label><br/>
        <label><input type="checkbox" ${e.negated_b ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.negated_b=this.checked;saveCurrent(false);" /> Target B explicitly negated</label>
      </div>
      <div class="toolbar">
        <label><input type="checkbox" ${e.hedged ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.hedged=this.checked;saveCurrent(false);" /> Hedged/Uncertain</label>
        <label><input type="checkbox" ${e.partial ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.partial=this.checked;saveCurrent(false);" /> Partial Match</label>
        <label><input type="checkbox" ${e.generic_only ? 'checked' : ''} onchange="STATE.saved['${taskKey(t)}'].e3.generic_only=this.checked;saveCurrent(false);" /> Generic Hypernym Only</label>
      </div>
      <div style="margin-top:10px;"><button class="btn btn-ok" onclick="saveCurrent(true); nav(1);">Mark Complete & Next &rarr;</button></div>
    `;
  }
}

function exportFile(fmt) { window.location.href = '/api/export?fmt=' + fmt; }
init();
</script>
</body>
</html>
"""


class AnnotationStore:
    """Thread-safe, per-annotator isolated atomic persistence for blind annotations."""

    def __init__(self, bundle_path: Path | str, save_dir: Path | str) -> None:
        self.bundle_path = Path(bundle_path)
        self.bundle = read_json(self.bundle_path)
        self.annotator_id = str(self.bundle["annotator_id"])
        self.save_dir = Path(save_dir) / f"annotator_{self.annotator_id}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.save_dir / "progress.json"
        self.saved: dict[str, dict[str, Any]] = read_json(self.state_file) if self.state_file.is_file() else {}
        self.allowed_media: set[str] = set()
        for t in self.bundle.get("tasks", []):
            if t.get("video_path"):
                self.allowed_media.add(str(Path(t["video_path"]).resolve()))
            for fp in t.get("sampled_frame_paths", []):
                self.allowed_media.add(str(Path(fp).resolve()))

    def save_record(self, task_key: str, record: dict[str, Any]) -> None:
        self.saved[task_key] = record
        atomic_write_json(self.state_file, self.saved)
        atomic_write_jsonl(self.save_dir / "annotations.jsonl", list(self.saved.values()))

    def export_json_bytes(self) -> bytes:
        payload = {
            "annotator_id": self.annotator_id,
            "bundle_version": self.bundle.get("bundle_version", ""),
            "bundle_sha256": self.bundle.get("bundle_sha256", ""),
            "records": list(self.saved.values()),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    def export_jsonl_bytes(self) -> bytes:
        lines = [json.dumps(r, ensure_ascii=False) for r in self.saved.values()]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    def export_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["annotator_id", "task_key", "task_type", "completed", "elapsed_sec", "payload_json"])
        for k, rec in sorted(self.saved.items()):
            writer.writerow(
                [
                    self.annotator_id,
                    k,
                    rec.get("task_type", ""),
                    rec.get("completed", False),
                    rec.get("elapsed_sec", 0),
                    json.dumps(rec, ensure_ascii=False),
                ]
            )
        return buf.getvalue().encode("utf-8")


def create_annotation_server(
    bundle_path: Path | str,
    save_dir: Path | str,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[ThreadingHTTPServer, AnnotationStore]:
    store = AnnotationStore(bundle_path, save_dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return  # Keep terminal & test output clean

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                payload = json.dumps({"bundle": store.bundle, "saved": store.saved}, ensure_ascii=False).encode("utf-8")
                self._send(HTTPStatus.OK, payload, "application/json; charset=utf-8")
                return
            if parsed.path == "/api/export":
                fmt = parse_qs(parsed.query).get("fmt", ["jsonl"])[0]
                if fmt == "csv":
                    self._send(HTTPStatus.OK, store.export_csv_bytes(), "text/csv; charset=utf-8")
                elif fmt == "json":
                    self._send(HTTPStatus.OK, store.export_json_bytes(), "application/json; charset=utf-8")
                else:
                    self._send(HTTPStatus.OK, store.export_jsonl_bytes(), "application/x-ndjson; charset=utf-8")
                return
            if parsed.path == "/media":
                req_path = parse_qs(parsed.query).get("path", [""])[0]
                if not req_path:
                    self._send(HTTPStatus.BAD_REQUEST, b"missing path", "text/plain")
                    return
                resolved = str(Path(req_path).resolve())
                if resolved not in store.allowed_media or not Path(resolved).is_file():
                    self._send(HTTPStatus.FORBIDDEN, b"forbidden media path", "text/plain")
                    return
                ctype = mimetypes.guess_type(resolved)[0] or "application/octet-stream"
                self._send(HTTPStatus.OK, Path(resolved).read_bytes(), ctype)
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/save":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8"))
                store.save_record(str(data["task_key"]), dict(data["record"]))
                self._send(HTTPStatus.OK, b'{"ok":true}', "application/json")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    server = ThreadingHTTPServer((host, port), Handler)
    return server, store
