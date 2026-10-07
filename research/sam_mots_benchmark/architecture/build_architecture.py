#!/usr/bin/env python3
"""Architecture page: official SAM 3 / SAM 3.1 figures (facebookresearch/sam3) + our redraw with the identity-recovery module.

    python architecture/build_architecture.py   ->   architecture/sam3_identity_recovery.html
"""
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent


def b64(p):
    return base64.b64encode((HERE / "official" / p).read_bytes()).decode()


SVG = """
<svg viewBox="0 0 1200 640" role="img" aria-labelledby="svgt svgd" class="arch">
<title id="svgt">SAM 3 video pipeline with the proposed identity-recovery module</title>
<desc id="svgd">Text and image encoders feed the detector and tracker; the association step merges detections with tracked masklets.
Our identity-recovery module and lost-track gallery sit on the branch where an unmatched detection would receive a new ID.</desc>
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahf"/></marker>
  <marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahpf"/></marker>
</defs>

<!-- inputs -->
<text x="20" y="100" class="lbl">“pedestrian”</text>
<text x="20" y="350" class="lbl">frame t</text>
<path d="M112,95 H146" class="ln" marker-end="url(#ah)"/>
<path d="M80,345 H146" class="ln" marker-end="url(#ah)"/>

<!-- encoders (from PE) -->
<polygon points="150,60 250,76 250,114 150,130" class="pe"/>
<text x="198" y="92" class="bx">Text</text><text x="198" y="110" class="bx">Encoder</text>
<polygon points="150,300 250,316 250,374 150,390" class="pe"/>
<text x="198" y="340" class="bx">Image</text><text x="198" y="358" class="bx">Encoder</text>

<path d="M250,95 H296" class="ln" marker-end="url(#ah)"/>
<path d="M250,345 H296" class="ln" marker-end="url(#ah)"/>
<path d="M272,345 V112 H296" class="ln" marker-end="url(#ah)"/>

<!-- detector (new in SAM 3) and tracker (from SAM 2) -->
<rect x="300" y="68" width="140" height="54" rx="8" class="s3"/>
<text x="370" y="92" class="bx">Detector</text><text x="370" y="110" class="sm">DETR + presence head</text>
<rect x="300" y="318" width="140" height="54" rx="8" class="s2"/>
<text x="370" y="342" class="bx">Tracker</text><text x="370" y="360" class="sm">memory attention</text>

<path d="M440,95 H486" class="ln" marker-end="url(#ah)"/>
<path d="M440,345 H486" class="ln" marker-end="url(#ah)"/>
<rect x="490" y="60" width="160" height="70" rx="6" class="nt"/>
<text x="570" y="90" class="bx">detections, frame t</text><text x="570" y="110" class="sm">mask + score ≥ 0.5</text>
<rect x="490" y="310" width="160" height="70" rx="6" class="nt"/>
<text x="570" y="340" class="bx">masklets, frame t</text><text x="570" y="360" class="sm">propagated from t − 1</text>

<!-- memory bank (from SAM 2) -->
<rect x="508" y="478" width="150" height="64" rx="6" class="s2 back"/>
<rect x="498" y="488" width="150" height="64" rx="6" class="s2"/>
<text x="573" y="516" class="bx">Memory Bank</text><text x="573" y="534" class="sm">7 frames + object pointers</text>
<path d="M498,520 H370 V376" class="ln" marker-end="url(#ah)"/>

<!-- association: the ⊕ of the official figure (new in SAM 3) -->
<text x="690" y="200" class="note e">= ⊕ in the official</text><text x="690" y="215" class="note e">SAM 3 figure</text>
<rect x="700" y="180" width="160" height="100" rx="8" class="s3"/>
<text x="780" y="210" class="bx">Association</text>
<text x="780" y="230" class="sm">Hungarian matching of</text><text x="780" y="246" class="sm">detections to masklets</text>
<text x="780" y="262" class="sm">on mask IoU ≥ 0.1</text>
<path d="M650,95 H760 V176" class="ln" marker-end="url(#ah)"/>
<path d="M650,345 H760 V284" class="ln" marker-end="url(#ah)"/>
<path d="M720,282 V440 H583 V474" class="ln" marker-end="url(#ah)"/>
<text x="652" y="432" class="sm">memory encoder</text>

<!-- output -->
<rect x="1060" y="180" width="125" height="70" rx="6" class="nt"/>
<text x="1122" y="210" class="bx">output, frame t</text><text x="1122" y="230" class="sm">(ID, mask, score)</text>
<path d="M860,205 H1056" class="ln" marker-end="url(#ah)"/>
<text x="958" y="197" class="sm">matched → keep ID</text>

<!-- ours: the branch where SAM 3 would create a new ID -->
<path d="M860,255 H960 V316" class="lnp" marker-end="url(#ahp)"/>
<text x="968" y="282" class="smp s">unmatched detection,</text><text x="968" y="297" class="smp s">score ≥ 0.7</text>
<text x="875" y="314" class="tag">OURS</text>
<rect x="875" y="320" width="170" height="86" rx="8" class="us"/>
<text x="960" y="344" class="bx">Identity recovery</text>
<text x="960" y="362" class="sm">motion gate (Kalman)</text><text x="960" y="378" class="sm">appearance similarity</text>
<text x="960" y="394" class="sm">accept if score ≥ τ</text>
<path d="M1045,350 H1122 V254" class="lnp" marker-end="url(#ahp)"/>
<text x="1130" y="300" class="smp s">old ID if</text><text x="1130" y="315" class="smp s">recovered,</text><text x="1130" y="330" class="smp s">else new ID</text>

<path d="M820,282 V520 H871" class="lnp" marker-end="url(#ahp)"/>
<text x="828" y="428" class="smp s">masklet unmatched,</text><text x="828" y="444" class="smp s">keep-alive expires</text>
<rect x="875" y="470" width="170" height="96" rx="8" class="us"/>
<text x="960" y="494" class="bx">Lost-track gallery</text>
<text x="960" y="512" class="sm">last mask and box</text><text x="960" y="528" class="sm">velocity (Kalman)</text>
<text x="960" y="544" class="sm">appearance embedding</text><text x="960" y="560" class="sm">kept for T seconds</text>
<path d="M960,468 V410" class="lnp" marker-end="url(#ahp)"/>
<text x="968" y="444" class="smp s">candidates</text>

<!-- appearance source: image features pooled under each mask -->
<path d="M198,390 V612 H960 V570" class="lnp" marker-end="url(#ahp)"/>
<text x="580" y="604" class="smp">image features pooled under each mask → appearance embedding</text>

<!-- level 2: restore memory -->
<path d="M875,548 H652" class="lnp dash" marker-end="url(#ahp)"/>
<text x="763" y="566" class="smp">level 2: restore the ID's memory</text>

<!-- legend -->
<g class="legend" transform="translate(22,452)">
  <rect x="0" y="0" width="14" height="14" rx="3" class="pe"/><text x="22" y="12">from PE</text>
  <rect x="0" y="24" width="14" height="14" rx="3" class="s2"/><text x="22" y="36">from SAM 2</text>
  <rect x="0" y="48" width="14" height="14" rx="3" class="s3"/><text x="22" y="60">new in SAM 3</text>
  <rect x="0" y="72" width="14" height="14" rx="3" class="us"/><text x="22" y="84">ours, level 1</text>
  <path d="M0,103 H14" class="lnp dash"/><text x="22" y="108">ours, level 2</text>
</g>
</svg>
"""

PAGE = """<title>SAM 3 Identity Recovery</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500&display=swap">
<style>
/* Layout: one column; official figures first, then our redraw at full width, then a compact change table. */
:root {
  --bg: #f5f6f7; --paper: #ffffff; --ink: #1c2025; --muted: #5a626c; --rule: #dcdfe4; --head: #eef0f3; --line: #7d8894;
  --pe-f: #d3efeb; --pe-s: #3a9e91; --s2-f: #d9e9f8; --s2-s: #4a8fd1; --s3-f: #fff2bf; --s3-s: #d9aa00;
  --us-f: #f3def4; --us-s: #a1339f; --us-ink: #8a2488;
  --serif: "IBM Plex Serif", Georgia, serif; --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #14171b; --paper: #1c2025; --ink: #e7e9ec; --muted: #9aa2ac; --rule: #333941; --head: #242930; --line: #8f9aa6;
  --pe-f: #173f3a; --pe-s: #5cc4b5; --s2-f: #1a3150; --s2-s: #6aa6ef; --s3-f: #3f3510; --s3-s: #e6c14a;
  --us-f: #3d1b3d; --us-s: #d77ad5; --us-ink: #e39be1; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #14171b; --paper: #1c2025; --ink: #e7e9ec; --muted: #9aa2ac; --rule: #333941; --head: #242930; --line: #8f9aa6;
  --pe-f: #173f3a; --pe-s: #5cc4b5; --s2-f: #1a3150; --s2-s: #6aa6ef; --s3-f: #3f3510; --s3-s: #e6c14a;
  --us-f: #3d1b3d; --us-s: #d77ad5; --us-ink: #e39be1; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 14.5px/1.55 var(--sans); }
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 34px 60px; display: grid; gap: 30px; }
header { display: grid; gap: 8px; }
.eyebrow { font: 500 11.5px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
h1 { font: 500 clamp(25px, 3.2vw, 34px)/1.15 var(--serif); margin: 0; text-wrap: balance; }
.lede { margin: 0; color: var(--muted); max-width: 90ch; }
section { display: grid; gap: 10px; min-width: 0; }
h2 { margin: 0; font: 600 17px/1.3 var(--sans); text-wrap: balance; }
h2 small { font: 400 13px var(--sans); color: var(--muted); margin-left: 6px; }
.figs { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 860px) { .figs { grid-template-columns: 1fr; } }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figure img { width: 100%; background: #fff; border: 1px solid var(--rule); border-radius: 5px; padding: 8px; box-sizing: border-box; }
figcaption { font-size: 12.5px; color: var(--muted); }
figcaption b { color: var(--ink); font-weight: 600; }
.canvas { overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; }
svg.arch { display: block; width: 100%; min-width: 860px; height: auto; font-family: var(--sans); }
.pe { fill: var(--pe-f); stroke: var(--pe-s); stroke-width: 2; }
.s2 { fill: var(--s2-f); stroke: var(--s2-s); stroke-width: 2; }
.s2.back { opacity: .55; stroke-dasharray: 4 3; }
.s3 { fill: var(--s3-f); stroke: var(--s3-s); stroke-width: 2; }
.us { fill: var(--us-f); stroke: var(--us-s); stroke-width: 2.2; }
.nt { fill: var(--paper); stroke: var(--line); stroke-width: 1.4; }
.ln { fill: none; stroke: var(--line); stroke-width: 1.8; }
.lnp { fill: none; stroke: var(--us-s); stroke-width: 2; }
.dash { stroke-dasharray: 6 4; }
.ahf { fill: var(--line); } .ahpf { fill: var(--us-s); }
text { fill: var(--ink); }
.lbl { font-size: 15px; }
.bx { font-size: 14px; font-weight: 600; text-anchor: middle; }
.sm { font-size: 11.5px; fill: var(--muted); text-anchor: middle; }
.smp { font-size: 11.5px; fill: var(--us-ink); text-anchor: middle; }
.smp.s { text-anchor: start; }
.note { font-size: 11.5px; font-style: italic; fill: var(--muted); text-anchor: middle; }
.note.e { text-anchor: end; }
.tag { font: 600 10.5px var(--mono); letter-spacing: .08em; fill: var(--us-ink); }
.legend text { font-size: 12px; fill: var(--muted); }
.cap { margin: 0; color: var(--muted); font-size: 13px; max-width: 100ch; }
.cap b { color: var(--ink); font-weight: 600; }
.box { overflow-x: auto; background: var(--paper); border: 1px solid var(--rule); border-radius: 5px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
thead th { background: var(--head); font: 500 11.5px/1.3 var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
tbody th { font-weight: 600; width: 17%; }
td.us { background: var(--us-f); fill: none; stroke: none; }
code { font: 12px var(--mono); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px 26px; }
.two div { display: grid; gap: 4px; min-width: 0; }
.two h3 { margin: 0; font: 600 14.5px var(--sans); }
.two p { margin: 0; color: var(--muted); font-size: 13.5px; }
@media print {
  @page { size: A4 landscape; margin: 10mm; }
  :root, :root[data-theme="dark"] { --bg: #fff; --paper: #fff; --ink: #000; --muted: #444; --rule: #bbb; --head: #f0f0f0; --line: #6d7884;
    --pe-f: #d3efeb; --pe-s: #3a9e91; --s2-f: #d9e9f8; --s2-s: #4a8fd1; --s3-f: #fff2bf; --s3-s: #d9aa00;
    --us-f: #f3def4; --us-s: #a1339f; --us-ink: #8a2488; color-scheme: light; }
  body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .wrap { max-width: none; padding: 0; gap: 14px; }
  .canvas, .box { overflow: visible; } svg.arch { min-width: 0; }
  section, figure, tr { break-inside: avoid; }
  .pb { break-before: page; }
}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Occlusion study · proposed architecture · 30 Sep 2026</div>
  <h1>SAM 3 with identity recovery</h1>
  <p class="lede">The official SAM 3 and SAM 3.1 architectures as published by Meta, and where our module attaches. The diagnosis showed that
  identity is lost when a person reappears after an occlusion and is given a new ID. In SAM 3 that decision is made in one place: the step that
  merges new detections with the objects already being tracked.</p>
</header>

<section>
  <h2>Official architectures <small>figures from the SAM 3 paper and code release</small></h2>
  <div class="figs">
    <figure><img alt="Official SAM 3 architecture: text and image encoders, detector, tracker, memory bank, merge of detected and propagated masks" src="data:image/png;base64,__SAM3__">
    <figcaption><b>SAM 3.</b> A detector (new in SAM 3) finds all instances of the prompt in frame t; a tracker from SAM 2 propagates the masks of
    already-tracked objects from frame t − 1 using its memory bank; the two sets are merged (⊕). Source: <i>SAM 3: Segment Anything with Concepts</i>,
    arXiv:2511.16719; figure <code>assets/model_diagram.png</code> in facebookresearch/sam3.</figcaption></figure>
    <figure><img alt="Official SAM 3.1 Object Multiplex tracker: per-object masks multiplexed into a shared memory space, then demultiplexed" src="data:image/png;base64,__SAM31__">
    <figcaption><b>SAM 3.1, Object Multiplex.</b> Replaces the per-object tracker: objects are grouped into fixed-capacity buckets (16 per bucket)
    that share memory encoding, memory bank, memory attention and mask decoding. Source: SAM 3 paper, Appendix H; figure
    <code>assets/sam3.1_diagram.png</code> in facebookresearch/sam3.</figcaption></figure>
  </div>
</section>

<section class="pb">
  <h2>Our redraw: SAM 3 plus identity recovery <small>same colour scheme as the official figure; the ⊕ is opened up</small></h2>
  <div class="canvas">__SVG__</div>
  <p class="cap"><b>Reading.</b> Everything in teal, blue and yellow is SAM 3 as released; numbers are the settings used in our benchmark runs.
  Association matches detections to tracked masklets on mask overlap only. A masklet that stays unmatched is deleted once its keep-alive runs out
  (up to 30 frames), and a detection that matches nothing and scores ≥ 0.7 becomes a new object with a new ID. Our module (purple) intercepts
  that branch and first asks whether the detection is someone we recently lost.</p>
</section>

<section>
  <h2>What changes</h2>
  <div class="box"><table>
    <thead><tr><th>Component</th><th>SAM 3 as released</th><th>With identity recovery</th></tr></thead>
    <tbody>
      <tr><th>Association cue</th><td>Mask overlap with a live masklet (IoU ≥ 0.1)</td><td class="us">Overlap, plus predicted position (motion) and appearance similarity</td></tr>
      <tr><th>Objects that disappear</th><td>Masklet and its memory deleted when keep-alive expires</td><td class="us">Kept in a lost-track gallery for T seconds: last mask and box, velocity, appearance embedding</td></tr>
      <tr><th>Unmatched detection</th><td>New ID if score ≥ 0.7</td><td class="us">Matched against the gallery first; new ID only if no candidate passes the motion gate and similarity ≥ τ</td></tr>
      <tr><th>Appearance source</th><td>Not used for identity</td><td class="us">Perception Encoder features averaged under the object's mask; no new network (a re-ID network is the fallback)</td></tr>
      <tr><th>Training</th><td>none needed for our use (zero-shot)</td><td class="us">none; T, τ and the gate are tuned on sequences that are not scored</td></tr>
      <tr><th>Where in code</th><td><code>Sam3VideoBase._associate_det_trk()</code></td><td class="us">Level 1: a causal layer on SAM's per-frame output. Level 2: a hook in the new-detection branch of <code>_associate_det_trk()</code></td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <h2>Two levels of integration</h2>
  <div class="two">
    <div><h3>Level 1: outside SAM (first experiment)</h3>
    <p>Runs frame by frame on SAM's outputs and only relabels IDs, using past frames only, so it works online. SAM's code and weights are untouched,
    and it can be evaluated on the tracker outputs we already have with the same TrackEval and occlusion metrics.</p></div>
    <div><h3>Level 2: inside SAM</h3>
    <p>The same decision placed in the new-detection branch of the association step. A recovered object also gets its saved memory back
    (dashed line), so the tracker continues from its history instead of starting from an empty memory. In SAM 3.1 this means writing back into a
    multiplexed memory bucket, which is harder; we target SAM 3 first.</p></div>
  </div>
  <p class="cap"><b>Success criteria.</b> Higher identity recovery R through occlusion (especially ≥ 2 s, where SAM 3 keeps 73%), higher AssRe and
  AssA, and no increase in merges (AssPr must not fall), measured with official TrackEval and the occlusion protocol used in the diagnosis.</p>
</section>
</div>
"""

page = PAGE.replace("__SAM3__", b64("model_diagram.png")).replace("__SAM31__", b64("sam3.1_diagram.png")).replace("__SVG__", SVG)
(HERE / "sam3_identity_recovery.html").write_text(page)
print("wrote", HERE / "sam3_identity_recovery.html", f"{len(page) / 1e6:.1f} MB")
