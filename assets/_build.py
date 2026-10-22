"""
Builds every SVG in this folder, in a light and a dark version that match GitHub's own colours.
Edit the settings below and run:  python3 _build.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

# ── venue colours ─────────────────────────────────────────────────────────────
# NEURIPS: set these to the official NeurIPS colour if it differs.
NEURIPS = {"dark": "#a371f7", "light": "#6f42c1"}
ML4H = {"dark": "#f778ba", "light": "#bf3989"}

# ── GitHub Primer palette ─────────────────────────────────────────────────────
GH = {
    "dark": dict(canvas="#0d1117", subtle="#161b22", border="#30363d", fg="#e6edf3", muted="#8b949e",
                 faint="#484f58", accent="#4493f8", accent_bg="#121d2f", danger="#f85149", grid="#21262d"),
    "light": dict(canvas="#ffffff", subtle="#f6f8fa", border="#d0d7de", fg="#1f2328", muted="#59636e",
                  faint="#afb8c1", accent="#0969da", accent_bg="#ddf4ff", danger="#cf222e", grid="#eaeef2"),
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ══════════════════════════════════════════════════════════════════════════════
# Header figure: a real (toy) simulation, not decoration.
# Outcomes are only observed under selection that depends on the outcome itself,
# so a naive kernel regression is biased; inverse-probability weights fix it.
# ══════════════════════════════════════════════════════════════════════════════
def simulate():
    rng = np.random.default_rng(7)
    n = 3000
    t = rng.uniform(0, 10, n)
    mu = lambda s: 0.6 + 3.0 * np.exp(-(((s - 5.2) / 2.5) ** 2))
    y = mu(t) + 0.75 * rng.standard_normal(n)
    logit = 0.3 + 5.0 * (y - mu(t)) * (t / 10)             # high-dose units are seen mostly when their outcome is high
    p = 1 / (1 + np.exp(-logit))
    seen = rng.uniform(size=n) < p
    grid = np.linspace(0.3, 9.7, 60)

    def nw(w):
        k = np.exp(-0.5 * ((grid[:, None] - t[seen][None, :]) / 0.7) ** 2) * w[None, :]
        return (k * y[seen][None, :]).sum(1) / k.sum(1)

    naive = nw(np.ones(seen.sum()))
    corrected = nw(1 / p[seen])
    pts = rng.choice(np.flatnonzero(seen), 110, replace=False)  # a readable subset of the observed points
    return dict(grid=grid, truth=mu(grid), naive=naive, corrected=corrected, pt=t[pts], py=y[pts])


SIM = simulate()


def figure(c, x0, y0, w, h):
    """matplotlib-ish axes, drawn by hand."""
    xmin, xmax, ymin, ymax = 0, 10, -0.5, 4.5
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * w
    Y = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
    path = lambda xs, ys: "M" + " L".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
    out = []
    for v in range(0, 5):  # horizontal grid + y ticks
        out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c["grid"]}"/>')
        out.append(f'<text x="{x0 - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" font-family="{MONO}" font-size="14" fill="{c["muted"]}">{v}</text>')
    for v in range(0, 11, 2):
        out.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y0 + h}" y2="{y0 + h + 4}" stroke="{c["muted"]}"/>')
        out.append(f'<text x="{X(v):.1f}" y="{y0 + h + 20}" text-anchor="middle" font-family="{MONO}" font-size="14" fill="{c["muted"]}">{v}</text>')
    out.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{c["muted"]}"/>')
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{y0}" y2="{y0 + h}" stroke="{c["muted"]}"/>')
    out.append(f'<text x="{x0 + w}" y="{y0 + h + 40}" text-anchor="end" font-family="{SANS}" font-size="15" fill="{c["muted"]}">dose  t</text>')
    out.append(f'<text x="{x0 - 34}" y="{y0 + h / 2}" text-anchor="middle" font-family="{SANS}" font-size="15" font-style="italic" fill="{c["muted"]}" transform="rotate(-90 {x0 - 34} {y0 + h / 2})">outcome</text>')
    dots = "".join(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.4"/>' for a, b in zip(SIM["pt"], SIM["py"]) if ymin < b < ymax)
    out.append(f'<g class="pts" fill="{c["faint"]}">{dots}</g>')
    g = SIM["grid"]
    out.append(f'<path class="ln l0" d="{path(g, SIM["truth"])}" fill="none" stroke="{c["fg"]}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    out.append(f'<path class="ln l1" pathLength="1" d="{path(g, SIM["naive"])}" fill="none" stroke="{c["danger"]}" stroke-width="2"/>')
    out.append(f'<path class="ln l2" pathLength="1" d="{path(g, SIM["corrected"])}" fill="none" stroke="{c["accent"]}" stroke-width="2.6"/>')
    # legend
    lx, ly = x0 + 14, y0 + 6
    out.append(f'<rect x="{lx - 8}" y="{ly - 4}" width="212" height="80" rx="6" fill="{c["canvas"]}" fill-opacity="0.9" stroke="{c["border"]}"/>')
    for i, (lab, col, dash, sw) in enumerate([("true dose–response", c["fg"], "5 4", 1.6),
                                              ("naive estimate", c["danger"], "", 2),
                                              ("reweighted estimate", c["accent"], "", 2.6)]):
        yy = ly + 14 + i * 23
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="{sw}"{da}/>')
        out.append(f'<text x="{lx + 30}" y="{yy + 4}" font-family="{SANS}" font-size="15" fill="{c["fg"]}">{lab}</text>')
    return "\n    ".join(out)


def header(theme):
    c = GH[theme]
    W, H = 1200, 320
    topics = ["causal-inference", "explainable-ai", "ml-for-healthcare"]
    chips, x = [], 48
    for tp in topics:
        w = 24 + 8.4 * len(tp)
        chips.append(f'<rect x="{x:.0f}" y="190" width="{w:.0f}" height="30" rx="15" fill="{c["accent_bg"]}"/>'
                     f'<text x="{x + w / 2:.0f}" y="210" text-anchor="middle" font-family="{SANS}" font-size="15" font-weight="500" fill="{c["accent"]}">{tp}</text>')
        x += w + 8
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Alessandro Marchese, PhD researcher in causal machine learning and explainable AI">
  <style>
    .ln {{ stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw 1.8s ease-out forwards; }}
    .l0 {{ stroke-dasharray: 5 4; stroke-dashoffset: 0; animation: none; }}
    .l1 {{ animation-delay: .3s; }} .l2 {{ animation-delay: 1.1s; }}
    .pts {{ opacity: 0; animation: fade .8s ease-out forwards; }}
    @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
    @keyframes fade {{ to {{ opacity: 1; }} }}
  </style>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="{c["canvas"]}" stroke="{c["border"]}"/>
  <text x="48" y="72" font-family="{MONO}" font-size="16" fill="{c["muted"]}">AlessandroMarchese / README.md</text>
  <text x="46" y="128" font-family="{SANS}" font-size="44" font-weight="600" fill="{c["fg"]}" letter-spacing="-0.5">Alessandro Marchese</text>
  <text x="48" y="164" font-family="{SANS}" font-size="21" fill="{c["muted"]}">PhD researcher · causal ML &amp; explainable AI</text>
  {''.join(chips)}
  <text x="48" y="256" font-family="{SANS}" font-size="15" fill="{c["muted"]}">VUB × UHasselt</text>
  <text x="48" y="280" font-family="{SANS}" font-size="15" fill="{c["muted"]}">previously KU Leuven · King’s College London</text>
  <line x1="590" x2="590" y1="40" y2="280" stroke="{c["border"]}"/>
  <g>
    {figure(c, 680, 40, 472, 200)}
  </g>
  <text x="1152" y="304" text-anchor="end" font-family="{SANS}" font-size="14" fill="{c["muted"]}">Toy simulation · outcome-dependent selection biases the naive fit; reweighting recovers it</text>
</svg>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Research cards — GitHub "box" look, venue colour on the left edge and label.
# ══════════════════════════════════════════════════════════════════════════════
def card(theme, venue, vcol, title, subtitle, desc, motif, label):
    c = GH[theme]
    W, H = 900, 180
    lw = 16 + 7.6 * len(venue)
    lines = "".join(f'<text x="34" y="{124 + i * 21}" font-family="{SANS}" font-size="14" fill="{c["muted"]}">{esc(l)}</text>'
                    for i, l in enumerate(desc))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{esc(label)}">
  <defs><clipPath id="k"><rect width="{W}" height="{H}" rx="6"/></clipPath></defs>
  <g clip-path="url(#k)">
    <rect width="{W}" height="{H}" fill="{c["canvas"]}"/>
    <rect width="4" height="{H}" fill="{vcol}"/>
  </g>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="none" stroke="{c["border"]}"/>
  <rect x="34.5" y="24.5" width="{lw:.0f}" height="22" rx="11" fill="none" stroke="{vcol}"/>
  <text x="{34.5 + lw / 2:.0f}" y="40" text-anchor="middle" font-family="{SANS}" font-size="12" font-weight="600" fill="{vcol}">{venue}</text>
  <text x="34" y="88" font-family="{SANS}" font-size="24" font-weight="600" fill="{c["fg"]}">{esc(title)}<tspan font-weight="400" font-size="17" fill="{c["muted"]}">{esc(subtitle)}</tspan></text>
  {lines}
  <line x1="590" x2="590" y1="24" y2="156" stroke="{c["border"]}"/>
  {motif(c, vcol)}
</svg>
"""


def m_recourse(c, v):
    """Counterfactual explanations: rejected points take the minimal move across a linear decision boundary."""
    rng = np.random.default_rng(5)
    ox, oy, w, h = 620, 30, 250, 124
    A, B = np.array([ox + 70.0, oy]), np.array([ox + w, oy + h - 20.0])      # boundary endpoints (screen space)
    d = (B - A) / np.linalg.norm(B - A)
    nrm = np.array([d[1], -d[0]])                                              # unit normal pointing to the accepted side
    P = np.column_stack([rng.uniform(ox + 8, ox + w - 8, 34), rng.uniform(oy + 8, oy + h - 8, 34)])
    s = [f'<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{v}"/></marker></defs>',
         f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{c["border"]}"/>',
         f'<line x1="{A[0]:.1f}" y1="{A[1]:.1f}" x2="{B[0]:.1f}" y2="{B[1]:.1f}" stroke="{c["muted"]}" stroke-width="1.3" stroke-dasharray="4 3"/>']
    L = np.linalg.norm(B - A)
    picked = [(A + t * d + off * nrm, off) for t, off in [(0.22 * L, -34), (0.42 * L, -22), (0.62 * L, -40), (0.82 * L, -26)]]
    for p in P:
        dist = (p - A) @ nrm                      # >0 accepted, <0 rejected
        if abs(dist) < 8 or any(np.linalg.norm(p - q) < 16 for q, _ in picked):
            continue
        if dist > 0:
            s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["faint"]}"/>')
        else:
            s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["canvas"]}" stroke="{c["fg"]}" stroke-width="1.2"/>')
    for i, (p, dist) in enumerate(picked):
        q = p + (-dist + 8) * nrm
        s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["canvas"]}" stroke="{c["fg"]}" stroke-width="1.2"/>')
        s.append(f'<g opacity="0"><animate attributeName="opacity" to="1" dur=".4s" begin="{0.8 + 0.3 * i:.1f}s" fill="freeze"/>'
                 f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="{v}" stroke-width="1.5" marker-end="url(#ah)"/>'
                 f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3" fill="{v}"/></g>')
    s.append(f'<text x="{ox}" y="{oy - 7}" font-family="{SANS}" font-size="10" fill="{c["muted"]}">rejected</text>')
    s.append(f'<text x="{ox + w}" y="{oy - 7}" text-anchor="end" font-family="{SANS}" font-size="10" fill="{c["muted"]}">accepted</text>')
    s.append(f'<text x="{ox + w}" y="{oy + h + 16}" text-anchor="end" font-family="{SANS}" font-size="11" fill="{c["muted"]}">x  →  counterfactual  x′  (recourse)</text>')
    return "\n  ".join(s)


def m_clexnet(c, v):
    """Schematic in the style of a paper's Figure 1."""
    f = SANS
    def box(x, y, w, h, t, strong=False, it=False):
        st = v if strong else c["border"]
        fill = c["subtle"]
        style = ' font-style="italic"' if it else ""
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{st}" stroke-width="{1.6 if strong else 1}"/>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" font-family="{f}" font-size="12"{style} fill="{c["fg"]}">{t}</text>')
    arrow = lambda x1, y1, x2, y2, col, dash="": (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.4" marker-end="url(#a2)"'
                                                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    s = [f'<defs><marker id="a2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{c["muted"]}"/></marker></defs>']
    s.append(box(612, 64, 78, 36, "offer x"))
    s.append(box(724, 64, 62, 36, "φ(x)", strong=True, it=True))
    s.append(box(820, 64, 60, 36, "accept?"))
    s.append(box(700, 128, 110, 26, "refusal reason r"))
    s.append(arrow(690, 82, 721, 82, c["muted"]))
    s.append(arrow(786, 82, 817, 82, c["muted"]))
    s.append(arrow(755, 128, 755, 103, c["muted"], "3 3"))
    s.append(f'<text x="755" y="52" text-anchor="middle" font-family="{f}" font-size="11" fill="{v}">invariant across allocation policies</text>')
    return "\n  ".join(s)


def m_dynamite(c, v):
    """Sequential offers along time while cold ischaemic time accumulates."""
    T, xs = 6, [632, 680, 728, 776, 824]
    s = [f'<text x="620" y="40" font-family="{SANS}" font-size="11" fill="{c["muted"]}">sequential offers</text>',
         f'<line x1="620" x2="866" y1="78" y2="78" stroke="{c["border"]}" stroke-width="2"/>']
    for i, x in enumerate(xs):
        if i == 4:
            s.append(f'<circle cx="{x}" cy="78" r="11" fill="{c["canvas"]}" stroke="{c["border"]}" stroke-dasharray="3 3"/>')
            continue
        t0 = 0.5 + i * 0.9
        ok = i == 3
        col = v if ok else c["muted"]
        mark = "✓" if ok else "×"
        s.append(f'<circle cx="{x}" cy="78" r="11" fill="{c["canvas"]}" stroke="{c["border"]}"/>')
        s.append(f'<g opacity="0"><animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;{t0 / T:.3f};{(t0 + .2) / T:.3f};.94;1" dur="{T}s" repeatCount="indefinite"/>'
                 f'<circle cx="{x}" cy="78" r="11" fill="{c["subtle"]}" stroke="{col}" stroke-width="1.6"/>'
                 f'<text x="{x}" y="83" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="600" fill="{col}">{mark}</text></g>')
        s.append(f'<text x="{x}" y="106" text-anchor="middle" font-family="{MONO}" font-size="10" fill="{c["muted"]}">{i + 1}</text>')
    end = (0.5 + 3 * 0.9 + 0.2) / T
    s.append(f'<rect x="620" y="122" width="246" height="8" rx="4" fill="{c["subtle"]}" stroke="{c["border"]}"/>')
    s.append(f'<rect x="620" y="122" width="0" height="8" rx="4" fill="{v}"><animate attributeName="width" values="0;150;150;0" keyTimes="0;{end:.3f};.94;1" dur="{T}s" repeatCount="indefinite"/></rect>')
    s.append(f'<text x="866" y="148" text-anchor="end" font-family="{SANS}" font-size="11" fill="{c["muted"]}">cold ischaemic time</text>')
    return "\n  ".join(s)


CARDS = [
    ("paper-recourse", "NeurIPS 2026", NEURIPS, "Continuous Treatment Effects", "  from recourse data",
     ["Counterfactual explanations as causal supervision:", "dose–response curves even when overlap fails."], m_recourse,
     "NeurIPS 2026: Estimating Continuous Treatment Effects with Recourse Data"),
    ("paper-clexnet", "NeurIPS 2025", NEURIPS, "ClexNet", "  causal, explanation-guided",
     ["Organ-offer acceptance model that learns policy-invariant", "representations from refusal-reason signals."], m_clexnet,
     "NeurIPS 2025: ClexNet, a causal explanation-guided organ-offer model"),
    ("paper-dynamite", "ML4H 2024", ML4H, "DynamITE", "  time-sensitive organ allocation",
     ["Models acceptance behaviour and cold ischaemic time", "across sequential organ offers."], m_dynamite,
     "ML4H 2024: DynamITE, time-sensitive organ allocation"),
]

if __name__ == "__main__":
    for theme in ("dark", "light"):
        (HERE / f"header-{theme}.svg").write_text(header(theme), encoding="utf-8")
        for name, venue, vc, title, sub, desc, motif, label in CARDS:
            (HERE / f"{name}-{theme}.svg").write_text(card(theme, venue, vc[theme], title, sub, desc, motif, label), encoding="utf-8")
    print("built:", ", ".join(sorted(p.name for p in HERE.glob("*.svg"))))
