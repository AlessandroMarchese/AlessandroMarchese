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
                 faint="#484f58", accent="#4493f8", accent_bg="#121d2f", btn="#21262d", btn_border="#363b42",
                 greens=["#0e4429", "#006d32", "#26a641", "#39d353"]),
    "light": dict(canvas="#ffffff", subtle="#f6f8fa", border="#d0d7de", fg="#1f2328", muted="#59636e",
                  faint="#afb8c1", accent="#0969da", accent_bg="#ddf4ff", btn="#f6f8fa", btn_border="#d1d9e0",
                  greens=["#9be9a8", "#40c463", "#30a14e", "#216e39"]),
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ══════════════════════════════════════════════════════════════════════════════
# Header: a 3D Gaussian-mixture density drawn as a wireframe mesh over a floor grid,
# morphing in a loop, floating above floor squares that glow in contribution-graph greens.
# ══════════════════════════════════════════════════════════════════════════════
N_FRAMES, LOOP_S = 40, 20               # surface keyframes per loop, loop length (s)
NG = 20                                 # mesh nodes per side
G = np.linspace(-3.2, 3.2, NG)
ROT, SX, SY, HZ, CX, CY = 0.42, 56, 23, 125, 885, 190


def density(u, v, th):
    m1 = np.array([1.35 * np.cos(th), 1.0 * np.sin(th)])
    m2 = np.array([-1.1 * np.cos(th + 0.6), -1.25 * np.sin(th + 0.6)])
    s1, s2 = 0.70 + 0.08 * np.cos(2 * th), 0.86 - 0.08 * np.cos(2 * th)
    w1 = 0.5 + 0.18 * np.sin(th)
    g = lambda m, s: np.exp(-((u - m[0]) ** 2 + (v - m[1]) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)
    return w1 * g(m1, s1) + (1 - w1) * g(m2, s2)


def project(u, v, z):
    x = CX + SX * (u * np.cos(ROT) + v * np.sin(ROT))
    depth = -u * np.sin(ROT) + v * np.cos(ROT)
    y = CY - SY * depth - HZ * z
    return x, y


GU, GV = np.meshgrid(G, G)              # [iv, iu]
THETAS = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)
ZMAX = max(density(GU, GV, th).max() for th in THETAS)
Z = np.array([density(GU, GV, th) / ZMAX for th in THETAS])          # [frame, iv, iu]


def z_at(phase):
    """Height field at a fractional loop phase, interpolated exactly as the browser interpolates keyframes."""
    f = (phase % 1.0) * N_FRAMES
    k0 = int(np.floor(f)) % N_FRAMES
    a = f - np.floor(f)
    return (1 - a) * Z[k0] + a * Z[(k0 + 1) % N_FRAMES]


def mesh_d(z):
    x, y = project(GU, GV, z)
    x, y = np.round(x * 10).astype(int), np.round(y * 10).astype(int)
    rows = "".join("M" + " ".join(f"{x[i, j]} {y[i, j]}" for j in range(NG)) for i in range(NG))
    cols = "".join("M" + " ".join(f"{x[i, j]} {y[i, j]}" for i in range(NG)) for j in range(NG))
    return rows + cols


N_GLOW = 30                             # floor squares that slowly breathe in and out


def glow_cells(c):
    """Squares on the flat floor grid that glow in GitHub contribution greens, slowly and out of step.
    Periods divide the 20 s surface loop, so the header still repeats cleanly."""
    rng = np.random.default_rng(42)
    cells = set()
    while len(cells) < N_GLOW:
        cells.add((int(rng.integers(0, NG - 1)), int(rng.integers(0, NG - 1))))
    out = []
    for iv, iu in sorted(cells):
        corners = [(iv, iu), (iv, iu + 1), (iv + 1, iu + 1), (iv + 1, iu)]
        xy = np.array([project(G[cu], G[cv], 0.0) for cv, cu in corners])
        ctr = xy.mean(0)
        xy = ctr + (xy - ctr) * 0.86
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in xy) + "Z"
        col = c["greens"][int(rng.choice(4, p=[0.2, 0.3, 0.3, 0.2]))]
        period = float(rng.choice([10.0, 20.0]))
        width = rng.uniform(0.55, 0.8) if period == 10 else rng.uniform(0.3, 0.45)   # 5.5–9 s per breath
        start = rng.uniform(0, 1 - width)
        kt = f"0;{start:.3f};{start + width / 2:.3f};{start + width:.3f};1"
        out.append(f'<g opacity="0" fill="{col}">'
                   f'<animate attributeName="opacity" dur="{period:.0f}s" repeatCount="indefinite" keyTimes="{kt}" '
                   f'values="0;0;1;0;0" calcMode="spline" keySplines="0 0 1 1;.45 0 .55 1;.45 0 .55 1;0 0 1 1"/>'
                   f'<path d="{d}" filter="url(#glow)" opacity="0.8"/><path d="{d}"/></g>')
    return (f'<defs><filter id="glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="4"/></filter></defs>'
            + "".join(out))


def surface(c, theme):
    frames = [mesh_d(Z[k]) for k in range(N_FRAMES)]
    vals = ";".join(frames + frames[:1])
    floor = mesh_d(np.zeros_like(GU))
    return (            f'<g transform="scale(0.1)"><path d="{floor}" fill="none" stroke="{c["border"]}" stroke-width="8" opacity="0.7"/></g>'
            + glow_cells(c) +
            f'<g transform="scale(0.1)">'
            f'<path d="{frames[0]}" fill="none" stroke="{c["accent"]}" stroke-opacity="0.75" stroke-width="10" stroke-linejoin="round">'
            f'<animate attributeName="d" dur="{LOOP_S}s" repeatCount="indefinite" values="{vals}"/></path></g>')


def header(theme):
    c = GH[theme]
    W, H = 1200, 320
    topics = ["causal-inference", "explainable-ai"]
    chips, x = [], 48
    for tp in topics:
        w = 24 + 8.4 * len(tp)
        chips.append(f'<rect x="{x:.0f}" y="190" width="{w:.0f}" height="30" rx="15" fill="{c["accent_bg"]}"/>'
                     f'<text x="{x + w / 2:.0f}" y="210" text-anchor="middle" font-family="{SANS}" font-size="14" font-weight="500" fill="{c["accent"]}">{tp}</text>')
        x += w + 8
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Alessandro Marchese, PhD researcher in causal machine learning and explainable AI">
  <defs><clipPath id="hc"><rect x="592" y="1" width="{W - 593}" height="{H - 2}" rx="5"/></clipPath></defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="{c["canvas"]}" stroke="{c["border"]}"/>
  <text x="48" y="72" font-family="{MONO}" font-size="13" fill="{c["muted"]}">AlessandroMarchese / README.md</text>
  <text x="46" y="128" font-family="{SANS}" font-size="44" font-weight="600" fill="{c["fg"]}" letter-spacing="-0.5">Alessandro Marchese</text>
  <text x="48" y="164" font-family="{SANS}" font-size="21" fill="{c["muted"]}">PhD researcher · causal ML &amp; explainable AI</text>
  {''.join(chips)}
  <text x="48" y="256" font-family="{SANS}" font-size="15" fill="{c["muted"]}">VUB × UHasselt</text>
  <text x="48" y="280" font-family="{SANS}" font-size="15" fill="{c["muted"]}">previously KU Leuven · King’s College London</text>
  <line x1="590" x2="590" y1="40" y2="280" stroke="{c["border"]}"/>
  <g clip-path="url(#hc)">{surface(c, theme)}</g>
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
    """Recourse: units with a negative outcome are moved the minimal distance to the positive side, in a loop."""
    T = 7.0
    rng = np.random.default_rng(5)
    ox, oy, w, h = 620, 30, 250, 124
    A, B = np.array([ox + 70.0, oy]), np.array([ox + w, oy + h - 20.0])
    d = (B - A) / np.linalg.norm(B - A)
    nrm = np.array([d[1], -d[0]])                      # points to the positive side
    L = np.linalg.norm(B - A)
    movers = [(A + t * d + o * nrm, o) for t, o in [(0.22 * L, -34), (0.62 * L, -40), (0.42 * L, -22), (0.82 * L, -26)]]
    P = np.column_stack([rng.uniform(ox + 8, ox + w - 8, 34), rng.uniform(oy + 8, oy + h - 8, 34)])
    s = [f'<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{v}"/></marker></defs>',
         f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="none" stroke="{c["border"]}"/>',
         f'<line x1="{A[0]:.1f}" y1="{A[1]:.1f}" x2="{B[0]:.1f}" y2="{B[1]:.1f}" stroke="{c["muted"]}" stroke-width="1.3" stroke-dasharray="4 3"/>',
         f'<text x="{ox}" y="{oy - 7}" font-family="{SANS}" font-size="10" fill="{c["muted"]}">○ negative outcome</text>',
         f'<text x="{ox + w}" y="{oy - 7}" text-anchor="end" font-family="{SANS}" font-size="10" fill="{c["muted"]}">● positive outcome</text>']
    for p in P:
        dist = (p - A) @ nrm
        if abs(dist) < 8 or any(np.linalg.norm(p - q) < 16 for q, _ in movers):
            continue
        if dist > 0:
            s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["faint"]}"/>')
        else:
            s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["canvas"]}" stroke="{c["fg"]}" stroke-width="1.2"/>')
    for i, (p, dist) in enumerate(movers):
        q = p + (-dist + 8) * nrm
        off = f'begin="-{i * T / len(movers):.2f}s" dur="{T}s" repeatCount="indefinite"'
        pos_kt = "0;0.08;0.24;0.80;0.86;1"                       # wait, move, hold, (hidden) return, wait
        op_kt = "0;0.08;0.12;0.70;0.80;1"                        # fade in as it moves, fade out before returning
        pv = lambda a0, a1: f"{a0:.1f};{a0:.1f};{a1:.1f};{a1:.1f};{a0:.1f};{a0:.1f}"
        s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["canvas"]}" stroke="{c["fg"]}" stroke-width="1.2"/>')
        s.append(
            f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{p[0]:.1f}" y2="{p[1]:.1f}" stroke="{v}" stroke-width="1.5" marker-end="url(#ah)" opacity="0">'
            f'<animate attributeName="x2" values="{pv(p[0], q[0])}" keyTimes="{pos_kt}" {off}/>'
            f'<animate attributeName="y2" values="{pv(p[1], q[1])}" keyTimes="{pos_kt}" {off}/>'
            f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{op_kt}" {off}/></line>')
        s.append(
            f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.4" fill="{v}" opacity="0">'
            f'<animate attributeName="cx" values="{pv(p[0], q[0])}" keyTimes="{pos_kt}" {off}/>'
            f'<animate attributeName="cy" values="{pv(p[1], q[1])}" keyTimes="{pos_kt}" {off}/>'
            f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{op_kt}" {off}/></circle>')
    s.append(f'<text x="{ox + w}" y="{oy + h + 16}" text-anchor="end" font-family="{SANS}" font-size="11" fill="{c["muted"]}">recourse  x → x′  turns a negative outcome positive</text>')
    return "\n  ".join(s)


def m_clexnet(c, v):
    """Different circuits through φ(x) light up as the refusal reason changes."""
    T = 9.0
    inputs = [("patient", 64), ("organ", 112)]
    hidden = [44, 66, 88, 110, 132]
    ix, hx, ox_, oy_ = 676, 754, 820, 88
    reasons = [("organ quality", [1], [0, 1]), ("patient condition", [0], [2, 3]), ("logistics / timing", [0, 1], [3, 4])]
    s = []
    # faint full network
    for _, iy in inputs:
        for hy in hidden:
            s.append(f'<line x1="{ix}" y1="{iy}" x2="{hx - 7}" y2="{hy}" stroke="{c["border"]}"/>')
    for hy in hidden:
        s.append(f'<line x1="{hx + 7}" y1="{hy}" x2="{ox_}" y2="{oy_}" stroke="{c["border"]}"/>')
    # one highlighted circuit per refusal reason
    def crossfade(a, b, f=0.05):
        """Opacity keyframes for a phase visible on [a, b] of a looping timeline, fading over f at each edge."""
        def o(t):
            d_in, d_out = (t - (a - f / 2)) % 1.0, ((b + f / 2) - t) % 1.0
            span = (b - a) + f
            if d_in > span:
                return 0.0
            return float(min(1.0, d_in / f, d_out / f))
        ts = sorted({0.0, 1.0} | {round(x % 1.0, 4) for x in (a - f / 2, a + f / 2, b - f / 2, b + f / 2)})
        return ";".join(f"{t:.4f}" for t in ts), ";".join(f"{o(t):.2f}" for t in ts)

    for k, (name, ins, hs) in enumerate(reasons):
        kt, vals = crossfade(k / 3, (k + 1) / 3)
        anim = f'<animate attributeName="opacity" values="{vals}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
        g = [f'<g opacity="0">{anim}']
        for i in ins:
            iy = inputs[i][1]
            for j in hs:
                pid = f"c{k}{i}{j}"
                g.append(f'<path id="{pid}" d="M{ix},{iy} L{hx - 7},{hidden[j]}" stroke="{v}" stroke-width="2" fill="none"/>')
                g.append(f'<circle r="2.6" fill="{v}"><animateMotion dur="1.125s" begin="{0.15 * j:.2f}s" repeatCount="indefinite"><mpath href="#{pid}"/></animateMotion></circle>')
            g.append(f'<rect x="604" y="{iy - 10}" width="72" height="20" rx="5" fill="{c["subtle"]}" stroke="{v}" stroke-width="1.6"/>'
                     f'<text x="640" y="{iy + 4}" text-anchor="middle" font-family="{SANS}" font-size="11" fill="{c["fg"]}">{inputs[i][0]}</text>')
        for j in hs:
            g.append(f'<path d="M{hx + 7},{hidden[j]} L{ox_},{oy_}" stroke="{v}" stroke-width="2" fill="none"/>')
            g.append(f'<circle cx="{hx}" cy="{hidden[j]}" r="7" fill="{v}"/>')
        g.append("</g>")
        tkt, tvals = crossfade(k / 3 + 0.03, (k + 1) / 3 - 0.03, 0.03)      # labels never overlap
        g.append(f'<text x="880" y="26" text-anchor="end" font-family="{SANS}" font-size="12" font-weight="600" fill="{v}" opacity="0">{name}'
                 f'<animate attributeName="opacity" values="{tvals}" keyTimes="{tkt}" dur="{T}s" repeatCount="indefinite"/></text>')
        s.insert(len(s), "".join(g))
    # static nodes drawn under the highlight layers would be hidden, so put the base nodes first
    base = []
    for name, iy in inputs:
        base.append(f'<rect x="604" y="{iy - 10}" width="72" height="20" rx="5" fill="{c["subtle"]}" stroke="{c["border"]}"/>'
                    f'<text x="640" y="{iy + 4}" text-anchor="middle" font-family="{SANS}" font-size="11" fill="{c["muted"]}">{name}</text>')
    for hy in hidden:
        base.append(f'<circle cx="{hx}" cy="{hy}" r="7" fill="{c["canvas"]}" stroke="{c["muted"]}" stroke-width="1.2"/>')
    base.append(f'<text x="{hx}" y="160" text-anchor="middle" font-family="{SANS}" font-size="12" font-style="italic" fill="{c["muted"]}">φ(x)</text>')
    base.append(f'<rect x="{ox_}" y="{oy_ - 13}" width="60" height="26" rx="6" fill="{c["subtle"]}" stroke="{c["border"]}"/>'
                f'<text x="{ox_ + 30}" y="{oy_ + 4}" text-anchor="middle" font-family="{SANS}" font-size="12" fill="{c["fg"]}">accept?</text>')
    base.append(f'<text x="742" y="26" text-anchor="end" font-family="{SANS}" font-size="12" fill="{c["muted"]}">refusal reason:</text>')
    edges = [x for x in s if x.startswith("<line")]
    layers = [x for x in s if not x.startswith("<line")]
    return "\n  ".join(edges + base + layers)


def m_dynamite(c, v):
    """Sequential offers along time while cold ischaemic time accumulates."""
    T, xs = 7, [632, 680, 728, 776, 824]
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
        s.append(f'<g opacity="0"><animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;{t0 / T:.3f};{(t0 + .2) / T:.3f};.84;.96;1" dur="{T}s" repeatCount="indefinite"/>'
                 f'<circle cx="{x}" cy="78" r="11" fill="{c["subtle"]}" stroke="{col}" stroke-width="1.6"/>'
                 f'<text x="{x}" y="83" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="600" fill="{col}">{mark}</text></g>')
        s.append(f'<text x="{x}" y="106" text-anchor="middle" font-family="{MONO}" font-size="10" fill="{c["muted"]}">{i + 1}</text>')
    end = (0.5 + 3 * 0.9 + 0.2) / T
    s.append(f'<rect x="620" y="122" width="246" height="8" rx="4" fill="{c["subtle"]}" stroke="{c["border"]}"/>')
    s.append(f'<rect x="620" y="122" width="0" height="8" rx="4" fill="{v}">'
             f'<animate attributeName="width" values="0;150;150;0;0" keyTimes="0;{end:.3f};.97;.98;1" dur="{T}s" repeatCount="indefinite"/>'
             f'<animate attributeName="opacity" values="1;1;0;0;1" keyTimes="0;.84;.96;.98;1" dur="{T}s" repeatCount="indefinite"/></rect>')
    s.append(f'<text x="866" y="148" text-anchor="end" font-family="{SANS}" font-size="11" fill="{c["muted"]}">cold ischaemic time</text>')
    return "\n  ".join(s)


# ══════════════════════════════════════════════════════════════════════════════
# Paper / Code buttons in GitHub's button style (these are the clickable parts).
# ══════════════════════════════════════════════════════════════════════════════
def button(theme, label, icon):
    c = GH[theme]
    W, H = 86, 30
    icons = {
        "paper": f'<path d="M3.5 1.5h6l3 3v10h-9z" fill="none" stroke="{c["muted"]}" stroke-width="1.3" stroke-linejoin="round"/>'
                 f'<path d="M9.5 1.5v3h3M5.5 8h5M5.5 10.5h5M5.5 13h3" stroke="{c["muted"]}" stroke-width="1.2" fill="none"/>',
        "code": f'<path d="M5 4 1.5 8 5 12M11 4l3.5 4L11 12" fill="none" stroke="{c["muted"]}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
    }
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{label}">
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6" fill="{c["btn"]}" stroke="{c["btn_border"]}"/>
  <g transform="translate(14,7)">{icons[icon]}</g>
  <text x="38" y="20" font-family="{SANS}" font-size="13" font-weight="500" fill="{c["fg"]}">{label}</text>
</svg>
"""


CARDS = [
    ("paper-recourse", "NeurIPS 2026", NEURIPS, "Treatment Effects", "  from recourse data",
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
    for old in HERE.glob("*.svg"):
        old.unlink()
    for theme in ("dark", "light"):
        (HERE / f"header-{theme}.svg").write_text(header(theme), encoding="utf-8")
        for name, venue, vc, title, sub, desc, motif, label in CARDS:
            (HERE / f"{name}-{theme}.svg").write_text(card(theme, venue, vc[theme], title, sub, desc, motif, label), encoding="utf-8")
        (HERE / f"btn-paper-{theme}.svg").write_text(button(theme, "Paper", "paper"), encoding="utf-8")
        (HERE / f"btn-code-{theme}.svg").write_text(button(theme, "Code", "code"), encoding="utf-8")
    for p in sorted(HERE.glob("*.svg")):
        print(f"{p.name:28s} {p.stat().st_size / 1024:6.1f} KB")
