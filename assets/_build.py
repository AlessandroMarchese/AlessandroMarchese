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
                 g_fill="#0b3d1c", g_line="#3fb950", g_block="#2ea043"),
    "light": dict(canvas="#ffffff", subtle="#f6f8fa", border="#d0d7de", fg="#1f2328", muted="#59636e",
                  faint="#afb8c1", accent="#0969da", accent_bg="#ddf4ff", btn="#f6f8fa", btn_border="#d1d9e0",
                  g_fill="#aceebb", g_line="#1a7f37", g_block="#2da44e"),
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ══════════════════════════════════════════════════════════════════════════════
# Header: a 3D Gaussian-mixture density, projected by hand, morphing in a loop.
# Each ridge is one path whose `d` is animated through precomputed frames.
# ══════════════════════════════════════════════════════════════════════════════
N_FRAMES, LOOP_S = 28, 16
U = np.linspace(-3.2, 3.2, 34)          # along each ridge
V = np.linspace(3.2, -3.2, 24)          # ridges, far → near (painter's order)
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


def surface_frames():
    thetas = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)
    zmax = max(density(*np.meshgrid(U, V), th).max() for th in thetas)
    frames = []                                  # frames[k][ridge] -> d string
    for th in thetas:
        ridges = []
        for v in V:
            z = density(U, v, th) / zmax
            x, y = project(U, v, z)
            bx, by = project(U[[-1, 0]], v, 0)   # floor line closes the shape
            pts = " ".join(f"{a * 10:.0f} {b * 10:.0f}" for a, b in zip(x, y))
            ridges.append(f"M{pts} L{bx[0] * 10:.0f} {by[0] * 10:.0f} L{bx[1] * 10:.0f} {by[1] * 10:.0f}Z")
        frames.append(ridges)
    return frames


FRAMES = surface_frames()


GLITCH_S = 11.0                                   # own cycle, out of step with the 16 s surface loop
BURSTS = [(3.0, 0.40), (7.9, 0.22), (9.65, 0.16)]  # (start, duration) in seconds


def surface(c):
    """The morphing surface, plus a glitch: bands tear sideways and reveal a green copy underneath."""
    n = len(V)
    ridges = []
    for r in range(n):
        vals = ";".join(FRAMES[k][r] for k in range(N_FRAMES)) + ";" + FRAMES[0][r]
        op = 0.35 + 0.65 * r / (n - 1)
        ridges.append(f'<path d="{FRAMES[0][r]}" stroke-opacity="{op:.2f}" stroke-width="14" stroke-linejoin="round">'
                      f'<animate attributeName="d" dur="{LOOP_S}s" repeatCount="indefinite" values="{vals}"/></path>')
    corners = [project(np.array(a), np.array(b), 0) for a, b in [(-3.2, 3.2), (3.2, 3.2), (3.2, -3.2), (-3.2, -3.2)]]
    floor = " L".join(f"{float(x):.1f} {float(y):.1f}" for x, y in corners)

    rng = np.random.default_rng(23)
    edges = np.concatenate([[0], np.sort(rng.uniform(40, 300, 11)), [320]])
    defs = [f'<g id="surf" transform="scale(0.1)">{"".join(ridges)}</g>']
    layers = [f'<path d="M{floor}Z" fill="none" stroke="{c["border"]}"/>',
              f'<use href="#surf" fill="{c["g_fill"]}" stroke="{c["g_line"]}" opacity="0">'
              f'<animate attributeName="opacity" calcMode="discrete" dur="{GLITCH_S}s" repeatCount="indefinite" '
              f'keyTimes="0;{";".join(f"{t0 / GLITCH_S:.4f};{(t0 + d) / GLITCH_S:.4f}" for t0, d in BURSTS)}" '
              f'values="0;{";".join("1;0" for _ in BURSTS)}"/></use>']
    for i in range(len(edges) - 1):
        y0, y1 = edges[i], edges[i + 1] + 1.0                      # 1px overlap hides seams
        defs.append(f'<clipPath id="b{i}"><rect x="590" y="{y0:.1f}" width="610" height="{y1 - y0:.1f}"/></clipPath>')
        keys = [(0.0, 0.0)]
        for t0, d in BURSTS:
            if rng.uniform() < 0.55:
                for k in range(3):
                    keys.append(((t0 + k * d / 3) / GLITCH_S, float(rng.choice([-1, 1]) * rng.uniform(5, 34))))
                keys.append(((t0 + d) / GLITCH_S, 0.0))
        keys.append((1.0, 0.0))
        anim = ""
        if len(keys) > 2:
            anim = (f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" dur="{GLITCH_S}s" repeatCount="indefinite" '
                    f'keyTimes="{";".join(f"{t:.4f}" for t, _ in keys)}" values="{";".join(f"{dx:.0f} 0" for _, dx in keys)}"/>')
        layers.append(f'<g clip-path="url(#b{i})"><use href="#surf" fill="{c["canvas"]}" stroke="{c["accent"]}">{anim}</use></g>')
    # green blocks flickering during the bursts
    for t0, d in BURSTS:
        for _ in range(7):
            x, y = rng.uniform(650, 1130), rng.uniform(70, 280)
            w, h = rng.uniform(6, 70), rng.uniform(3, 14)
            on = t0 + rng.uniform(0, d * 0.6)
            off = min(on + rng.uniform(0.05, d * 0.6), t0 + d)
            layers.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{c["g_block"]}" opacity="0">'
                          f'<animate attributeName="opacity" calcMode="discrete" dur="{GLITCH_S}s" repeatCount="indefinite" '
                          f'keyTimes="0;{on / GLITCH_S:.4f};{off / GLITCH_S:.4f}" values="0;0.9;0"/></rect>')
    return "<defs>" + "".join(defs) + "</defs>" + "".join(layers)


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
  <g clip-path="url(#hc)">{surface(c)}</g>
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
    movers = [(A + t * d + off * nrm, off) for t, off in [(0.22 * L, -34), (0.42 * L, -22), (0.62 * L, -40), (0.82 * L, -26)]]
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
        a = (0.6 + 1.1 * i) / T              # start of this unit's move
        b = a + 0.7 / T                      # end of move
        kt = f"0;{a:.3f};{b:.3f};0.90;0.97;1"
        s.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3" fill="{c["canvas"]}" stroke="{c["fg"]}" stroke-width="1.2"/>')
        s.append(
            f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{p[0]:.1f}" y2="{p[1]:.1f}" stroke="{v}" stroke-width="1.5" marker-end="url(#ah)" opacity="0">'
            f'<animate attributeName="x2" values="{p[0]:.1f};{p[0]:.1f};{q[0]:.1f};{q[0]:.1f};{q[0]:.1f};{p[0]:.1f}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{p[1]:.1f};{p[1]:.1f};{q[1]:.1f};{q[1]:.1f};{q[1]:.1f};{p[1]:.1f}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/></line>')
        s.append(
            f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.4" fill="{v}" opacity="0">'
            f'<animate attributeName="cx" values="{p[0]:.1f};{p[0]:.1f};{q[0]:.1f};{q[0]:.1f};{q[0]:.1f};{p[0]:.1f}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{p[1]:.1f};{p[1]:.1f};{q[1]:.1f};{q[1]:.1f};{q[1]:.1f};{p[1]:.1f}" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/></circle>')
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
    for k, (name, ins, hs) in enumerate(reasons):
        a, b, f = k / 3, (k + 1) / 3, 0.035
        kt = f"0;{a:.3f};{a + f:.3f};{b - f:.3f};{b:.3f};1"
        anim = f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{kt}" dur="{T}s" repeatCount="indefinite"/>'
        if k == 0:
            anim = f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;{b - f:.3f};{b:.3f};1" dur="{T}s" repeatCount="indefinite"/>'
        g = [f'<g opacity="{1 if k == 0 else 0}">{anim}']
        for i in ins:
            iy = inputs[i][1]
            for j in hs:
                pid = f"c{k}{i}{j}"
                g.append(f'<path id="{pid}" d="M{ix},{iy} L{hx - 7},{hidden[j]}" stroke="{v}" stroke-width="2" fill="none"/>')
                g.append(f'<circle r="2.6" fill="{v}"><animateMotion dur="1.1s" begin="{0.15 * j:.2f}s" repeatCount="indefinite"><mpath href="#{pid}"/></animateMotion></circle>')
            g.append(f'<rect x="604" y="{iy - 10}" width="72" height="20" rx="5" fill="{c["subtle"]}" stroke="{v}" stroke-width="1.6"/>'
                     f'<text x="640" y="{iy + 4}" text-anchor="middle" font-family="{SANS}" font-size="11" fill="{c["fg"]}">{inputs[i][0]}</text>')
        for j in hs:
            g.append(f'<path d="M{hx + 7},{hidden[j]} L{ox_},{oy_}" stroke="{v}" stroke-width="2" fill="none"/>')
            g.append(f'<circle cx="{hx}" cy="{hidden[j]}" r="7" fill="{v}"/>')
        g.append(f'<text x="880" y="26" text-anchor="end" font-family="{SANS}" font-size="12" font-weight="600" fill="{v}">{name}</text>')
        g.append("</g>")
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
