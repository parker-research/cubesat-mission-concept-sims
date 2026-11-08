# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "pymsis",
# ]
# ///
"""
Rough simulation of a finned "lawn dart" in very low Earth orbit (VLEO) and
through reentry, using NRLMSIS 2.0 (via pymsis) for the atmosphere.

Subcommands
-----------
  render   Draw the dart geometry (3D view + dimensioned side view).
  survey   Fixed-altitude attitude runs, aero vs gravity-gradient stiffness,
           orbit-averaged lifetime, and a reentry from 140 km.
  run      Continuous orbit+attitude run from release to the ground
           (saves every integrator step to an .npz file).
  plot     Overview + attitude-snapshot figures from a `run` output file.
  all      render + run + plot.

Examples
--------
  uv run vleo_dart_sim.py render
  uv run vleo_dart_sim.py run --mass 0.1 --release-alt 500 --aoa0 90
  uv run vleo_dart_sim.py plot
  uv run vleo_dart_sim.py survey --mass 1.0

Model (deliberately rough)
--------------------------
  * Planar: equatorial prograde orbit, attitude = pitch only (rotation about
    the orbit normal). Spherical Earth, point-mass gravity + gravity-gradient
    torque. Atmosphere co-rotates with the Earth.
  * Atmosphere: NRLMSIS 2.0 tabulated vs altitude and local solar time
    (equator, 2026-11-07) for a chosen solar/geomagnetic activity level.
  * Aerodynamics:
      - Free-molecular: Sentman diffuse-reflection panel model (full
        accommodation, 300 K wall), including thermal motion, on a faceted
        cone-cylinder with 4 cruciform fins.
      - Continuum: Barrowman-style normal-force build-up (nose + fins + body
        crossflow) with a Mach-dependent axial drag.
      - Transitional: blended by Knudsen number (Kn 0.01 .. 10).
      - Local velocity at each panel includes omega x r, so aerodynamic
        pitch damping emerges naturally. No shadowing between panels.
  * Attitude convention: aoa = angle from relative wind to nose, positive
    counter-clockwise. In the (counter-clockwise) orbit, aoa = +90 is nadir,
    -90 is zenith, 0 is nose into the wind.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pymsis
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================================================================
# Constants
# =============================================================================
MU = 3.986004418e14
RE = 6378137.0
WE = 7.2921159e-5
KB = 1.380649e-23
D_MOL = 3.7e-10          # effective molecular diameter for mean free path
T_WALL = 300.0
DATE = np.datetime64("2026-11-07T12:00")

SOLAR = {  # name: (F10.7, F10.7a, Ap)
    "low": (70, 70, 4),
    "moderate": (150, 150, 15),
    "high": (220, 220, 30),
}
SOLAR_COLORS = {"low": "tab:blue", "moderate": "tab:green", "high": "tab:red"}


def solar_label(name):
    return f"{name} solar activity, F10.7={SOLAR[name][0]}"


# =============================================================================
# Atmosphere
# =============================================================================
class Atmosphere:
    """NRLMSIS table vs altitude x longitude (= local solar time at fixed UTC;
    with the Sun ~fixed in inertial space, the satellite's inertial angle maps
    onto this longitude axis)."""

    def __init__(self, f107, f107a, ap, dh=1.0, dlon=10.0):
        self.alts = np.arange(0, 1000 + dh, dh)
        self.lons = np.arange(0, 360 + dlon, dlon)
        self.dh, self.dlon = dh, dlon
        out = pymsis.calculate(DATE, self.lons, np.array([0.0]), self.alts,
                               f107s=f107, f107as=f107a, aps=[[ap] * 7])
        out = out[0, :, 0, :, :]            # (nlon, nalt, 11)
        self.logrho = np.log(out[..., 0])
        self.logn = np.log(np.nansum(out[..., 1:10], axis=-1))
        self.T = out[..., 10]

    @classmethod
    def named(cls, name):
        return cls(*SOLAR[name])

    def __call__(self, h_km, lon_deg):
        h = min(max(h_km, 0.0), self.alts[-1] - 1e-6)
        lo = lon_deg % 360.0
        fi, fj = h / self.dh, lo / self.dlon
        i, j = int(fi), int(fj)
        a, b = fi - i, fj - j

        def bil(tab):
            return ((1 - a) * (1 - b) * tab[j, i] + a * (1 - b) * tab[j, i + 1]
                    + (1 - a) * b * tab[j + 1, i] + a * b * tab[j + 1, i + 1])

        return np.exp(bil(self.logrho)), np.exp(bil(self.logn)), bil(self.T)

    def mean_profile(self):
        rho = np.exp(self.logrho)
        return self.alts, rho.mean(0), rho.min(0), rho.max(0)


# =============================================================================
# Vehicle
# =============================================================================
class Dart:
    """0.5 m x 50 mm cone-cylinder dart, CG 0.20 m behind tip, four 80 x 100 mm
    tail fins. Body frame: x toward nose, origin at CG."""

    def __init__(self, m=0.1):
        self.m = m
        self.L, self.D = 0.50, 0.05
        self.r = self.D / 2
        self.Lc = 0.10
        self.x_tip = 0.20
        self.x_tail = self.x_tip - self.L
        self.fin_span, self.fin_chord = 0.08, 0.10
        self.A_ref = np.pi * self.r ** 2
        self.I_t = self.m * self.L ** 2 / 12 + self.m * (self.L / 2 - self.x_tip) ** 2
        self.I_a = 0.5 * self.m * self.r ** 2
        self._build_panels()
        # continuum (Barrowman-style) component locations
        self.x_nose_cp = self.x_tip - 0.666 * self.Lc
        self.x_cyl = 0.5 * (self.x_tip - self.Lc + self.x_tail)
        self.L_cyl = self.L - self.Lc
        self.x_fin = self.x_tail + self.fin_chord / 2
        s = self.fin_span
        self.A_fin_pair = 2 * s * self.fin_chord
        self.fin_AR = (2 * s) ** 2 / self.A_fin_pair
        self.K_int = 1 + self.r / (s + self.r)

    def _build_panels(self):
        P_r, P_n, P_A = [], [], []
        nf = 12
        phis = (np.arange(nf) + 0.5) * 2 * np.pi / nf
        ha = np.arctan(self.r / self.Lc)
        slant = np.hypot(self.r, self.Lc)
        xc = self.x_tip - 2 / 3 * self.Lc
        for p in phis:  # nose cone facets
            P_n.append([np.sin(ha), np.cos(ha) * np.cos(p), np.cos(ha) * np.sin(p)])
            P_r.append([xc, 2 / 3 * self.r * np.cos(p), 2 / 3 * self.r * np.sin(p)])
            P_A.append(np.pi * self.r * slant / nf)
        x0, x1 = self.x_tail, self.x_tip - self.Lc
        nseg = 8
        segL = (x1 - x0) / nseg
        for k in range(nseg):  # cylinder facets
            xs = x0 + (k + 0.5) * segL
            for p in phis:
                P_n.append([0, np.cos(p), np.sin(p)])
                P_r.append([xs, self.r * np.cos(p), self.r * np.sin(p)])
                P_A.append(2 * np.pi * self.r * segL / nf)
        P_n.append([-1, 0, 0]); P_r.append([self.x_tail, 0, 0]); P_A.append(np.pi * self.r ** 2)
        # 4 fins, two faces each. Fins spanning +/-z have normals +/-y (pitch-active).
        xf = self.x_tail + self.fin_chord / 2
        off = self.r + self.fin_span / 2
        Af = self.fin_span * self.fin_chord
        for span_dir, nrm in (([0, 0, 1], [0, 1, 0]), ([0, 0, -1], [0, 1, 0]),
                              ([0, 1, 0], [0, 0, 1]), ([0, -1, 0], [0, 0, 1])):
            c = np.array([xf, 0, 0]) + off * np.array(span_dir)
            for sgn in (1, -1):
                P_n.append(list(sgn * np.array(nrm))); P_r.append(list(c)); P_A.append(Af)
        self.P_r = np.array(P_r, float)
        self.P_n = np.array(P_n, float)
        self.P_A = np.array(P_A, float)


# =============================================================================
# Aerodynamics
# =============================================================================
def sentman_forces(dart, v_body, omega, rho, n, T):
    """Free-molecular (Sentman, fully diffuse) force and z-torque, body frame.
    v_body: velocity of the body relative to the air, body frame."""
    vloc = v_body + np.cross([0.0, 0.0, omega], dart.P_r)
    V = np.linalg.norm(vloc, axis=1)
    uD = -vloc / V[:, None]
    m_mol = rho / n
    S = V / np.sqrt(2 * KB * T / m_mol)
    vre = np.sqrt(2 * KB * T_WALL / m_mol) / V
    nrm = dart.P_n
    g = -np.einsum("ij,ij->i", uD, nrm)
    P = np.exp(-(g * S) ** 2) / S
    G = 1 / (2 * S ** 2)
    Q = 1 + G
    Z = 1 + erf(g * S)
    q = 0.5 * rho * V ** 2
    cD = P / np.sqrt(np.pi) + g * Q * Z
    lift_vec = -nrm - g[:, None] * uD
    reem = 0.5 * vre * (g * np.sqrt(np.pi) * Z + P)
    F = (q * dart.P_A)[:, None] * (cD[:, None] * uD + (G * Z)[:, None] * lift_vec
                                   - reem[:, None] * nrm)
    return F.sum(0), np.cross(dart.P_r, F)[:, 2].sum()


def cd0_axial(M):
    return np.interp(M, [0.0, 0.8, 1.05, 1.3, 2.0, 5.0, 30.0],
                     [0.45, 0.50, 0.95, 0.85, 0.70, 0.55, 0.50])


def fin_cna(dart, M):
    AR = dart.fin_AR
    sub = 2 * np.pi * AR / (2 + np.sqrt(4 + AR ** 2))
    if M <= 0.8:
        c = sub
    elif M >= 1.3:
        c = 4 / np.sqrt(M ** 2 - 1)
    else:
        c = np.interp(M, [0.8, 1.3], [sub, 4 / np.sqrt(1.3 ** 2 - 1)])
    return min(c, sub)


def continuum_forces(dart, v_body, omega, rho, n, T):
    a_snd = np.sqrt(1.4 * KB * T / (rho / n))
    F = np.zeros(3)
    tau = 0.0

    def local(x):
        vl = v_body + np.cross([0, 0, omega], [x, 0, 0])
        V = np.linalg.norm(vl)
        u = -vl / V
        return V, np.arctan2(u[1], -u[0]), u

    V, al, u = local(0.0)  # axial drag
    F[0] += 0.5 * rho * V ** 2 * dart.A_ref * cd0_axial(V / a_snd) * np.sign(u[0])
    V, al, u = local(dart.x_nose_cp)  # nose (slender body CNa = 2)
    fy = 0.5 * rho * V ** 2 * dart.A_ref * 2 * np.sin(al) * np.cos(al)
    F[1] += fy; tau += dart.x_nose_cp * fy
    V, al, u = local(dart.x_cyl)  # body crossflow
    fy = 0.5 * rho * V ** 2 * 1.2 * dart.D * dart.L_cyl * np.sin(al) * abs(np.sin(al))
    F[1] += fy; tau += dart.x_cyl * fy
    V, al, u = local(dart.x_fin)  # fins
    cn = (dart.K_int * fin_cna(dart, V / a_snd) * np.sin(al) * np.cos(al)
          + 1.28 * np.sin(al) * abs(np.sin(al)))
    fy = 0.5 * rho * V ** 2 * dart.A_fin_pair * cn
    F[1] += fy; tau += dart.x_fin * fy
    return F, tau


def fm_weight(n, Lref=0.5):
    """Free-molecular weight from Knudsen number (1 = free molecular)."""
    lam = 1 / (np.sqrt(2) * np.pi * D_MOL ** 2 * n)
    x = np.log10(lam / Lref)
    return float(np.clip(0.5 * (1 + np.sin(np.pi / 2 * np.clip((x + 0.5) / 1.5, -1, 1))), 0, 1))


def aero(dart, v_body, omega, rho, n, T):
    w = fm_weight(n)
    F = np.zeros(3); tau = 0.0
    if w > 0:
        f, t = sentman_forces(dart, v_body, omega, rho, n, T)
        F += w * f; tau += w * t
    if w < 1:
        f, t = continuum_forces(dart, v_body, omega, rho, n, T)
        F += (1 - w) * f; tau += (1 - w) * t
    return F, tau


# =============================================================================
# Dynamics
# =============================================================================
def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_rhs(dart, atm):
    """State: [X, Y, VX, VY, psi, omega] (inertial, planar)."""
    def rhs(t, y):
        X, Y, VX, VY, psi, om = y
        r = np.array([X, Y, 0.0]); rn = np.linalg.norm(r)
        rho, n, T = atm((rn - RE) / 1e3, np.degrees(np.arctan2(Y, X)))
        v_rel = np.array([VX + WE * Y, VY - WE * X, 0.0])
        Rb = rot(psi)
        Fb, tau = aero(dart, Rb.T @ v_rel, om, rho, n, T)
        acc = -MU * r / rn ** 3 + (Rb @ Fb) / dart.m
        r_b = Rb.T @ r
        tau_gg = 3 * MU / rn ** 5 * r_b[0] * r_b[1] * (dart.I_t - dart.I_a)
        return [VX, VY, acc[0], acc[1], om, (tau + tau_gg) / dart.I_t]
    return rhs


def ground_event(t, y):
    return np.hypot(y[0], y[1]) - RE
ground_event.terminal = True


def circ_state(h_km, aoa0_deg):
    """Circular orbit at h_km, nose at aoa0 from the wind, rotating with the
    local vertical. Returns (state, orbital period)."""
    r = RE + h_km * 1e3
    v = np.sqrt(MU / r)
    n = v / r
    return [r, 0.0, 0.0, v, np.pi / 2 + np.radians(aoa0_deg), n], 2 * np.pi / n


def diagnostics(t, y):
    X, Y, VX, VY, psi, om = y
    rn = np.hypot(X, Y)
    up = np.stack([X, Y]) / rn
    hor = np.stack([-up[1], up[0]])
    v = np.stack([VX + WE * Y, VY - WE * X])
    nose = np.stack([np.cos(psi), np.sin(psi)])
    vhat = v / np.linalg.norm(v, axis=0)
    return dict(
        t=t, h=(rn - RE) / 1e3,
        nose_pitch=np.degrees(np.arctan2((nose * up).sum(0), (nose * hor).sum(0))),
        fpa=np.degrees(np.arctan2((v * up).sum(0), (v * hor).sum(0))),
        aoa=np.degrees(np.arctan2(vhat[0] * nose[1] - vhat[1] * nose[0], (vhat * nose).sum(0))),
        speed=np.linalg.norm(v, axis=0))


def sol_diag(sol):
    return diagnostics(sol.t, sol.y)


# =============================================================================
# Analysis helpers
# =============================================================================
def stiffness_curve(dart, atm, hs):
    """Aero restoring stiffness about nose-into-wind (mean / night / day
    density) and gravity-gradient stiffness, N m/rad."""
    d = np.radians(1.0)
    out = {"mean": [], "min": [], "max": [], "gg": []}
    for h in hs:
        trip = [atm(h, lo) for lo in atm.lons[:-1]]
        rhos = np.array([q[0] for q in trip])
        cases = {"mean": (rhos.mean(), np.mean([q[1] for q in trip]), np.mean([q[2] for q in trip])),
                 "min": trip[rhos.argmin()], "max": trip[rhos.argmax()]}
        r = RE + h * 1e3
        v = np.sqrt(MU / r) - WE * r
        for k, (rho, n, T) in cases.items():
            tp = aero(dart, rot(-d) @ np.array([v, 0, 0]), 0.0, rho, n, T)[1]
            tm = aero(dart, rot(d) @ np.array([v, 0, 0]), 0.0, rho, n, T)[1]
            out[k].append(-(tp - tm) / (2 * d))
        out["gg"].append(3 * MU / r ** 3 * (dart.I_t - dart.I_a))
    out = {k: np.array(v) for k, v in out.items()}
    out["crossover"] = float(np.interp(0, (out["mean"] - out["gg"])[::-1], np.asarray(hs)[::-1]))
    return out


def nose_first_lifetime(dart, atm, h_start, h_end=140.0):
    """Orbit-averaged decay assuming the dart flies nose-first. Returns
    (days, altitude_km) arrays."""
    lons = atm.lons[:-1:3]

    def a_drag(h):
        r = RE + h * 1e3
        v = np.sqrt(MU / r) - WE * r
        tot = sum(np.linalg.norm(aero(dart, np.array([v, 0, 0]), 0.0, *atm(h, lo))[0]) for lo in lons)
        return tot / len(lons) / dart.m

    hg = np.arange(h_end - 20, h_start + 2, 2.0)
    lad = np.log([a_drag(h) for h in hg])
    f = lambda t, y: [-2 * np.exp(np.interp((y[0] - RE) / 1e3, hg, lad)) * np.sqrt(y[0] ** 3 / MU)]
    ev = lambda t, y: y[0] - (RE + h_end * 1e3)
    ev.terminal = True
    s = solve_ivp(f, (0, 20 * 365 * 86400), [RE + h_start * 1e3], events=ev,
                  max_step=6 * 3600, rtol=1e-8)
    return s.t / 86400, (s.y[0] - RE) / 1e3


def plot_density(ax, atms):
    for name, at in atms.items():
        h, m, lo, hi = at.mean_profile()
        c = SOLAR_COLORS[name]
        ax.semilogx(m, h, c=c, label=solar_label(name))
        ax.fill_betweenx(h, lo, hi, color=c, alpha=0.15)
    ax.set_ylim(0, 600); ax.set_xlim(1e-15, 2)
    ax.set_xlabel("density (kg/m³)"); ax.set_ylabel("altitude (km)")
    ax.set_title("NRLMSIS 2.0, equator, 7 Nov 2026\n(line = day/night mean, band = day–night range)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def plot_stiffness(ax, hs, k, title, mark_alt=None):
    ax.semilogy(hs, k["mean"], "k", label="aerodynamic restoring (mean density)")
    ax.fill_between(hs, k["min"], k["max"], color="k", alpha=0.15, label="aero, night → day")
    ax.semilogy(hs, k["gg"], "tab:orange", label="gravity gradient (pulls toward vertical)")
    ax.axvline(k["crossover"], ls=":", c="gray")
    ax.text(k["crossover"] + 8, k["gg"][0] * 3, f"crossover ≈ {k['crossover']:.0f} km", fontsize=9)
    if mark_alt is not None:
        ax.axvline(mark_alt, ls="--", c="tab:purple", lw=0.8)
        ax.text(mark_alt + 5, k["mean"].max() / 3, "release", color="tab:purple", fontsize=9)
    ax.set_xlabel("altitude (km)"); ax.set_ylabel("pitch stiffness (N·m/rad)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, which="both")


def plot_reentry(ax_a, ax_b, d, title_a, title_b):
    ax_a.plot(d["nose_pitch"], d["h"], c="k", lw=0.4, alpha=0.6, label="nose pitch vs local horizon")
    ax_a.plot(d["fpa"], d["h"], c="tab:orange", lw=1.5, ls="--", label="flight path angle (rel. air)")
    ax_a.set_xlim(-100, 40)
    ax_a.set_xlabel("degrees (0 = horizontal, −90 = straight down)"); ax_a.set_ylabel("altitude (km)")
    ax_a.set_title(title_a); ax_a.legend(fontsize=8, loc="upper left"); ax_a.grid(alpha=0.3)
    ax_b.plot(d["speed"], d["h"], c="tab:blue")
    ax_b.set_xlim(0, 8000)
    ax_b.set_xlabel("speed relative to air (m/s)", color="tab:blue"); ax_b.set_ylabel("altitude (km)")
    tw = ax_b.twiny()
    tw.plot(np.abs(d["aoa"]), d["h"], c="tab:red", lw=0.5, alpha=0.8)
    tw.set_xlim(0, 60); tw.set_xlabel("|angle of attack| (deg)", color="tab:red")
    ax_b.set_title(title_b, pad=30); ax_b.grid(alpha=0.3)


def footer(fig, dart, extra):
    fig.text(0.5, 0.005,
             f"Dart: 0.5 m × 50 mm cone-cylinder, {dart.m*1000:.0f} g, CG 0.20 m from tip, four 80×100 mm "
             f"tail fins. {extra} Planar equatorial orbit, pitch only.", ha="center", fontsize=8, wrap=True)


# =============================================================================
# Subcommand: render
# =============================================================================
def cmd_render(args):
    d = Dart(m=args.mass)
    mm = 1000.0
    x_tip, x_tail, r, Lc = d.x_tip * mm, d.x_tail * mm, d.r * mm, d.Lc * mm
    x_sh = x_tip - Lc
    span, chord = d.fin_span * mm, d.fin_chord * mm

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])

    ax = fig.add_subplot(gs[0], projection="3d")
    th = np.linspace(0, 2 * np.pi, 48)
    light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
    shade = lambda base, nrm: tuple(np.clip(np.array(base) * (0.45 + 0.55 * max(0, np.dot(nrm, light))), 0, 1))
    body_c, nose_c, fin_c = (0.85, 0.85, 0.88), (0.85, 0.25, 0.2), (0.15, 0.35, 0.75)
    ha = np.arctan(r / Lc)
    polys, cols = [], []
    for a0, a1 in zip(th[:-1], th[1:]):
        am = 0.5 * (a0 + a1)
        polys.append([[x_tail, r*np.cos(a0), r*np.sin(a0)], [x_sh, r*np.cos(a0), r*np.sin(a0)],
                      [x_sh, r*np.cos(a1), r*np.sin(a1)], [x_tail, r*np.cos(a1), r*np.sin(a1)]])
        cols.append(shade(body_c, np.array([0, np.cos(am), np.sin(am)])))
        polys.append([[x_sh, r*np.cos(a0), r*np.sin(a0)], [x_tip, 0, 0], [x_sh, r*np.cos(a1), r*np.sin(a1)]])
        cols.append(shade(nose_c, np.array([np.sin(ha), np.cos(ha)*np.cos(am), np.cos(ha)*np.sin(am)])))
    polys.append([[x_tail, r*np.cos(a), r*np.sin(a)] for a in th]); cols.append(shade(body_c, np.array([-1, 0, 0])))
    for dy, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        e = np.array([0, dy, dz], float)
        base = np.array([x_tail, 0, 0])
        p = [base + r*e, base + [chord, 0, 0] + r*e, base + [chord, 0, 0] + (r+span)*e, base + (r+span)*e]
        nrm = np.array([0, dz, dy], float)
        polys.append([list(q) for q in p]); cols.append(shade(fin_c, nrm if np.dot(nrm, light) > 0 else -nrm))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.15), linewidths=0.3))
    ax.set_xlim(x_tail - 30, x_tip + 30); ax.set_ylim(-110, 110); ax.set_zlim(-110, 110)
    ax.set_box_aspect((x_tip - x_tail + 60, 220, 220), zoom=1.05)
    ax.view_init(elev=22, azim=-58); ax.set_axis_off()
    ax.set_title(f"Simulated dart: {d.m*1000:.0f} g, 0.5 m cone-cylinder with four tail fins", fontsize=13)

    ax2 = fig.add_subplot(gs[1])
    ax2.fill([x_tail, x_sh, x_tip, x_sh, x_tail, x_tail], [r, r, 0, -r, -r, r], color=body_c, ec="k", lw=1)
    ax2.fill([x_sh, x_tip, x_sh], [r, 0, -r], color=nose_c, ec="k", lw=1)
    for s in (1, -1):
        ax2.fill([x_tail, x_tail + chord, x_tail + chord, x_tail], [s*r, s*r, s*(r+span), s*(r+span)],
                 color=fin_c, ec="k", lw=1, alpha=0.9)
    ax2.plot(0, 0, marker="o", ms=12, mfc="w", mec="k", zorder=5)
    ax2.plot([0, 0], [-6, 6], "k", lw=1, zorder=6); ax2.plot([-6, 6], [0, 0], "k", lw=1, zorder=6)
    ax2.annotate("CG", (0, 0), (0, 40), ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))

    def dim(x0, x1, y, txt):
        ax2.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax2.text((x0 + x1) / 2, y + 4, txt, ha="center", fontsize=9)
    dim(x_tail, x_tip, -150, f"{(x_tip - x_tail):.0f} mm overall")
    dim(0, x_tip, -128, f"{x_tip:.0f} mm tip → CG")
    dim(x_sh, x_tip, 45, f"cone {Lc:.0f} mm")
    dim(x_tail, x_tail + chord, r + span + 10, f"fin chord {chord:.0f} mm")
    ax2.annotate("", (x_tail + chord + 8, r), (x_tail + chord + 8, r + span), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax2.text(x_tail + chord + 12, r + span / 2, f"fin span {span:.0f} mm", va="center", fontsize=9)
    ax2.annotate(f"⌀ {2*r:.0f} mm", (x_sh - 60, r), (x_sh - 60, 70), ha="center", fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    ax2.text(x_tip - 40, -60, "flight direction →", fontsize=9, color="gray")
    ax2.set_aspect("equal"); ax2.set_xlim(x_tail - 20, x_tip + 25); ax2.set_ylim(-165, 140); ax2.axis("off")
    ax2.set_title("Side view (as modelled; fins are flat plates, CG fixed 200 mm behind the tip)", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) / "dart_render.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


# =============================================================================
# Subcommand: survey (fixed-altitude study)
# =============================================================================
def cmd_survey(args):
    dart = Dart(m=args.mass)
    atms = {k: Atmosphere.named(k) for k in SOLAR}
    atm = atms[args.solar]
    rhs = make_rhs(dart, atm)

    hs = np.arange(150, 1001, 10)
    k = stiffness_curve(dart, atm, hs)
    print(f"aero/GG stiffness crossover: ~{k['crossover']:.0f} km")

    p1 = {}
    for h0 in args.alts:
        y0, Torb = circ_state(h0, args.aoa0)
        sol = solve_ivp(rhs, (0, args.orbits * Torb), y0, method="DOP853", rtol=1e-9, atol=1e-9, max_step=20)
        p1[h0] = (sol_diag(sol), Torb)
        print(f"attitude run at {h0} km done")

    p2 = {}
    for name, a in atms.items():
        p2[name] = nose_first_lifetime(dart, a, args.decay_from)
        print(f"nose-first lifetime {args.decay_from:.0f}->140 km, {name}: {p2[name][0][-1]:.1f} d")

    y0, Torb = circ_state(140, 10.0)
    sol = solve_ivp(rhs, (0, 20 * Torb), y0, method="DOP853", rtol=1e-8, atol=1e-8,
                    events=ground_event, max_step=10)
    p3 = sol_diag(sol)
    print(f"reentry: {p3['t'][-1]/60:.1f} min from 140 km, impact {p3['speed'][-1]:.0f} m/s, "
          f"nose pitch {p3['nose_pitch'][-1]:.1f} deg")

    fig, ax = plt.subplots(3, 2, figsize=(13, 14))
    plot_density(ax[0, 0], atms)
    plot_stiffness(ax[0, 1], hs, k, f"Air vs gravity gradient, {dart.m*1000:.0f} g dart\n({solar_label(args.solar)})")
    a = ax[1, 0]
    cmap = plt.get_cmap("viridis")
    for i, h0 in enumerate(args.alts):
        d, T = p1[h0]
        a.plot(d["t"] / T, d["aoa"], c=cmap(i / max(1, len(args.alts) - 1)), lw=0.8, label=f"{h0:g} km")
    a.set_xlabel("orbits"); a.set_ylabel("angle between nose and relative wind (deg)")
    a.set_title(f"Attitude after release {args.aoa0:g}° off the wind")
    a.set_yticks(np.arange(-180, 181, 45)); a.legend(fontsize=8); a.grid(alpha=0.3)
    a = ax[1, 1]
    for name, (td, hd) in p2.items():
        a.plot(td, hd, c=SOLAR_COLORS[name], label=f"{solar_label(name)}: {td[-1]:.0f} d")
    a.set_xlabel("days"); a.set_ylabel("altitude (km)")
    a.set_title(f"Orbital decay {args.decay_from:g} → 140 km if flown nose-first")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    plot_reentry(ax[2, 0], ax[2, 1], p3, "Reentry from 140 km: nose follows the path\nfrom horizontal to straight down",
                 "Reentry: speed and angle of attack")
    footer(fig, dart, "")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = Path(args.out) / "survey.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


# =============================================================================
# Subcommand: run (continuous release -> ground)
# =============================================================================
def cmd_run(args):
    dart = Dart(m=args.mass)
    atm = Atmosphere.named(args.solar)
    rhs = make_rhs(dart, atm)
    y, _ = circ_state(args.release_alt, args.aoa0)
    T_all, Y_all = [0.0], [np.array(y)]
    t0, chunk, wall = 0.0, 86400.0, time.time()
    atol = [1e-1, 1e-1, 1e-4, 1e-4, 1e-8, 1e-10]
    while True:
        s = solve_ivp(rhs, (t0, t0 + chunk), y, method="DOP853", rtol=1e-8, atol=atol, events=ground_event)
        T_all.extend(s.t[1:]); Y_all.extend(s.y[:, 1:].T)
        y, t0 = s.y[:, -1], s.t[-1]
        h = (np.hypot(y[0], y[1]) - RE) / 1e3
        print(f"day {t0/86400:7.2f}  alt {h:7.2f} km  steps {len(T_all)}  wall {time.time()-wall:6.0f}s", flush=True)
        if s.status == 1:
            break
        if h < 160:
            chunk = 3600.0
        if t0 > args.max_days * 86400:
            print("stopped: --max-days reached")
            break
    out = Path(args.out) / args.file
    np.savez(out, t=np.array(T_all), y=np.array(Y_all).T, mass=dart.m,
             release_alt=args.release_alt, aoa0=args.aoa0, solar=args.solar)
    print(f"saved {out}")


# =============================================================================
# Subcommand: plot (from a run file)
# =============================================================================
WINDOWS = [(250, 1200.0), (200, 600.0), (150, 300.0), (120, 60.0), (100, 20.0)]


def cmd_plot(args):
    D = np.load(Path(args.out) / args.file)
    dart = Dart(m=float(D["mass"]))
    meta = lambda k, default: D[k] if k in D.files else default  # older run files lack metadata
    rel_alt, aoa0, solar = float(meta("release_alt", 500.0)), float(meta("aoa0", 90.0)), str(meta("solar", "moderate"))
    atms = {k: Atmosphere.named(k) for k in SOLAR}
    atm = atms[solar]
    rhs = make_rhs(dart, atm)
    full = diagnostics(D["t"], D["y"])
    t, h = full["t"], full["h"]
    Trel = 2 * np.pi * np.sqrt((RE + rel_alt * 1e3) ** 3 / MU)
    print(f"lifetime {t[-1]/86400:.1f} days, {len(t)} steps")

    # ---- attitude windows (re-integrated densely from stored states) ----
    first = (rel_alt, 4 * Trel, "orbit") if rel_alt >= 400 else (rel_alt, 1200.0, "s")
    targets = [first] + [(a, w, "s") for a, w in WINDOWS if a < rel_alt - 10]
    ncols = 2
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), squeeze=False)
    for ax, (h0, W, unit) in zip(axs.flat, targets):
        i = 0 if h0 == rel_alt else int(np.argmax(h <= h0))
        if h0 != rel_alt and h[i] > h0:
            ax.set_visible(False); continue
        s = solve_ivp(rhs, (0, W), D["y"][:, i], method="DOP853", rtol=1e-10, atol=1e-10,
                      t_eval=np.linspace(0, W, 4000), events=ground_event)
        d = sol_diag(s)
        x = d["t"] / Trel if unit == "orbit" else d["t"]
        ax.plot(x, d["aoa"], c="k", lw=0.9, label="nose vs relative wind")
        ax.plot(x, d["nose_pitch"], c="tab:blue", lw=0.8, ls="--", alpha=0.8, label="nose vs local horizon")
        for yv, c in ((-90, "tab:green"), (90, "tab:green"), (0, "gray")):
            ax.axhline(yv, c=c, lw=0.6, ls=":" if yv else "-")
        ax.set_ylim(-185, 185); ax.set_yticks(np.arange(-180, 181, 45))
        ax.set_xlabel("orbits since window start" if unit == "orbit" else "seconds since window start")
        ax.set_ylabel("degrees")
        when = "release (day 0)" if h0 == rel_alt else f"day {t[i]/86400:.1f}"
        ax.set_title(f"{h0:g} km — {when}  (alt {d['h'].max():.0f}→{d['h'].min():.0f} km in window)", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(targets):]:
        ax.set_visible(False)
    axs[0, 0].text(0.02, 0.04, "−90 = nadir, +90 = zenith (nose vs horizon)\n0 = nose into the wind",
                   transform=axs[0, 0].transAxes, fontsize=8, va="bottom", bbox=dict(fc="w", ec="0.8"))
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{dart.m*1000:.0f} g dart released at {rel_alt:g} km ({aoa0:+g}° from the wind) — "
                 f"attitude snapshots on the way down\n({solar_label(solar)})", fontsize=12)
    fig.tight_layout()
    out = Path(args.out) / "attitude_windows.png"
    fig.savefig(out, dpi=130); print(f"saved {out}")

    # ---- overview ----
    hs = np.arange(150, 1001, 10)
    k = stiffness_curve(dart, atm, hs)
    p2 = {name: nose_first_lifetime(dart, a, rel_alt) for name, a in atms.items()}
    fig, ax = plt.subplots(3, 2, figsize=(13, 14))
    plot_density(ax[0, 0], atms)
    plot_stiffness(ax[0, 1], hs, k, f"Air vs gravity gradient, {dart.m*1000:.0f} g dart\n({solar_label(solar)})",
                   mark_alt=rel_alt)
    a = ax[1, 0]
    step = max(1, len(t) // 300000)
    a.scatter(np.abs(full["aoa"][::step]), h[::step], s=0.3, c="k", alpha=0.25, rasterized=True)
    a.set_xlim(0, 180); a.set_ylim(0, rel_alt + 10); a.set_xticks(np.arange(0, 181, 30))
    a.set_xlabel("|angle between nose and relative wind| (deg)"); a.set_ylabel("altitude (km)")
    a.set_title("Whole life, release → ground: the swing\nshrinks as the air thickens"); a.grid(alpha=0.3)
    a = ax[1, 1]
    for name, (td, hd) in p2.items():
        a.plot(td, hd, c=SOLAR_COLORS[name], lw=1, ls="--",
               label=f"if flown nose-first (orbit-avg), {solar_label(name)}: {td[-1]:.0f} d")
    stp = max(1, len(t) // 20000)
    a.plot(t[::stp] / 86400, h[::stp], c="k", lw=0.6, label=f"full simulation ({solar}): {t[-1]/86400:.0f} d")
    a.set_xlim(0, min(300, max(td.max() for td, _ in p2.values())))
    a.set_xlabel("days since release"); a.set_ylabel("altitude (km)")
    a.set_title(f"Orbital decay from {rel_alt:g} km"); a.legend(fontsize=8); a.grid(alpha=0.3)
    m = h < 150
    plot_reentry(ax[2, 0], ax[2, 1], {kk: v[m] for kk, v in full.items()},
                 "Final descent below 150 km: nose follows the path\nfrom horizontal to straight down",
                 "Final descent: speed and angle of attack")
    footer(fig, dart, f"Released at {rel_alt:g} km, {aoa0:+g}° from the wind, rotating with the local vertical.")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = Path(args.out) / "overview.png"
    fig.savefig(out, dpi=130); print(f"saved {out}")
    print(f"crossover ~{k['crossover']:.0f} km; impact {full['speed'][-1]:.0f} m/s, "
          f"final nose pitch {full['nose_pitch'][-1]:.1f} deg")


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="output directory (default: .)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, release=True):
        sp.add_argument("--mass", type=float, default=0.1, help="dart mass, kg (default 0.1)")
        sp.add_argument("--solar", choices=list(SOLAR), default="moderate", help="solar activity level")
        if release:
            sp.add_argument("--release-alt", type=float, default=500.0, help="release altitude, km (default 500)")
            sp.add_argument("--aoa0", type=float, default=90.0,
                            help="initial nose angle from the wind, deg: 90 = nadir, -90 = zenith, 0 = prograde")
        sp.add_argument("--file", default="continuous_run.npz", help="run data file (default continuous_run.npz)")

    sp = sub.add_parser("render", help="draw the dart geometry")
    sp.add_argument("--mass", type=float, default=0.1)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("survey", help="fixed-altitude attitude study + lifetime + reentry from 140 km")
    sp.add_argument("--mass", type=float, default=0.1)
    sp.add_argument("--solar", choices=list(SOLAR), default="moderate")
    sp.add_argument("--alts", type=float, nargs="+", default=[300, 500, 700], help="altitudes for attitude runs, km")
    sp.add_argument("--aoa0", type=float, default=30.0, help="initial nose angle from the wind, deg")
    sp.add_argument("--orbits", type=float, default=6, help="orbits per attitude run")
    sp.add_argument("--decay-from", type=float, default=300.0, help="start altitude for lifetime curves, km")
    sp.set_defaults(func=cmd_survey)

    sp = sub.add_parser("run", help="continuous run from release to the ground")
    common(sp)
    sp.add_argument("--max-days", type=float, default=400, help="stop after this many days")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("plot", help="figures from a run file")
    sp.add_argument("--file", default="continuous_run.npz")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("all", help="render + run + plot")
    common(sp)
    sp.add_argument("--max-days", type=float, default=400)
    sp.set_defaults(func=lambda a: (cmd_render(a), cmd_run(a), cmd_plot(a)))

    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
