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
through reentry, using NRLMSIS 2.0 (via pymsis) for the atmosphere, with a
coupled aerothermal model of the fins.

Subcommands
-----------
  render   Draw the dart geometry (3D view, dimensioned side view, front view).
  survey   Fixed-altitude attitude runs, aero vs gravity-gradient stiffness,
           orbit-averaged lifetime, and a reentry from 140 km.
  run      Continuous orbit + attitude + fin-thermal run from release to the
           ground (saves every integrator step to an .npz file).
  plot     Overview, attitude-snapshot and heating figures from a run file.
  video    MP4 of the descent with live indicators (needs ffmpeg on PATH).
  all      render + run + plot + video.

Examples
--------
  uv run vleo_dart_sim.py render
  uv run vleo_dart_sim.py run --mass 0.1 --release-alt 500 --aoa0 90 --fin-material balsa
  uv run vleo_dart_sim.py plot
  uv run vleo_dart_sim.py video
  uv run vleo_dart_sim.py --out results all

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
        cone-cylinder with 4 cruciform 3 mm thick fins (faces + blunt
        leading/trailing edges).
      - Continuum: Barrowman-style normal-force build-up (nose + fins + body
        crossflow), Mach-dependent axial drag, blunt fin leading-edge drag.
      - Transitional: blended by Knudsen number (Kn 0.01 .. 10).
      - Local velocity at each panel includes omega x r, so aerodynamic
        pitch damping emerges naturally. No shadowing between panels.
  * Heating:
      - Free-molecular: incident energy flux per panel (full energy
        accommodation), including thermal motion.
      - Continuum: Sutton-Graves stagnation heating (nose tip radius 3 mm
        assumed; fin leading edge = 1.5 mm radius cylinder), laminar
        flat-plate heating on fin faces plus an incidence term, capped at the
        free-stream kinetic energy flux. Bridged by local Knudsen number.
      - Fins: one lumped thermal mass (all four fins), radiating to a 250 K
        environment; once at the material's failure temperature, net heat
        consumes fin material (effective heat of ablation). Lost fin area is
        fed back into the aerodynamics. Total vehicle mass is held constant.
      - Nose: radiative-equilibrium temperature only (no material response).
  * Attitude convention: aoa = angle from relative wind to nose, positive
    counter-clockwise. In the (counter-clockwise) orbit, aoa = +90 is nadir,
    -90 is zenith, 0 is nose into the wind.
"""
from __future__ import annotations

import argparse
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pymsis
from scipy.integrate import solve_ivp
from scipy.special import erf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
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
SIGMA = 5.670374e-8      # Stefan-Boltzmann
K_SG = 1.7415e-4         # Sutton-Graves constant for Earth air (SI)
T_ENV = 250.0            # radiative sink temperature, K
T_FIN0 = 280.0           # fin temperature at release, K
PR = 0.71
EPS_NOSE = 0.8

# Fin materials: density kg/m3, specific heat J/kg/K, emissivity, failure
# temperature K (melt / pyrolysis / resin decomposition), effective heat to
# consume material once at the failure temperature J/kg (very rough).
MATERIALS = {
    "balsa":    dict(rho=160,  c=2000, eps=0.85, T_fail=573,  H=3.0e6, label="balsa (chars ~300 °C)"),
    "g10":      dict(rho=1850, c=1000, eps=0.90, T_fail=600,  H=2.0e6, label="G10 fibreglass (resin ~330 °C)"),
    "fr4":      dict(rho=1900, c=1150, eps=0.90, T_fail=588,  H=2.0e6, label="FR4 fibreglass (resin ~315 °C)"),
    "aluminium": dict(rho=2700, c=900,  eps=0.20, T_fail=855,  H=4.0e5, label="aluminium 6061 (melts ~580 °C)"),
    "titanium": dict(rho=4430, c=560,  eps=0.40, T_fail=1878, H=3.7e5, label="Ti-6Al-4V (melts ~1600 °C)"),
    "steel":    dict(rho=8000, c=500,  eps=0.40, T_fail=1673, H=2.7e5, label="stainless steel (melts ~1400 °C)"),
}
MAT_COLORS = {"balsa": "tab:brown", "g10": "tab:olive", "aluminium": "tab:gray",
              "titanium": "tab:purple", "steel": "k"}

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
    x 3 mm tail fins. Body frame: x toward nose, origin at CG."""

    def __init__(self, m=0.1, fin_t=0.003, nose_r=0.003):
        self.m = m
        self.L, self.D = 0.50, 0.05
        self.r = self.D / 2
        self.Lc = 0.10
        self.x_tip = 0.20
        self.x_tail = self.x_tip - self.L
        self.fin_span, self.fin_chord, self.fin_t = 0.08, 0.10, fin_t
        self.nose_r = nose_r                       # assumed nose-tip radius (heating)
        self.le_r = fin_t / 2                      # fin leading-edge radius
        self.A_ref = np.pi * self.r ** 2
        self.I_t = self.m * self.L ** 2 / 12 + self.m * (self.L / 2 - self.x_tip) ** 2
        self.I_a = 0.5 * self.m * self.r ** 2
        s, c, t = self.fin_span, self.fin_chord, fin_t
        self.A_face = s * c
        self.A_le_total = 4 * s * t
        self.fin_volume = 4 * s * c * t
        self.fin_A_rad = 4 * (2 * s * c + 2 * s * t + c * t)
        self.cone_area = np.pi * self.r * np.hypot(self.r, self.Lc)
        self.A_wet_body = np.pi * self.D * (self.L - self.Lc) + self.cone_area
        self._build_panels()
        # continuum (Barrowman-style) component locations
        self.x_nose_cp = self.x_tip - 0.666 * self.Lc
        self.x_cyl = 0.5 * (self.x_tip - self.Lc + self.x_tail)
        self.L_cyl = self.L - self.Lc
        self.x_fin = self.x_tail + c / 2
        self.A_fin_pair = 2 * s * c
        self.fin_AR = (2 * s) ** 2 / self.A_fin_pair
        self.K_int = 1 + self.r / (s + self.r)

    def fin_mass(self, material):
        return MATERIALS[material]["rho"] * self.fin_volume

    def _build_panels(self):
        P_r, P_n, P_A, fin = [], [], [], []

        def add(r, n, A, is_fin=False):
            P_r.append(list(r)); P_n.append(list(n)); P_A.append(A); fin.append(is_fin)

        nf = 12
        phis = (np.arange(nf) + 0.5) * 2 * np.pi / nf
        ha = np.arctan(self.r / self.Lc)
        slant = np.hypot(self.r, self.Lc)
        xc = self.x_tip - 2 / 3 * self.Lc
        for p in phis:  # nose cone facets
            add([xc, 2 / 3 * self.r * np.cos(p), 2 / 3 * self.r * np.sin(p)],
                [np.sin(ha), np.cos(ha) * np.cos(p), np.cos(ha) * np.sin(p)], np.pi * self.r * slant / nf)
        x0, x1 = self.x_tail, self.x_tip - self.Lc
        nseg = 8
        segL = (x1 - x0) / nseg
        for k in range(nseg):  # cylinder facets
            xs = x0 + (k + 0.5) * segL
            for p in phis:
                add([xs, self.r * np.cos(p), self.r * np.sin(p)], [0, np.cos(p), np.sin(p)],
                    2 * np.pi * self.r * segL / nf)
        add([self.x_tail, 0, 0], [-1, 0, 0], np.pi * self.r ** 2)  # base
        # 4 fins: two faces + blunt leading edge + trailing edge each.
        # Fins spanning +/-z have face normals +/-y (pitch-active).
        s, c, t = self.fin_span, self.fin_chord, self.fin_t
        xf = self.x_tail + c / 2
        off = self.r + s / 2
        for span_dir, nrm in (([0, 0, 1], [0, 1, 0]), ([0, 0, -1], [0, 1, 0]),
                              ([0, 1, 0], [0, 0, 1]), ([0, -1, 0], [0, 0, 1])):
            ctr = np.array([xf, 0, 0]) + off * np.array(span_dir)
            nrm = np.array(nrm, float)
            for sgn in (1, -1):
                add(ctr + sgn * nrm * t / 2, sgn * nrm, s * c, True)
            add(ctr + [c / 2, 0, 0], [1, 0, 0], s * t, True)    # leading edge
            add(ctr - [c / 2, 0, 0], [-1, 0, 0], s * t, True)   # trailing edge
        self.P_r = np.array(P_r, float)
        self.P_n = np.array(P_n, float)
        self.P_A = np.array(P_A, float)
        self.fin_mask = np.array(fin, bool)


# =============================================================================
# Aerodynamics and heating
# =============================================================================
def sutherland_mu(T):
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def cd0_axial(M):
    return np.interp(M, [0.0, 0.8, 1.05, 1.3, 2.0, 5.0, 30.0],
                     [0.45, 0.50, 0.95, 0.85, 0.70, 0.55, 0.50])


def cd_fin_le(M):
    """Blunt (3 mm) fin leading-edge drag coefficient on its frontal area."""
    return np.interp(M, [0.0, 0.8, 1.2, 3.0, 30.0], [0.5, 0.6, 1.2, 1.25, 1.2])


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


def fm_weight(n, Lref=0.5):
    """Free-molecular weight from Knudsen number (1 = free molecular)."""
    lam = 1 / (np.sqrt(2) * np.pi * D_MOL ** 2 * n)
    x = np.log10(lam / Lref)
    return float(np.clip(0.5 * (1 + np.sin(np.pi / 2 * np.clip((x + 0.5) / 1.5, -1, 1))), 0, 1))


def free_molecular(dart, v_body, omega, rho, n, T, scale):
    """Sentman forces/torque and incident energy flux per panel (body frame).
    scale = remaining fraction of fin material (scales fin panel areas)."""
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
    Z = 1 + erf(g * S)
    A = np.where(dart.fin_mask, dart.P_A * scale, dart.P_A)
    q = 0.5 * rho * V ** 2
    cD = P / np.sqrt(np.pi) + g * (1 + G) * Z
    reem = 0.5 * vre * (g * np.sqrt(np.pi) * Z + P)
    F = (q * A)[:, None] * (cD[:, None] * uD + (G * Z)[:, None] * (-nrm - g[:, None] * uD)
                            - reem[:, None] * nrm)
    heat = 0.5 * (g * Z + P / np.sqrt(np.pi)) * 0.5 * rho * V ** 3 * A   # W per panel
    return F.sum(0), np.cross(dart.P_r, F)[:, 2].sum(), heat


def continuum_forces(dart, v_body, omega, rho, n, T, scale):
    a_snd = np.sqrt(1.4 * KB * T / (rho / n))
    F = np.zeros(3)
    tau = 0.0

    def local(x):
        vl = v_body + np.cross([0, 0, omega], [x, 0, 0])
        V = np.linalg.norm(vl)
        u = -vl / V
        return V, np.arctan2(u[1], -u[0]), u

    V, al, u = local(0.0)  # axial drag: body + blunt fin leading edges
    M = V / a_snd
    q = 0.5 * rho * V ** 2
    F[0] += q * (dart.A_ref * cd0_axial(M) + scale * dart.A_le_total * cd_fin_le(M)) * np.sign(u[0])
    V, al, u = local(dart.x_nose_cp)  # nose (slender body CNa = 2)
    fy = 0.5 * rho * V ** 2 * dart.A_ref * 2 * np.sin(al) * np.cos(al)
    F[1] += fy; tau += dart.x_nose_cp * fy
    V, al, u = local(dart.x_cyl)  # body crossflow
    fy = 0.5 * rho * V ** 2 * 1.2 * dart.D * dart.L_cyl * np.sin(al) * abs(np.sin(al))
    F[1] += fy; tau += dart.x_cyl * fy
    V, al, u = local(dart.x_fin)  # fins
    cn = (dart.K_int * fin_cna(dart, V / a_snd) * np.sin(al) * np.cos(al)
          + 1.28 * np.sin(al) * abs(np.sin(al)))
    fy = 0.5 * rho * V ** 2 * dart.A_fin_pair * scale * cn
    F[1] += fy; tau += dart.x_fin * fy
    return F, tau


def continuum_heating(dart, v_body, rho, T, scale):
    """Continuum heat fluxes (W/m2) and powers (W), each capped at the
    free-stream kinetic energy flux."""
    V = np.linalg.norm(v_body)
    alpha = np.arctan2(abs(v_body[1]), v_body[0])  # angle between nose and velocity
    qke = 0.5 * rho * V ** 3
    mu = sutherland_mu(T)
    q_nose = min(K_SG * np.sqrt(rho / dart.nose_r) * V ** 3, qke)
    q_le = min(K_SG * np.sqrt(rho / dart.le_r) * V ** 3 / np.sqrt(2), qke)
    st = lambda L: 0.664 / np.sqrt(max(rho * V * L / mu, 1e-9)) * PR ** (-2 / 3)
    q_face = min(st(dart.fin_chord) * qke, qke)
    q_inc = min(K_SG * np.sqrt(rho / dart.fin_chord) * V ** 3 * abs(np.sin(alpha)) ** 1.5, qke)
    q_body = min(st(dart.L) * qke, qke)
    P_fins = scale * (4 * 0.6 * q_le * np.pi * dart.le_r * dart.fin_span
                      + 8 * dart.A_face * q_face + 2 * dart.A_face * q_inc)
    P_body = q_body * dart.A_wet_body + q_nose * np.pi * dart.nose_r ** 2
    return dict(q_nose=q_nose, q_le=q_le, P_fins=P_fins, P_body=P_body, qke=qke)


def aerothermal(dart, v_body, omega, rho, n, T, scale=1.0, heat=True):
    """Blended forces (body frame), pitch torque, and (optionally) heating."""
    w = fm_weight(n, dart.L)
    w_fin = fm_weight(n, dart.fin_chord)
    F = np.zeros(3); tau = 0.0
    need_fm = w > 0 or (heat and w_fin > 0)
    if need_fm:
        f_fm, t_fm, h_fm = free_molecular(dart, v_body, omega, rho, n, T, scale)
        F += w * f_fm; tau += w * t_fm
    if w < 1:
        f_c, t_c = continuum_forces(dart, v_body, omega, rho, n, T, scale)
        F += (1 - w) * f_c; tau += (1 - w) * t_c
    if not heat:
        return F, tau, None
    ch = continuum_heating(dart, v_body, rho, T, scale)
    w_n, w_le = fm_weight(n, dart.nose_r), fm_weight(n, dart.le_r)
    P_fins_fm = h_fm[dart.fin_mask].sum() if need_fm else 0.0
    P_all_fm = h_fm.sum() if need_fm else 0.0
    H = dict(
        q_nose=w_n * ch["qke"] + (1 - w_n) * ch["q_nose"],
        q_le=w_le * ch["qke"] + (1 - w_le) * ch["q_le"],
        P_fins=w_fin * P_fins_fm + (1 - w_fin) * ch["P_fins"],
        P_total=w * P_all_fm + (1 - w) * (ch["P_fins"] + ch["P_body"]),
        qke=ch["qke"], w=w)
    return F, tau, H


def aero(dart, v_body, omega, rho, n, T, scale=1.0):
    F, tau, _ = aerothermal(dart, v_body, omega, rho, n, T, scale, heat=False)
    return F, tau


def fin_thermal_rates(dart, mat, Tf, f, P_in):
    """d(T_fin)/dt and d(consumed fraction)/dt for the lumped fin mass."""
    scale = 1.0 - min(max(f, 0.0), 1.0)
    if scale <= 1e-6:
        return 0.0, 0.0
    m0 = mat["rho"] * dart.fin_volume
    net = P_in - mat["eps"] * SIGMA * dart.fin_A_rad * scale * (Tf ** 4 - T_ENV ** 4)
    if Tf >= mat["T_fail"] and net > 0:
        return 0.0, net / (m0 * mat["H"])
    return net / (m0 * scale * mat["c"]), 0.0


# =============================================================================
# Dynamics
# =============================================================================
def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def local_state(dart, atm, y):
    """Atmosphere + body-frame relative velocity for one state vector."""
    X, Y, VX, VY, psi, om = y[:6]
    rn = np.hypot(X, Y)
    rho, n, T = atm((rn - RE) / 1e3, np.degrees(np.arctan2(Y, X)))
    v_rel = np.array([VX + WE * Y, VY - WE * X, 0.0])
    return rho, n, T, rot(psi).T @ v_rel


def make_rhs(dart, atm, material="balsa"):
    """State: [X, Y, VX, VY, psi, omega, T_fin, fin_consumed_fraction]."""
    mat = MATERIALS[material]

    def rhs(t, y):
        X, Y, VX, VY, psi, om, Tf, f = y
        f = min(max(f, 0.0), 1.0)
        r = np.array([X, Y, 0.0]); rn = np.linalg.norm(r)
        rho, n, T, v_b = local_state(dart, atm, y)
        Rb = rot(psi)
        Fb, tau, H = aerothermal(dart, v_b, om, rho, n, T, 1.0 - f, heat=True)
        acc = -MU * r / rn ** 3 + (Rb @ Fb) / dart.m
        r_b = Rb.T @ r
        tau_gg = 3 * MU / rn ** 5 * r_b[0] * r_b[1] * (dart.I_t - dart.I_a)
        dT, df = fin_thermal_rates(dart, mat, Tf, f, H["P_fins"])
        return [VX, VY, acc[0], acc[1], om, (tau + tau_gg) / dart.I_t, dT, df]
    return rhs


ATOL = [1e-1, 1e-1, 1e-4, 1e-4, 1e-8, 1e-10, 1e-3, 1e-9]


def ground_event(t, y):
    return np.hypot(y[0], y[1]) - RE
ground_event.terminal = True


def circ_state(h_km, aoa0_deg):
    """Circular orbit at h_km, nose at aoa0 from the wind, rotating with the
    local vertical, fins intact at T_FIN0. Returns (state, orbital period)."""
    r = RE + h_km * 1e3
    v = np.sqrt(MU / r)
    n = v / r
    return [r, 0.0, 0.0, v, np.pi / 2 + np.radians(aoa0_deg), n, T_FIN0, 0.0], 2 * np.pi / n


def diagnostics(t, y):
    X, Y, VX, VY, psi, om = y[:6]
    rn = np.hypot(X, Y)
    up = np.stack([X, Y]) / rn
    hor = np.stack([-up[1], up[0]])
    v = np.stack([VX + WE * Y, VY - WE * X])
    nose = np.stack([np.cos(psi), np.sin(psi)])
    vhat = v / np.linalg.norm(v, axis=0)
    d = dict(
        t=t, h=(rn - RE) / 1e3,
        nose_pitch=np.degrees(np.arctan2((nose * up).sum(0), (nose * hor).sum(0))),
        fpa=np.degrees(np.arctan2((v * up).sum(0), (v * hor).sum(0))),
        aoa=np.degrees(np.arctan2(vhat[0] * nose[1] - vhat[1] * nose[0], (vhat * nose).sum(0))),
        speed=np.linalg.norm(v, axis=0))
    if y.shape[0] >= 8:
        d["T_fin"], d["fin_left"] = y[6], 1 - np.clip(y[7], 0, 1)
    return d


def sol_diag(sol):
    return diagnostics(sol.t, sol.y)


def state_props(dart, atm, y, material):
    """Everything the video/plots show for one state vector."""
    rho, n, T, v_b = local_state(dart, atm, y)
    f = float(np.clip(y[7], 0, 1)) if len(y) >= 8 else 0.0
    _, _, H = aerothermal(dart, v_b, y[5], rho, n, T, 1 - f, heat=True)
    V = np.linalg.norm(v_b)
    lam = 1 / (np.sqrt(2) * np.pi * D_MOL ** 2 * n)
    eps = MATERIALS[material]["eps"]
    return dict(
        rho=rho, T_air=T, mach=V / np.sqrt(1.4 * KB * T / (rho / n)), Kn=lam / dart.L,
        qdyn=0.5 * rho * V ** 2, q_nose=H["q_nose"], q_le=H["q_le"], P_fins=H["P_fins"],
        P_total=H["P_total"], qke=H["qke"], w=H["w"],
        T_nose_eq=(H["q_nose"] / (EPS_NOSE * SIGMA) + T_ENV ** 4) ** 0.25,
        T_le_eq=(H["q_le"] / (eps * SIGMA) + T_ENV ** 4) ** 0.25)


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
    t_mm = d.fin_t * mm
    for dy, dz in ((0, 1), (0, -1), (1, 0), (-1, 0)):   # fins as 3 mm thick boxes
        e = np.array([0, dy, dz], float)                # span direction
        nn = np.array([0, dz, dy], float)               # face normal
        p0 = np.array([x_tail, 0, 0]) + r * e - nn * t_mm / 2
        a, b, c = np.array([chord, 0, 0]), span * e, nn * t_mm
        faces = [([p0, p0 + a, p0 + a + b, p0 + b], -nn), ([p0 + c, p0 + a + c, p0 + a + b + c, p0 + b + c], nn),
                 ([p0 + a, p0 + a + c, p0 + a + b + c, p0 + a + b], np.array([1, 0, 0])),
                 ([p0, p0 + c, p0 + b + c, p0 + b], np.array([-1, 0, 0])),
                 ([p0 + b, p0 + a + b, p0 + a + b + c, p0 + b + c], e)]
        for f, fn in faces:
            polys.append([list(q) for q in f]); cols.append(shade(fin_c, fn))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.15), linewidths=0.3))
    ax.set_xlim(x_tail - 30, x_tip + 30); ax.set_ylim(-110, 110); ax.set_zlim(-110, 110)
    ax.set_box_aspect((x_tip - x_tail + 60, 220, 220), zoom=1.05)
    ax.view_init(elev=22, azim=-58); ax.set_axis_off()
    ax.set_title(f"Simulated dart: {d.m*1000:.0f} g, 0.5 m cone-cylinder with four "
                 f"{d.fin_t*1e3:.0f} mm thick tail fins", fontsize=13)

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
    ax2.set_aspect("equal"); ax2.set_xlim(x_tail - 20, x_tip + 25); ax2.set_ylim(-185, 140); ax2.axis("off")
    ax2.set_title("Side view (as modelled; CG fixed 200 mm behind the tip)", fontsize=11)
    ax2.text(x_tail, -178, f"fins {d.fin_t*1e3:.0f} mm thick, blunt leading edge "
             f"(r = {d.le_r*1e3:.1f} mm); nose tip radius assumed {d.nose_r*1e3:.0f} mm for heating",
             fontsize=9, color="0.3")
    fig.tight_layout()
    ax3 = fig.add_axes([0.80, 0.07, 0.17, 0.27])       # front view (looking aft from the nose)
    ax3.add_patch(Circle((0, 0), r, fc=body_c, ec="k"))
    ax3.add_patch(Circle((0, 0), d.nose_r * mm * 2, fc=nose_c, ec="k", lw=0.5))
    for ang in (0, 90, 180, 270):
        R = rot(np.radians(ang))[:2, :2]
        box = np.array([[r, -t_mm / 2], [r + span, -t_mm / 2], [r + span, t_mm / 2], [r, t_mm / 2]]) @ R.T
        ax3.add_patch(Polygon(box, fc=fin_c, ec="k", lw=0.5))
    ax3.set_xlim(-r - span - 5, r + span + 5); ax3.set_ylim(-r - span - 5, r + span + 5)
    ax3.set_aspect("equal"); ax3.axis("off"); ax3.set_title("front view", fontsize=9)
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
    rhs = make_rhs(dart, atm, args.fin_material)

    hs = np.arange(150, 1001, 10)
    k = stiffness_curve(dart, atm, hs)
    print(f"aero/GG stiffness crossover: ~{k['crossover']:.0f} km")

    p1 = {}
    for h0 in args.alts:
        y0, Torb = circ_state(h0, args.aoa0)
        sol = solve_ivp(rhs, (0, args.orbits * Torb), y0, method="DOP853", rtol=1e-9, atol=ATOL, max_step=20)
        p1[h0] = (sol_diag(sol), Torb)
        print(f"attitude run at {h0} km done")

    p2 = {}
    for name, a in atms.items():
        p2[name] = nose_first_lifetime(dart, a, args.decay_from)
        print(f"nose-first lifetime {args.decay_from:.0f}->140 km, {name}: {p2[name][0][-1]:.1f} d")

    y0, Torb = circ_state(140, 10.0)
    sol = solve_ivp(rhs, (0, 20 * Torb), y0, method="DOP853", rtol=1e-8, atol=ATOL,
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
    plot_reentry(ax[2, 0], ax[2, 1], p3, "Reentry from 140 km:\nnose attitude vs flight path",
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
    rhs = make_rhs(dart, atm, args.fin_material)
    y, _ = circ_state(args.release_alt, args.aoa0)
    print(f"fins: {MATERIALS[args.fin_material]['label']}, {dart.fin_mass(args.fin_material)*1e3:.0f} g "
          f"(vehicle mass held at {dart.m*1e3:.0f} g)")
    T_all, Y_all = [0.0], [np.array(y)]
    t0, chunk, wall = 0.0, 86400.0, time.time()
    while True:
        s = solve_ivp(rhs, (t0, t0 + chunk), y, method="DOP853", rtol=1e-8, atol=ATOL, events=ground_event)
        T_all.extend(s.t[1:]); Y_all.extend(s.y[:, 1:].T)
        y, t0 = s.y[:, -1], s.t[-1]
        h = (np.hypot(y[0], y[1]) - RE) / 1e3
        print(f"day {t0/86400:7.2f}  alt {h:7.2f} km  fin {y[6]:6.0f} K  fin left {100*(1-min(y[7],1)):5.1f}%  "
              f"steps {len(T_all)}  wall {time.time()-wall:6.0f}s", flush=True)
        if s.status == 1:
            break
        if h < 160:
            chunk = 3600.0
        if t0 > args.max_days * 86400:
            print("stopped: --max-days reached")
            break
    out = Path(args.out) / args.file
    np.savez(out, t=np.array(T_all), y=np.array(Y_all).T, mass=dart.m,
             release_alt=args.release_alt, aoa0=args.aoa0, solar=args.solar,
             material=args.fin_material)
    print(f"saved {out}")


class Run:
    """A saved continuous run plus everything needed to re-integrate it."""

    def __init__(self, path):
        D = np.load(path)
        meta = lambda k, default: D[k] if k in D.files else default  # older files lack metadata
        self.t = D["t"]
        y = D["y"]
        if y.shape[0] == 6:   # pre-heating run file: assume intact fins
            y = np.vstack([y, np.full(len(self.t), T_ENV), np.zeros(len(self.t))])
        self.y = y
        self.dart = Dart(m=float(D["mass"]))
        self.rel_alt = float(meta("release_alt", 500.0))
        self.aoa0 = float(meta("aoa0", 90.0))
        self.solar = str(meta("solar", "moderate"))
        self.material = str(meta("material", "balsa"))
        self.atms = {k: Atmosphere.named(k) for k in SOLAR}
        self.atm = self.atms[self.solar]
        self.rhs = make_rhs(self.dart, self.atm, self.material)
        self.full = diagnostics(self.t, self.y)
        self.h = self.full["h"]
        self.T_rel = 2 * np.pi * np.sqrt((RE + self.rel_alt * 1e3) ** 3 / MU)

    def t_at_alt(self, a):
        i = int(np.argmax(self.h <= a))
        return float(self.t[i]) if self.h[i] <= a else None

    def states_at(self, ts):
        """States at arbitrary (sorted) times, re-integrated from stored steps."""
        ts = np.asarray(ts, float)
        out = np.zeros((self.y.shape[0], len(ts)))
        idx = np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.t) - 1)
        k = 0
        while k < len(ts):
            i = idx[k]
            j = k
            while j + 1 < len(ts) and idx[j + 1] == i:
                j += 1
            grp = ts[k:j + 1]
            if i == len(self.t) - 1 or grp[-1] <= self.t[i]:
                out[:, k:j + 1] = self.y[:, i][:, None]
            else:
                s = solve_ivp(self.rhs, (self.t[i], grp[-1]), self.y[:, i], method="DOP853", rtol=1e-9,
                              atol=ATOL, t_eval=np.clip(grp, self.t[i], grp[-1]))
                out[:, k:k + s.y.shape[1]] = s.y
                if s.y.shape[1] < len(grp):           # hit the ground: pad
                    out[:, k + s.y.shape[1]:j + 1] = s.y[:, -1][:, None]
            k = j + 1
        return out


def trajectory_heating(R, h_max=200.0, max_points=None):
    """Heating quantities at the stored steps below h_max (optionally thinned)."""
    sel = np.where(R.h < h_max)[0]
    if max_points and len(sel) > max_points:
        sel = sel[np.linspace(0, len(sel) - 1, max_points).astype(int)]
    rows = [state_props(R.dart, R.atm, R.y[:, i], R.material) for i in sel]
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(t=R.t[sel], h=R.h[sel], T_fin=R.y[6, sel], fin_left=1 - np.clip(R.y[7, sel], 0, 1))
    # fin heating per unit remaining fraction, for material replays
    out["P_fins_unit"] = out["P_fins"] / np.maximum(out["fin_left"], 1e-6)
    return out


def cumtrapz(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def replay_fin_thermal(dart, material, t, P_unit, T0=T_ENV):
    """Lumped fin temperature / remaining fraction along a fixed trajectory
    (uncoupled: the trajectory does not respond). Semi-implicit radiation."""
    mat = MATERIALS[material]
    m0 = mat["rho"] * dart.fin_volume
    Tf, f = T0, 0.0
    Ts, Ls = [Tf], [1.0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        scale = 1 - f
        if scale > 1e-6:
            P_in = P_unit[k] * scale
            ra = mat["eps"] * SIGMA * dart.fin_A_rad * scale
            C = m0 * scale * mat["c"]
            net = P_in - ra * (Tf ** 4 - T_ENV ** 4)
            if Tf >= mat["T_fail"] and net > 0:
                f = min(1.0, f + net * dt / (m0 * mat["H"]))
            else:
                Tf = Tf + dt * net / (C + dt * 4 * ra * Tf ** 3)
                if Tf > mat["T_fail"]:
                    excess = (Tf - mat["T_fail"]) * C
                    Tf = mat["T_fail"]
                    f = min(1.0, f + excess / (m0 * mat["H"]))
        Ts.append(Tf); Ls.append(1 - f)
    return np.array(Ts), np.array(Ls)


def heating_figure(R, H, out_path):
    dart = R.dart
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    hmax = 160
    a = ax[0, 0]
    a.semilogx(H["qke"], H["h"], c="0.6", lw=0.8, ls=":", label="free-stream kinetic energy flux ½ρV³")
    a.semilogx(H["q_nose"], H["h"], c="tab:red", lw=1, label=f"nose stagnation (r = {dart.nose_r*1e3:.0f} mm)")
    a.semilogx(H["q_le"], H["h"], c="tab:orange", lw=1, label=f"fin leading edge (r = {dart.le_r*1e3:.1f} mm)")
    i = int(np.argmax(H["q_nose"]))
    a.annotate(f"peak {H['q_nose'][i]/1e6:.2f} MW/m²\nat {H['h'][i]:.0f} km", (H["q_nose"][i], H["h"][i]),
               (H["q_nose"][i] / 300, H["h"][i] + 25), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.6))
    a.set_xlim(1e-1, None); a.set_ylim(0, hmax)
    a.set_xlabel("heat flux (W/m²)"); a.set_ylabel("altitude (km)")
    a.set_title("Heat flux along the trajectory"); a.legend(fontsize=8, loc="lower left"); a.grid(alpha=0.3, which="both")

    a = ax[0, 1]
    a.plot(H["T_nose_eq"] - 273.15, H["h"], c="tab:red", lw=1, label="nose tip, radiative equilibrium")
    a.plot(H["T_le_eq"] - 273.15, H["h"], c="tab:orange", lw=1, label="fin leading edge, radiative equilibrium")
    for m, mat in MATERIALS.items():
        a.axvline(mat["T_fail"] - 273.15, c=MAT_COLORS[m], lw=0.8, ls="--")
        a.text(mat["T_fail"] - 273.15, hmax - 3, " " + m, rotation=90, va="top", fontsize=8, color=MAT_COLORS[m])
    a.set_ylim(0, hmax); a.set_xlabel("temperature (°C)"); a.set_ylabel("altitude (km)")
    a.set_title("Local surface temperatures if each spot could radiate its heat away\n"
                "(upper bound for thin edges/tips; dashed = material limits)")
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3)

    a, b = ax[1, 0], ax[1, 1]
    for m in MATERIALS:
        if m == R.material:
            Tm, Lm = H["T_fin"], H["fin_left"]
            kw = dict(lw=2, ls="-")
            lab = f"{MATERIALS[m]['label']}, {dart.fin_mass(m)*1e3:.0f} g — coupled (this run)"
        else:
            Tm, Lm = replay_fin_thermal(dart, m, H["t"], H["P_fins_unit"], T0=H["T_fin"][0])
            kw = dict(lw=1, ls="--")
            lab = f"{MATERIALS[m]['label']}, {dart.fin_mass(m)*1e3:.0f} g — replay"
        Tm = np.where(Lm > 0.001, Tm, np.nan)          # nothing left to have a temperature
        a.plot(Tm - 273.15, H["h"], c=MAT_COLORS[m], label=lab, **kw)
        b.plot(100 * Lm, H["h"], c=MAT_COLORS[m], label=m, **kw)
    a.set_ylim(0, hmax); a.set_xlabel("lumped fin temperature (°C)"); a.set_ylabel("altitude (km)")
    a.set_title("Bulk fin temperature by material\n(solid: coupled run; dashed: replayed on the same trajectory)")
    a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3)
    b.set_ylim(0, hmax); b.set_xlim(-3, 103); b.set_xlabel("fin material remaining (%)"); b.set_ylabel("altitude (km)")
    b.set_title("Fin survival"); b.legend(fontsize=8, loc="lower left"); b.grid(alpha=0.3)
    E_tot = cumtrapz(H["P_total"], H["t"])[-1]
    E_fin = cumtrapz(H["P_fins"], H["t"])[-1]
    Q_nose = cumtrapz(H["q_nose"], H["t"])[-1]
    txt = (f"Heat loads below 200 km\n"
           f"  nose stagnation: {Q_nose/1e6:.1f} MJ/m²\n"
           f"  into the whole vehicle: {E_tot/1e3:.0f} kJ\n"
           f"  into the fins: {E_fin/1e3:.0f} kJ\n"
           f"  (vehicle KE at 200 km ≈ {0.5*dart.m*7790**2/1e3:.0f} kJ)")
    b.text(0.97, 0.97, txt, transform=b.transAxes, ha="right", va="top", fontsize=8.5, family="monospace",
           bbox=dict(fc="w", ec="0.7"))
    fig.suptitle(f"Aerothermal environment — {dart.m*1e3:.0f} g dart, {dart.fin_t*1e3:.0f} mm fins, "
                 f"released at {R.rel_alt:g} km ({solar_label(R.solar)})", fontsize=12)
    footer(fig, dart, "Fins: one lumped thermal mass radiating to 250 K; consumed at the failure temperature. "
                      "Nose: radiative equilibrium only. Very rough.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    print(f"peak nose heat flux {H['q_nose'].max()/1e6:.2f} MW/m² at {H['h'][np.argmax(H['q_nose'])]:.0f} km; "
          f"peak fin LE {H['q_le'].max()/1e6:.2f} MW/m²; fin max {H['T_fin'].max()-273.15:.0f} °C; "
          f"fin left at impact {100*H['fin_left'][-1]:.0f}%")


# =============================================================================
# Subcommand: plot (from a run file)
# =============================================================================
WINDOWS = [(250, 1200.0), (200, 600.0), (150, 300.0), (120, 60.0), (100, 20.0)]


def cmd_plot(args):
    R = Run(Path(args.out) / args.file)
    dart, atm, atms, rhs = R.dart, R.atm, R.atms, R.rhs
    rel_alt, aoa0, solar = R.rel_alt, R.aoa0, R.solar
    full, t, h, Trel = R.full, R.t, R.h, R.T_rel
    print(f"lifetime {t[-1]/86400:.1f} days, {len(t)} steps, fins: {R.material}")

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
        s = solve_ivp(rhs, (0, W), R.y[:, i], method="DOP853", rtol=1e-10, atol=ATOL,
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
                 f"attitude snapshots on the way down\n({solar_label(solar)}, {R.material} fins)", fontsize=12)
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
                 "Final descent below 150 km:\nnose attitude vs flight path",
                 "Final descent: speed and angle of attack")
    footer(fig, dart, f"Released at {rel_alt:g} km, {aoa0:+g}° from the wind, rotating with the local vertical.")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = Path(args.out) / "overview.png"
    fig.savefig(out, dpi=130); print(f"saved {out}")
    print(f"crossover ~{k['crossover']:.0f} km; impact {full['speed'][-1]:.0f} m/s, "
          f"final nose pitch {full['nose_pitch'][-1]:.1f} deg")

    # ---- heating ----
    H = trajectory_heating(R, 200.0)
    heating_figure(R, H, Path(args.out) / "heating.png")


# =============================================================================
# Subcommand: video
# =============================================================================
def attitude_period(R, y):
    rho, n, T, v_b = local_state(R.dart, R.atm, y)
    d = np.radians(1.0)
    f = float(np.clip(y[7], 0, 1))
    tp = aero(R.dart, rot(-d) @ v_b, 0.0, rho, n, T, 1 - f)[1]
    tm = aero(R.dart, rot(d) @ v_b, 0.0, rho, n, T, 1 - f)[1]
    k = -(tp - tm) / (2 * d)
    return 2 * np.pi * np.sqrt(R.dart.I_t / k) if k > 0 else R.T_rel / 4


def video_timeline(R, fps, dwell_s, ff_s):
    """Frame times: dwell windows (real dynamics, slowed so ~3 swings show)
    joined by fast-forwards, then continuous playback through reentry."""
    frames = []

    def seg(t0, t1, seconds, label):
        n = max(2, int(round(seconds * fps)))
        rate = (t1 - t0) / seconds
        frames.extend((tt, label, rate) for tt in np.linspace(t0, t1, n, endpoint=False))

    t_imp = float(R.t[-1])
    dwell = [R.rel_alt] + [a for a in (400, 300, 250, 200, 150, 120) if a < R.rel_alt - 10]
    prev_end = None
    for a in dwell:
        ta = 0.0 if a == R.rel_alt else R.t_at_alt(a)
        if ta is None or (prev_end is not None and ta < prev_end):
            continue
        if prev_end is not None and ta > prev_end:
            seg(prev_end, ta, ff_s, "fast-forward")
        P = attitude_period(R, R.states_at([ta])[:, 0])
        dur = min(3 * P, 2 * R.T_rel, t_imp - ta)
        seg(ta, ta + dur, dwell_s, f"swinging at {a:g} km")
        prev_end = ta + dur
    t110, t20 = R.t_at_alt(110), R.t_at_alt(20)
    start = prev_end or 0.0
    if t110 is not None and t110 > start:
        seg(start, t110, 1.0, "fast-forward"); start = t110
    if t20 is not None and t20 > start:
        seg(start, t20, 25.0, "reentry"); start = t20
    seg(start, t_imp, 6.0, "terminal descent")
    frames.extend([(t_imp, "impact", 0.0)] * int(2.5 * fps))
    return frames


def fmt_met(t):
    d, rem = divmod(t, 86400)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    return f"T+ {int(d):2d} d {int(hh):02d}:{int(mm):02d}:{ss:04.1f}"


def fmt_rate(r):
    if r <= 0:
        return ""
    if r >= 1e4:
        return f"×{r:,.0f}".replace(",", " ")
    return f"×{r:.0f}" if r >= 10 else f"×{r:.1f}"


def cmd_video(args):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    R = Run(Path(args.out) / args.file)
    dart, mat = R.dart, MATERIALS[R.material]
    fps = args.fps
    frames = video_timeline(R, fps, args.dwell_seconds, args.ff_seconds)
    times = np.array([f[0] for f in frames])
    print(f"{len(frames)} frames ({len(frames)/fps:.0f} s); integrating states…")
    order = np.argsort(times, kind="stable")
    Ys = np.zeros((R.y.shape[0], len(times)))
    Ys[:, order] = R.states_at(times[order])
    diag = diagnostics(times, Ys)
    props = [state_props(dart, R.atm, Ys[:, k], R.material) for k in range(len(times))]

    # background curves (below 160 km) and cumulative nose heat load
    H = trajectory_heating(R, 200.0, max_points=6000)
    Qn = cumtrapz(H["q_nose"], H["t"])
    t_imp = float(R.t[-1])

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 11})
    W, Hpx, dpi = 1920, 1080, 100
    fig = plt.figure(figsize=(W / dpi, Hpx / dpi), dpi=dpi, facecolor="#0b0f1a")
    txtc = "#e8ecf4"

    def style(ax):
        ax.set_facecolor("#111827")
        for sp in ax.spines.values():
            sp.set_color("#55607a")
        ax.tick_params(colors=txtc, labelsize=9)
        ax.xaxis.label.set_color(txtc); ax.yaxis.label.set_color(txtc); ax.title.set_color(txtc)
        ax.grid(alpha=0.15)

    title = fig.text(0.02, 0.955, f"VLEO dart — {dart.m*1e3:.0f} g, {dart.fin_t*1e3:.0f} mm {R.material} fins, "
                     f"released {R.rel_alt:g} km {'nadir' if abs(R.aoa0-90)<1 else f'{R.aoa0:+g}°'}-pointing",
                     color=txtc, fontsize=16, weight="bold")
    met = fig.text(0.50, 0.955, "", color=txtc, fontsize=16, family="monospace", ha="center")
    phase = fig.text(0.98, 0.955, "", color="#ffd166", fontsize=16, ha="right", weight="bold")

    # dart view
    axd = fig.add_axes([0.02, 0.30, 0.46, 0.62])
    axd.set_xlim(-0.42, 0.42); axd.set_ylim(-0.32, 0.32); axd.set_aspect("equal")
    axd.set_xticks([]); axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_color("#55607a")
    axd.axhline(0, c="#8899bb", lw=0.8, ls="--", alpha=0.6)
    axd.text(-0.41, 0.006, "local horizon (prograde →)", color="#8899bb", fontsize=9, ha="left", va="bottom")
    axd.annotate("Earth (nadir)", (-0.37, -0.31), (-0.37, -0.20), color="#8899bb", fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#8899bb"))
    axd.plot([0.25, 0.35], [-0.29, -0.29], c=txtc, lw=2); axd.text(0.30, -0.282, "10 cm", color=txtc, ha="center", fontsize=9)
    glow = Circle((0, 0), 0.05, fc="#ff7b00", ec="none", alpha=0.0, zorder=1)
    glow2 = Circle((0, 0), 0.02, fc="#ffe7a0", ec="none", alpha=0.0, zorder=1)
    axd.add_patch(glow); axd.add_patch(glow2)
    x_tip, x_tail, r, Lc = dart.x_tip, dart.x_tail, dart.r, dart.Lc
    x_sh = x_tip - Lc
    body_xy = np.array([[x_tail, r], [x_sh, r], [x_sh, -r], [x_tail, -r]])
    cone_xy = np.array([[x_sh, r], [x_tip, 0], [x_sh, -r]])
    body = Polygon(body_xy, fc="#c9ced8", ec="k", zorder=3); axd.add_patch(body)
    cone = Polygon(cone_xy, fc="#c9ced8", ec="k", zorder=3); axd.add_patch(cone)
    fins = [Polygon(np.zeros((4, 2)), ec="k", zorder=2) for _ in range(2)]
    for p in fins:
        axd.add_patch(p)
    tip = Circle((x_tip, 0), 0.008, ec="none", zorder=4); axd.add_patch(tip)
    cg = axd.plot([0], [0], marker="o", ms=7, mfc="w", mec="k", zorder=5)[0]
    wind = axd.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#4cc9f0", lw=2.5), zorder=6)
    wind_lbl = axd.text(0, 0, "relative wind", color="#4cc9f0", fontsize=10, ha="center")
    regime_lbl = axd.text(-0.41, 0.30, "", color=txtc, fontsize=11, va="top")
    tnorm = plt.Normalize(200, max(1600, mat["T_fail"] + 300))
    cmap = plt.get_cmap("inferno")
    import matplotlib.colors as mcolors

    def hot(T, base):
        """Material colour when cold, fading into the inferno heat colour as it warms."""
        w = float(np.clip((T - 330) / 350, 0, 1))
        return tuple(np.array(mcolors.to_rgb(base)) * (1 - w) + np.array(cmap(tnorm(T))[:3]) * w)
    fin_base = {"balsa": "#d9b77e", "g10": "#b7c46a", "aluminium": "#b8bec8",
                "titanium": "#9c95b0", "steel": "#8c9198"}[R.material]

    # indicator text + fin bar
    ind = fig.text(0.51, 0.90, "", color=txtc, fontsize=13.5, family="monospace", va="top", linespacing=1.45)
    axb = fig.add_axes([0.70, 0.335, 0.27, 0.025])
    axb.set_xlim(0, tnorm.vmax - 273.15); axb.set_yticks([]); style(axb)
    axb.axvline(mat["T_fail"] - 273.15, c="#ef476f", lw=2)
    axb.text(mat["T_fail"] - 273.15, 1.3, f" {MATERIALS[R.material]['label']}", color="#ef476f", fontsize=9,
             transform=axb.get_xaxis_transform())
    finbar = axb.barh([0], [0], height=1.0, color="#ffd166")[0]
    axb.set_xlabel("bulk fin temperature (°C)", fontsize=9)
    axl = fig.add_axes([0.51, 0.335, 0.15, 0.025]); axl.set_xlim(0, 100); axl.set_yticks([]); style(axl)
    lifebar = axl.barh([0], [100], height=1.0, color="#06d6a0")[0]
    axl.set_xlabel("fin material left (%)", fontsize=9)

    # mini plots
    m1 = fig.add_axes([0.05, 0.065, 0.26, 0.17]); style(m1)
    tti = np.maximum(t_imp - R.t, 0.1)
    stp = max(1, len(R.t) // 20000)
    m1.semilogx(tti[::stp], R.h[::stp], c="#4cc9f0", lw=0.8)
    m1.invert_xaxis(); m1.set_xlabel("time to impact (s, log)"); m1.set_ylabel("altitude (km)")
    m1.axvline(86400, c="#55607a", lw=0.6); m1.text(86400, R.rel_alt * 0.95, " 1 day", color="#8899bb", fontsize=8)
    m1.axvline(3600, c="#55607a", lw=0.6); m1.text(3600, R.rel_alt * 0.95, " 1 h", color="#8899bb", fontsize=8)
    mk1 = m1.plot([], [], "o", c="#ffd166", ms=7)[0]
    m2 = fig.add_axes([0.38, 0.065, 0.26, 0.17]); style(m2)
    m2.semilogx(H["q_nose"], H["h"], c="#ef476f", lw=0.8, label="nose")
    m2.semilogx(H["q_le"], H["h"], c="#ffd166", lw=0.8, label="fin LE")
    m2.set_ylim(0, 160); m2.set_xlim(1, H["q_le"].max() * 3)
    m2.set_xlabel("heat flux (W/m², log)"); m2.set_ylabel("altitude (km)")
    m2.legend(fontsize=8, facecolor="#111827", labelcolor=txtc, loc="upper right")
    mk2 = m2.plot([], [], "o", c="#ef476f", ms=7)[0]; mk2b = m2.plot([], [], "o", c="#ffd166", ms=7)[0]
    m3 = fig.add_axes([0.71, 0.065, 0.26, 0.17]); style(m3)
    m3.plot(H["T_fin"] - 273.15, H["h"], c="#ffd166", lw=0.9, label="bulk fin")
    m3.plot(H["T_nose_eq"] - 273.15, H["h"], c="#ef476f", lw=0.8, alpha=0.7, label="nose (rad. eq.)")
    m3.axvline(mat["T_fail"] - 273.15, c="#ef476f", lw=1, ls="--")
    m3.set_ylim(0, 160); m3.set_xlabel("temperature (°C)"); m3.set_ylabel("altitude (km)")
    m3.legend(fontsize=8, facecolor="#111827", labelcolor=txtc, loc="upper right")
    mk3 = m3.plot([], [], "o", c="#ffd166", ms=7)[0]; mk3b = m3.plot([], [], "o", c="#ef476f", ms=7)[0]
    fig.text(0.5, 0.008, "Rough model: planar equatorial orbit, pitch only; NRLMSIS 2.0; Sentman free-molecular + "
             "Barrowman-style continuum aero; Sutton-Graves heating; lumped fins. Dart drawn to scale.",
             color="#8899bb", fontsize=9, ha="center")

    def regime(Kn):
        return "free-molecular" if Kn > 10 else ("transitional" if Kn > 0.01 else "continuum")

    out = Path(args.out) / args.video_file
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{Hpx}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    wall = time.time()
    for k, (tt, label, rate) in enumerate(frames):
        y, p = Ys[:, k], props[k]
        hk, Tf, left = diag["h"][k], y[6], 1 - float(np.clip(y[7], 0, 1))
        pitch = np.radians(diag["nose_pitch"][k])
        Rm = rot(pitch)[:2, :2]
        tr = lambda xy: xy @ Rm.T
        body.set_xy(tr(body_xy)); cone.set_xy(tr(cone_xy))
        sp = dart.fin_span * left
        fc = hot(Tf, fin_base)
        for p_, s_ in zip(fins, (1, -1)):
            p_.set_xy(tr(np.array([[x_tail, s_*r], [x_tail + dart.fin_chord, s_*r],
                                   [x_tail + dart.fin_chord, s_*(r + sp)], [x_tail, s_*(r + sp)]])))
            p_.set_facecolor(fc); p_.set_visible(left > 0.005)
        tip_xy = tr(np.array([[x_tip, 0.0]]))[0]
        tip.center = tip_xy; tip.set_facecolor(hot(p["T_nose_eq"], "#c9ced8"))
        lq = np.log10(max(p["q_nose"], 1.0))
        a = float(np.clip((lq - 4.0) / 3.5, 0, 1))
        glow.center = glow2.center = tip_xy
        glow.set_radius(0.02 + 0.10 * a); glow.set_alpha(0.35 * a)
        glow2.set_radius(0.01 + 0.035 * a); glow2.set_alpha(0.8 * a)
        fpa = np.radians(diag["fpa"][k])
        u = np.array([np.cos(fpa), np.sin(fpa)])
        wind.xy = tuple(0.27 * u); wind.set_position(tuple(0.40 * u))
        wind_lbl.set_position(tuple(0.335 * u + [0, -0.035])); wind_lbl.set_visible(label != "impact")
        wind.set_visible(label != "impact")
        # atmosphere tint
        tint = float(np.clip((np.log10(p["rho"]) + 12) / 12, 0, 1))
        axd.set_facecolor(tuple(np.array([0.03, 0.04, 0.08]) * (1 - tint) + np.array([0.35, 0.55, 0.85]) * tint))
        regime_lbl.set_text(f"flow regime: {regime(p['Kn'])}")
        met.set_text(fmt_met(tt))
        if label == "impact":
            phase.set_text("IMPACT"); phase.set_color("#ef476f")
        elif label == "fast-forward":
            phase.set_text(f">> fast-forward {fmt_rate(rate)}"); phase.set_color("#8899bb")
        else:
            phase.set_text(f"▶ {label}  {fmt_rate(rate)}"); phase.set_color("#ffd166")
        Qk = np.interp(tt, H["t"], Qn) if tt >= H["t"][0] else 0.0
        ind.set_text(
            f"Altitude            {hk:9.2f} km\n"
            f"Speed (rel. air)    {diag['speed'][k]:9.0f} m/s    Mach {p['mach']:5.1f}\n"
            f"Flight path angle   {diag['fpa'][k]:9.1f}°\n"
            f"Nose vs wind (AoA)  {diag['aoa'][k]:9.1f}°\n"
            f"Nose vs horizon     {diag['nose_pitch'][k]:9.1f}°\n"
            f"Air density         {p['rho']:9.2e} kg/m³\n"
            f"Dynamic pressure    {p['qdyn']:9.3g} Pa\n"
            f"Knudsen number      {p['Kn']:9.3g}   ({regime(p['Kn'])})\n"
            f"Nose heat flux      {p['q_nose']:9.3g} W/m²   → {p['T_nose_eq']-273.15:6.0f} °C rad.-eq.\n"
            f"Fin LE heat flux    {p['q_le']:9.3g} W/m²   → {p['T_le_eq']-273.15:6.0f} °C rad.-eq.\n"
            f"Heat into vehicle   {p['P_total']:9.3g} W   (fins {p['P_fins']:.3g} W)\n"
            f"Nose heat load      {Qk/1e6:9.2f} MJ/m²\n"
            f"Bulk fin temp       {Tf-273.15:9.0f} °C\n"
            f"Fin material left   {100*left:9.1f} %")
        finbar.set_width(max(Tf - 273.15, 0)); finbar.set_color(cmap(tnorm(Tf)))
        lifebar.set_width(100 * left)
        mk1.set_data([max(t_imp - tt, 0.1)], [hk])
        vis = hk < 160
        mk2.set_data([p["q_nose"]], [hk]); mk2b.set_data([p["q_le"]], [hk])
        mk3.set_data([Tf - 273.15], [hk]); mk3b.set_data([p["T_nose_eq"] - 273.15], [hk])
        for mk in (mk2, mk2b, mk3, mk3b):
            mk.set_visible(vis)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
        if k % 300 == 0:
            print(f"  frame {k}/{len(frames)}  {time.time()-wall:.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    plt.close(fig)
    print(f"saved {out}")


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
        sp.add_argument("--fin-material", choices=list(MATERIALS), default="balsa",
                        help="fin material (default balsa; the only one light enough for a 100 g dart)")

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
    sp.add_argument("--fin-material", choices=list(MATERIALS), default="fr4")
    sp.set_defaults(func=cmd_survey)

    sp = sub.add_parser("run", help="continuous run from release to the ground")
    common(sp)
    sp.add_argument("--max-days", type=float, default=400, help="stop after this many days")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("plot", help="figures from a run file")
    sp.add_argument("--file", default="continuous_run.npz")
    sp.set_defaults(func=cmd_plot)

    def video_opts(sp):
        sp.add_argument("--fps", type=int, default=30)
        sp.add_argument("--dwell-seconds", type=float, default=4.0, help="video seconds per altitude snapshot")
        sp.add_argument("--ff-seconds", type=float, default=1.5, help="video seconds per fast-forward")
        sp.add_argument("--video-file", default="dart_descent.mp4")

    sp = sub.add_parser("video", help="MP4 of the descent with live indicators (needs ffmpeg)")
    sp.add_argument("--file", default="continuous_run.npz")
    video_opts(sp)
    sp.set_defaults(func=cmd_video)

    sp = sub.add_parser("all", help="render + run + plot + video")
    common(sp)
    sp.add_argument("--max-days", type=float, default=400)
    video_opts(sp)
    sp.set_defaults(func=lambda a: (cmd_render(a), cmd_run(a), cmd_plot(a), cmd_video(a)))

    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
