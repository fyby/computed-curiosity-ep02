"""Physical constants and fixed parameters (sources: FACTS.md section 1).

Units everywhere: au, yr, solar masses, G = 4 pi^2 exactly.

Note: every REBOUND simulation sets ``sim.G = 4 pi^2`` directly. REBOUND's
``sim.units = ("yr", "AU", "Msun")`` would instead use the Gaussian constant
(G = 39.4769, 3.8e-5 smaller), which made the giant's period differ from the
spec's Kepler value by 2e-5 and broke validation V1. With G = 4 pi^2 the "yr"
is the period of a 1-au orbit around 1 solar mass.
"""
import math
import os

G = 4.0 * math.pi ** 2
M_STAR = 1.0

MU_JUPITER = 1.0 / 1047.348644          # Sun / Jupiter-system mass ratio (JPL DE440 GMs)
A_G = 5.2038                            # au, NASA Jupiter fact sheet (778.479e6 km)
E_G = 0.0487                            # NASA Jupiter fact sheet

AU_KM = 149_597_870.7
R_STAR = 695_700.0 / AU_KM              # au (NASA Sun fact sheet)
R_GIANT = 69_911.0 / AU_KM              # au (NASA Jupiter fact sheet, volumetric mean)


def period(m_giant=MU_JUPITER, a=A_G):
    """Kepler period of the giant (yr) around a 1 Msun star."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G * (M_STAR + m_giant)))


def hill(m_giant=MU_JUPITER, a=A_G):
    """Hill radius a (m/3M)^(1/3), the same for every giant eccentricity."""
    return a * (m_giant / (3.0 * M_STAR)) ** (1.0 / 3.0)


P_G = period()                          # 11.865 yr
R_HILL = hill()                         # 0.3553 au

# Disc of test planets (E1)
A_LO, A_HI = 0.30 * A_G, 2.00 * A_G     # 1.561 .. 10.408 au
E0_MAX = 0.3
INC_MAX_DEG = 5.0

# Outcome boundaries
R_IN = 0.2 * A_G                        # 1.041 au -> INWARD
R_OUT = 10.0 * A_G                      # 52.04 au -> OUTWARD

# Outcome classes
SURVIVED, ENCOUNTER, INWARD, OUTWARD = 0, 1, 2, 3
CLASS_NAMES = {SURVIVED: "SURVIVED", ENCOUNTER: "ENCOUNTER", INWARD: "INWARD", OUTWARD: "OUTWARD"}

# Step-size zones by starting semimajor axis: (name, a_lo, a_hi, steps per giant lap).
# Every e0 <= 0.3 orbit gets >= 20 steps per pericentre time q^1.5, the giant >= 60 per lap.
ZONES = [("A", A_LO, 1.72, 240), ("B", 1.72, 2.25, 180), ("C", 2.25, 3.60, 120), ("D", 3.60, A_HI, 60)]
CHECKS_PER_LAP = 15                      # event test every P_G/15
SAMPLES_PER_LAP = 3                      # element sample every P_G/3
UNDER_RESOLVED_STEPS = 12                # flag if pericentre time < 12 steps

# Seeds
SEED_E1 = 2_000_001
SEED_E3 = 2_000_003
SEED_VALID = 2_100_000                   # + test number (V2 2_100_002, ...)
SEED_E5 = 2_000_005

# Nominal run lengths
T_MAIN = 50_000.0
T_FALLBACK = 40_000.0
FALLBACK_LIMIT_MIN = 85.0
T_DIAL = 10_000.0
T_FATE = 200_000.0

N_MAIN = 1_000_000
N_DIAL = 100_000

N_PROCESSES = int(os.environ.get("MILLION_PROCESSES", "10"))

# Smoke-test mode: MILLION_SMOKE=1 shrinks every experiment (pipeline test only,
# results are meaningless). Never set for a run of record.
SMOKE = os.environ.get("MILLION_SMOKE", "0") == "1"
if SMOKE:
    N_MAIN, N_DIAL = 20_000, 4_000
    T_MAIN, T_FALLBACK, T_DIAL, T_FATE = 2_000.0, 1_600.0, 600.0, 3_000.0


def zone_index(a):
    """Index into ZONES for starting semimajor axis a (scalar or array)."""
    import numpy as np
    edges = np.array([z[2] for z in ZONES[:-1]])
    return np.searchsorted(edges, a, side="right")
