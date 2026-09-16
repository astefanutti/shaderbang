# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Per-frame parameters shared by the plasma globe simulation modules.

Everything that changes from frame to frame (camera, fingers, drive voltage/frequency, growth
knobs, gravity sign, toggles, dt, seed) travels in ONE device array of ``PlasmaParams`` that the
application fills from a pinned host mirror and copies as the first node of the frame graph.
Kernels read ``params[0]`` -- nothing time-varying is ever baked into a captured graph as a
by-value kernel argument (the failure mode ``examples/cloth.py`` works around by recapturing its
graph every frame).

Provenance of the default values: see the module docstring of ``examples/plasma_globe.py`` and
``docs/plasma_globe.md`` (MEASURED = PPPL-4485, Campanell et al. 2010; DERIVED = neon transport
properties at 740 Torr / 300 K; CHOSEN = design constants to be calibrated in the headless lab).
"""

import ctypes

import numpy as np
import warp as wp


MAX_FINGERS = 10          # matches the shadertoy ``iTouch`` vec4[10] convention

# ---- geometry (m) ---------------------------------------------------------------------------
R1 = 0.011                # electrode bulb radius: the reference footage shows the bulb at ~1/7 of the
                          # globe diameter (PPPL's lab globe had 1.5-2 cm at R1/R2 = 0.2)   CHOSEN
R2I = 0.075               # inner glass radius                         MEASURED (PPPL Table 1)
R2O = 0.0775              # outer glass radius (2.5 mm soda-lime)      CHOSEN
NODE_SPACING = 1.5e-3     # h: filament node spacing                   CHOSEN
CHARGE_RADIUS = NODE_SPACING / 4.0   # a: conductor radius of a node   DERIVED (a/h sweep in M2)

# ---- gas (neon + 2 % xenon, 740 Torr, 300 K) --------------------------------------------------
T0 = 300.0                # ambient temperature (K)
RHO0 = 0.798              # density (kg/m^3)                           DERIVED
CP = 1030.0               # heat capacity (J/kg/K)                     DERIVED
K_TH = 0.0491             # thermal conductivity (W/m/K)               DERIVED
NU = 3.97e-5              # kinematic viscosity (m^2/s)                DERIVED
ALPHA_TH = 5.97e-5        # thermal diffusivity (m^2/s)                DERIVED

# ---- drive / circuit ------------------------------------------------------------------------
FREQ0 = 26.0e3            # drive frequency (Hz)                       MEASURED
VOLT0 = 5.0e3             # drive peak voltage (V)                     MEASURED
I_SUPPLY = 1.5e-3         # driver current limit (A)                   CHOSEN (headroom for a touched filament)
V_CH = 2.0e3              # channel sustaining voltage (V)             CHOSEN (~330 V/cm x 6 cm)
I_STRIKE = 2.0e-5         # current needed to admit a new tree (A)     CHOSEN (-> N_max ~ 60; the footage
                          # shows 30+ thin filaments with a wide brightness spread)
I_SUSTAIN_RATIO = 0.2     # sustain / strike hysteresis                 CHOSEN (an established hot channel
                          # survives at a fraction of its strike current: under a finger the other
                          # filaments dim to ~10 % instead of vanishing; PPPL Sec. 5 gives the trend)
EPS_R_GLASS = 6.0         # soda-lime relative permittivity
GLASS_T = R2O - R2I       # glass thickness (m)
C_D = 8.854e-12 * EPS_R_GLASS / GLASS_T           # dielectric capacitance per area (F/m^2) ~ 21 nF/m^2
FOOT_RADIUS = 2.0e-3      # effective footprint radius on the glass    CHOSEN
A_FOOT = np.pi * FOOT_RADIUS ** 2
R_CH = 25.0e6             # differential channel resistance per unit length (ohm/m) CHOSEN: with the 215x touched
                          # termination it caps a touched filament at ~15x an ordinary one (footage: the touched
                          # channel is white but 15-20 thin filaments stay lit); 3.3e6 gave 60-80x and starved them
TOUCH_GAIN = 214.0        # finger termination admittance gain          DERIVED (215x per unit area)
E_BD0 = 190.0e3           # root breakdown field at T0 (V/m)           CHOSEN (strike ~ 2.5 kV)
E_PROP_RATIO = 0.25       # propagation threshold / root threshold     CHOSEN

# ---- surface charge records (units of sigma_sat) ----------------------------------------------
TAU_ENV = 0.090           # envelope decay (s)                         MEASURED (Burin 2015)
D_SIGMA = 2.5e-4          # lateral spreading diffusivity (m^2/s)      MEASURED (Burin 2015)
TAU_FINGER = 0.005        # finger drain time constant (s)             CHOSEN
SIGMA_A = 0.0             # activator gain at the footprint            CHOSEN: off (the persistent channel
                          # is the foot memory; with the sharp selection (eta ~100) any activator
                          # would glue every new foot onto an existing one)
SIGMA_B = 0.5             # inhibitor gain over the annulus            CHOSEN (flagged)
R_INH = 8.0e-3            # inhibition annulus radius (m)              CHOSEN (swept in M4)

# ---- touch ------------------------------------------------------------------------------------
FINGER_STANDOFF = 0.045 * R2I          # h_f above the inner glass     MEASURED (follow-up 8)
FINGER_HWHM = 2.40 * FINGER_STANDOFF   # contact half width at half maximum: ~8 mm, a fingertip pad
                                       # (~15 mm across) coupling through 2.5 mm of glass   CHOSEN
Q_FINGER0 = 0.05                       # finger charge (phi * m)        CHOSEN (calibrated in M2)


@wp.struct
class PlasmaParams:
    # camera basis (for picking on the device if ever needed) and frame bookkeeping
    frame: wp.int32
    seed: wp.int32
    dt: wp.float32
    time: wp.float32
    # run / step / reset flags
    running: wp.int32
    reset: wp.int32
    # drive
    voltage: wp.float32
    frequency: wp.float32
    # growth knobs
    eta: wp.float32
    gamma: wp.float32
    global_norm: wp.int32
    q_finger: wp.float32
    # gas toggles
    g_sign: wp.float32
    ice_cap: wp.int32
    # lifecycle toggles
    hybrid: wp.int32          # 1 = re-strike / re-route enabled, 0 = persistent only
    quincunx: wp.int32        # 1 = footprint self-inhibition (rare regime), 0 = activator footprint
    # the electrode as a heated sphere (host-side balance of its sheath power, globe.fill_params)
    t_ball: wp.float32        # K above ambient at the envelope's surface
    u_bl: wp.float32          # m/s, velocity scale of its natural-convection boundary layer
    delta_bl: wp.float32      # m, thickness of that layer
    # fingers on the glass (unit directions + contact weights); count <= MAX_FINGERS
    num_fingers: wp.int32
    finger_dir0: wp.vec3
    finger_dir1: wp.vec3
    finger_dir2: wp.vec3
    finger_dir3: wp.vec3
    finger_dir4: wp.vec3
    finger_dir5: wp.vec3
    finger_dir6: wp.vec3
    finger_dir7: wp.vec3
    finger_dir8: wp.vec3
    finger_dir9: wp.vec3


@wp.func
def finger_dir(p: PlasmaParams, i: int) -> wp.vec3:
    """Return the i-th finger direction (kernels cannot index struct fields dynamically)."""
    if i == 0:
        return p.finger_dir0
    if i == 1:
        return p.finger_dir1
    if i == 2:
        return p.finger_dir2
    if i == 3:
        return p.finger_dir3
    if i == 4:
        return p.finger_dir4
    if i == 5:
        return p.finger_dir5
    if i == 6:
        return p.finger_dir6
    if i == 7:
        return p.finger_dir7
    if i == 8:
        return p.finger_dir8
    return p.finger_dir9


class ParamsBuffer:
    """Pinned host mirror + device array of one ``PlasmaParams``.

    ``host`` is a numpy structured view the application mutates each frame (fields have the
    same names as the struct); ``upload()`` issues the single H2D copy that the frame graph
    records as its first node (``wp.copy`` between two fixed arrays is capturable).
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        self.device_array = wp.zeros(1, dtype=PlasmaParams, device=device)
        self.host_array = wp.zeros(1, dtype=PlasmaParams, device="cpu", pinned=True)
        self.host = self.host_array.numpy()
        self.reset_defaults()

    def reset_defaults(self):
        h = self.host[0]
        h["dt"] = 1.0 / 60.0
        h["running"] = 1
        h["voltage"] = VOLT0
        h["frequency"] = FREQ0
        h["eta"] = 1.5
        h["gamma"] = 2.0
        h["q_finger"] = Q_FINGER0
        h["g_sign"] = 1.0
        h["hybrid"] = 1

    def set_fingers(self, dirs):
        """dirs: iterable of unit vectors (at most MAX_FINGERS)."""
        dirs = list(dirs)[:MAX_FINGERS]
        self.host[0]["num_fingers"] = len(dirs)
        for i in range(MAX_FINGERS):
            d = dirs[i] if i < len(dirs) else (0.0, 1.0, 0.0)
            self.host[0][f"finger_dir{i}"] = (float(d[0]), float(d[1]), float(d[2]))

    def upload(self):
        wp.copy(self.device_array, self.host_array)


if __name__ == "__main__":
    wp.init()
    pb = ParamsBuffer()
    pb.host[0]["frame"] = 7
    pb.set_fingers([(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
    pb.upload()
    wp.synchronize()
    back = pb.device_array.numpy()[0]
    assert back["frame"] == 7 and back["num_fingers"] == 2
    assert abs(back["finger_dir1"][0] - 1.0) < 1e-6
    print("PlasmaParams itemsize", pb.host.dtype.itemsize, "bytes; upload round trip OK")


# ---- node / tree enumerations shared by dbm.py and circuit.py ----------------------------------
# node flag bits
NODE_ALIVE = 1
NODE_CHARGED = 2
NODE_DECAYING = 4
NODE_FOOT = 8
NODE_ROOT = 16
NODE_MAINCUT = 32        # transient: main-channel node beyond a re-route fork
NODE_FREE_PENDING = 64   # decayed; the growth engine reclaims it into its free list

# tree states
TREE_FREE = 0
TREE_SEED = 1
TREE_GROW = 2
TREE_ATTACHED = 3
TREE_RETRACT = 4
TREE_REROUTE = 5   # legacy (the engine re-routes in place: ATTACHED with a leader, see dbm.Dbm)

# per-frame tree commands issued by the circuit / lifecycle kernels
CMD_NONE = 0
CMD_REROUTE = 1
CMD_RETRACT = 2
