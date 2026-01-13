#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import atexit
from math import pi
from typing import Any, Callable, Optional, cast

import numpy as np

from Karana.Core import allFinalized, discard
from Karana.Frame import FrameContainer
from Karana.Math import HomTran, IntegratorType, SpatialInertia
from Karana.Dynamics import (
    HingeType,
    Multibody,
    PhysicalBody,
    PhysicalHinge,
    PhysicalModalBody,
    PinSubhinge,
    StatePropagator,
    TimedEvent,
    SpSolverType,
)
from Karana.Models import PyKModelBase, SyncRealTime, UniformGravity, UpdateProxyScene
from Karana.Scene import BoxGeometry, Color, PhysicalMaterial, PhysicalMaterialInfo, ProxyScenePart

# -----------------------------
# Config
# -----------------------------
ROD_LEN: float = 1.0
N_MODES: int = 3

# Visual scaling for the force indicator marker (purely cosmetic)
FORCE_VIS_SCALE: float = 0.002  # meters per Newton
FORCE_VIS_MAX: float = 0.35     # clamp distance

# Phase D/E forcing parameters (tune these)
PHASE_D_FORCE_MAG: float = 10.0     # N
PHASE_D_SWEEP_HZ: float = 0.06      # cycles/sec (slower)
PHASE_E_FORCE_MAG: float = 10.0     # N
PHASE_E_CIRCLE_HZ: float = 0.08     # cycles/sec (slower)

# Hinge2 anti-collision limits (this is the big fix)
HINGE2_Q_MIN: float = -0.90
HINGE2_Q_MAX: float = +0.90

# Force component testing mode:
# "NONE" | "X_ONLY" | "Y_ONLY"
FORCE_COMPONENT_MODE: str = "NONE"

# --- VISUAL BENDING HACK (mode-shape scaling) ---
# Higher = more visible bend. Try 0.1 .. 1.0
VIS_BEND_SCALE: float = 0.35

# -----------------------------
# Globals for visuals/debug
# -----------------------------
bd2_pnode: Any = None
bd2_tip_onode: Any = None

force_marker: Optional[ProxyScenePart] = None
force_phase: str = "NONE"  # "D" | "E" | "NONE"

PHASE_F_FX_CONST: float = 15.0  # N
PHASE_G_FY_CONST: float = 15.0  # N

def _scalar(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


# -----------------------------
# Build multibody
# -----------------------------
def create_multibody(fc: FrameContainer) -> tuple[Multibody, PhysicalBody, PhysicalModalBody, PhysicalBody]:
    """
    mb: virtualRoot --PIN--> bd1 (rigid) --PIN--> bd2 (modal) --PIN--> tip (tiny rigid)
    """
    global bd2_pnode, bd2_tip_onode

    mb: Multibody = Multibody("rod_mb", fc)

    spI: SpatialInertia = SpatialInertia(2.0, np.zeros(3), np.diag([3.0, 2.0, 1.0]))

    # bd1 rigid link
    bd1: PhysicalBody = PhysicalBody("bd1", mb)
    bd1.setSpatialInertia(spI)
    bd1_hge: PhysicalHinge = PhysicalHinge(cast(PhysicalBody, mb.virtualRoot()), bd1, HingeType.PIN)
    bd1.setBodyToJointTransform(HomTran(np.array([0.0, 0.0, 0.0], dtype=float)))
    cast(PinSubhinge, bd1_hge.subhinge(0)).setUnitAxis(np.array([0.0, 0.0, 1.0], dtype=float))

    # bd2 modal link
    bd2: PhysicalModalBody = PhysicalModalBody.create("bd2", mb, N_MODES)
    bd2.setSpatialInertia(spI)
    bd2_hge: PhysicalHinge = PhysicalHinge(bd1, bd2, HingeType.PIN)
    bd2.setBodyToJointTransform(HomTran(np.array([0.0, 0.0, 0.0], dtype=float)))
    bd2_hge.onode().setBodyToNodeTransform(HomTran(np.array([+ROD_LEN, 0.0, 0.0], dtype=float)))
    cast(PinSubhinge, bd2_hge.subhinge(0)).setUnitAxis(np.array([0.0, 0.0, 1.0], dtype=float))

    # Softer + lower damping makes the "wiggle" easier to see
    bd2.setStiffnessVector(np.array([2.0, 10.0, 25.0], dtype=float))
    bd2.setDampingVector(np.array([0.02, 0.02, 0.02], dtype=float))

    # tip rigid body
    tip: PhysicalBody = PhysicalBody("tip", mb)
    tipI: SpatialInertia = SpatialInertia(0.001, np.zeros(3), np.diag([1e-6, 1e-6, 1e-6]))
    tip.setSpatialInertia(tipI)

    tip_hge: PhysicalHinge = PhysicalHinge(bd2, tip, HingeType.PIN)
    tip.setBodyToJointTransform(HomTran(np.array([0.0, 0.0, 0.0], dtype=float)))
    tip_hge.onode().setBodyToNodeTransform(HomTran(np.array([ROD_LEN, 0.0, 0.0], dtype=float)))
    cast(PinSubhinge, tip_hge.subhinge(0)).setUnitAxis(np.array([0.0, 0.0, 1.0], dtype=float))

    mb.ensureCurrent()

    # ---- finalize deformation providers (avoid allFinalized() assert bombs) ----
    # This is a VISUAL mode-shape hack: we map mode0 to a noticeable +Y translation at nodes.
    bd2_pnode = bd2.parentHinge().pnode()
    try:
        prov_p: Any = bd2_pnode.deformationProvider()
        if prov_p is not None:
            Phi_p: np.ndarray = np.zeros((6, N_MODES), dtype=float)

            # Mode 0: fake bend in +Y at mid-node
            Phi_p[1, 0] = 0.50 * VIS_BEND_SCALE   # translation Y
            Phi_p[5, 0] = 0.10 * VIS_BEND_SCALE   # small rotation (optional)

            prov_p.setNodalMatrix(Phi_p)
    except Exception:
        pass

    bd2_tip_onode = tip_hge.onode()
    try:
        prov_tip: Any = bd2_tip_onode.deformationProvider()
        if prov_tip is not None:
            Phi_tip: np.ndarray = np.zeros((6, N_MODES), dtype=float)

            # Mode 0: bigger bend in +Y at tip-node
            Phi_tip[1, 0] = 1.00 * VIS_BEND_SCALE  # translation Y
            Phi_tip[5, 0] = 0.25 * VIS_BEND_SCALE  # optional rotation

            prov_tip.setNodalMatrix(Phi_tip)
    except Exception:
        pass

    mb.resetData()

    if not allFinalized():
        print("[warning] allFinalized() == False after create_multibody(). Continuing anyway.")

    return mb, bd1, bd2, tip


# -----------------------------
# Models
# -----------------------------
freeze_hinges_enabled: bool = False


class FreezeHingesModel(PyKModelBase):
    def __init__(self, name: str, sp: Any, pin1: PinSubhinge, pin2: PinSubhinge) -> None:
        super().__init__(name, sp)
        self._pin1: PinSubhinge = pin1
        self._pin2: PinSubhinge = pin2

    def preDeriv(self, t: Any, x: Any) -> None:
        global freeze_hinges_enabled
        if not freeze_hinges_enabled:
            return
        self._pin1.setQ(0.0)
        self._pin1.setU(0.0)
        self._pin1.setT(0.0)
        self._pin2.setQ(0.0)
        self._pin2.setU(0.0)
        self._pin2.setT(0.0)


class HingeSoftLimits(PyKModelBase):
    """
    Stronger soft-stops + tighter hinge2 bounds to prevent rods "ghost colliding".
    """
    def __init__(self, name: str, sp: Any, pin1: PinSubhinge, pin2: PinSubhinge) -> None:
        super().__init__(name, sp)
        self._p1: PinSubhinge = pin1
        self._p2: PinSubhinge = pin2
        self.enabled: bool = True

        # hinge 1 bounds
        self._q1_min: float = -1.2
        self._q1_max: float = +1.2

        # hinge 2 bounds
        self._q2_min: float = HINGE2_Q_MIN
        self._q2_max: float = HINGE2_Q_MAX

        # stop region size
        self._eps: float = 0.04

        # strong stop
        self._k_stop: float = 3000.0
        self._d_stop: float = 250.0
        self._tau_max: float = 800.0

    def _limit_torque(self, q: float, u: float, qmin: float, qmax: float) -> float:
        tau: float = 0.0
        eps: float = self._eps

        if q < (qmin + eps) and u < 0.0:
            err: float = (qmin - q)
            tau = +self._k_stop * err - self._d_stop * u
        elif q > (qmax - eps) and u > 0.0:
            err2: float = (q - qmax)
            tau = -self._k_stop * err2 - self._d_stop * u

        return float(np.clip(tau, -self._tau_max, +self._tau_max))

    def preDeriv(self, t: Any, x: Any) -> None:
        if not self.enabled:
            return

        q1: float = _scalar(self._p1.getQ())
        u1: float = _scalar(self._p1.getU())
        tau1: float = self._limit_torque(q1, u1, self._q1_min, self._q1_max)

        q2: float = _scalar(self._p2.getQ())
        u2: float = _scalar(self._p2.getU())
        tau2: float = self._limit_torque(q2, u2, self._q2_min, self._q2_max)

        self._p1.setT(tau1)
        self._p2.setT(tau2)


class HardClampHinge(PyKModelBase):
    """
    Last-line-of-defense: clamp q/u so hinge2 cannot cross [q_min, q_max].
    This is not "physics", it's "nope".
    """
    def __init__(self, name: str, sp: Any, pin: PinSubhinge, q_min: float, q_max: float) -> None:
        super().__init__(name, sp)
        self._pin: PinSubhinge = pin
        self._q_min: float = float(q_min)
        self._q_max: float = float(q_max)
        self.enabled: bool = True

    def preDeriv(self, t: Any, x: Any) -> None:
        if not self.enabled:
            return

        q: float = _scalar(self._pin.getQ())
        u: float = _scalar(self._pin.getU())

        if q < self._q_min:
            self._pin.setQ(self._q_min)
            if u < 0.0:
                self._pin.setU(0.0)

        if q > self._q_max:
            self._pin.setQ(self._q_max)
            if u > 0.0:
                self._pin.setU(0.0)


class TipForceXYToHingeTorque(PyKModelBase):
    """
    Build limitation workaround:
    - Instead of applying Fx/Fy at the tip (in bd2 frame)
    - we convert to pin torque about +Z

    For r = [L, 0, 0] and F = [Fx, Fy, 0]:
      tau_z = (r x F)_z = L * Fy
    """
    def __init__(self, name: str, sp: Any, pin1: PinSubhinge, pin2: PinSubhinge, lever_len: float) -> None:
        super().__init__(name, sp)
        self._pin1: PinSubhinge = pin1
        self._pin2: PinSubhinge = pin2
        self._L: float = float(lever_len)
        self.enabled: bool = False
        self.force_xy: np.ndarray = np.zeros(2, dtype=float)  # [Fx, Fy]

    def preDeriv(self, t: Any, x: Any) -> None:
        if not self.enabled:
            return
        q1: float = _scalar(self._pin1.getQ())
        q2: float = _scalar(self._pin2.getQ())
        theta: float = q1 + q2

        fx: float = float(self.force_xy[0])
        fy: float = float(self.force_xy[1])

        rx: float = self._L * float(np.cos(theta))
        ry: float = self._L * float(np.sin(theta))

        tau_z: float = (rx * fy) - (ry * fx)
        self._pin2.setT(tau_z) 


class ForceDriver(PyKModelBase):
    """
    Drives tip_force_xy.force_xy during Phase D/E.
    """
    def __init__(self, name: str, sp: Any, force_model: TipForceXYToHingeTorque) -> None:
        super().__init__(name, sp)
        self._force_model: TipForceXYToHingeTorque = force_model
        self._t0_ns: Optional[int] = None

    def _set_marker(self, fx: float, fy: float) -> None:
        global force_marker
        if force_marker is None:
            return

        mag: float = float(np.hypot(fx, fy))
        if mag < 1e-12:
            force_marker.setTranslation([0.0, 0.0, 0.0])
            return

        ux_w: float = fx / mag
        uy_w: float = fy / mag
        q1: float = _scalar(bd1_pin.getQ())
        q2: float = _scalar(bd2_pin.getQ())
        theta: float = q1 + q2

        # Convert WORLD direction -> LOCAL tip-node direction by rotating -theta
        c: float = float(np.cos(-theta))
        s: float = float(np.sin(-theta))
        ux_l: float = c * ux_w - s * uy_w
        uy_l: float = s * ux_w + c * uy_w

        dist: float = min(FORCE_VIS_MAX, FORCE_VIS_SCALE * mag)
        force_marker.setTranslation([dist * ux_l, dist * uy_l, 0.0])
    
    def preDeriv(self, t: Any, x: Any) -> None:
        global force_phase
        if not self._force_model.enabled:
            return

        t_ns: int = int(np.asarray(t).reshape(-1)[0])
        if self._t0_ns is None:
            self._t0_ns = t_ns

        dt_s: float = (t_ns - self._t0_ns) / 1e9

        fx: float = float(self._force_model.force_xy[0])
        fy: float = float(self._force_model.force_xy[1])

        if force_phase == "D":
            theta: float = 2.0 * pi * PHASE_D_SWEEP_HZ * dt_s
            fx = PHASE_D_FORCE_MAG * float(np.cos(theta))
            fy = PHASE_D_FORCE_MAG * float(np.sin(theta))
        elif force_phase == "E":
            theta2: float = 2.0 * pi * PHASE_E_CIRCLE_HZ * dt_s
            fx = PHASE_E_FORCE_MAG * float(np.cos(theta2))
            fy = PHASE_E_FORCE_MAG * float(np.sin(theta2))

        # Optional filtering for component testing
        if FORCE_COMPONENT_MODE == "X_ONLY":
            fy = 0.0
        elif FORCE_COMPONENT_MODE == "Y_ONLY":
            fx = 0.0

        self._force_model.force_xy[0] = fx
        self._force_model.force_xy[1] = fy
        self._set_marker(fx, fy)


# -----------------------------
# Main
# -----------------------------
fc: FrameContainer = FrameContainer("root")
mb, bd1, bd2, tip = create_multibody(fc)

cleanup_graphics, web_scene = mb.setupGraphics(port=0, axes=0.5)
web_scene.defaultCamera().pointCameraAt([0.5, 3.0, 2.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0])
proxy_scene = mb.getScene()

# Visuals
rod_geom: BoxGeometry = BoxGeometry(ROD_LEN, 0.05, 0.05)
bead_geom: BoxGeometry = BoxGeometry(0.08, 0.08, 0.08)
force_marker_geom: BoxGeometry = BoxGeometry(0.035, 0.035, 0.035)

mat_info: PhysicalMaterialInfo = PhysicalMaterialInfo()
mat_info.color = Color.FIREBRICK
brown: PhysicalMaterial = PhysicalMaterial(mat_info)
mat_info.color = Color.GOLD
gold: PhysicalMaterial = PhysicalMaterial(mat_info)
mat_info2: PhysicalMaterialInfo = PhysicalMaterialInfo()
mat_info2.color = Color.LIME
lime: PhysicalMaterial = PhysicalMaterial(mat_info2)

scene_parts: list[Any] = []

bd1_body: ProxyScenePart = ProxyScenePart("bd1_body", scene=proxy_scene, geometry=rod_geom, material=brown)
bd1_body.attachTo(bd1)
bd1_body.setTranslation([ROD_LEN * 0.5, 0.0, 0.0])
scene_parts.append(bd1_body)

bd2_body: ProxyScenePart = ProxyScenePart("bd2_body", scene=proxy_scene, geometry=rod_geom, material=brown)
bd2_body.attachTo(bd2)
bd2_body.setTranslation([ROD_LEN * 0.5, 0.0, 0.0])
scene_parts.append(bd2_body)

if bd2_pnode is not None:
    bead0: ProxyScenePart = ProxyScenePart("bead_pnode", scene=proxy_scene, geometry=bead_geom, material=gold)
    bead0.attachTo(bd2_pnode)
    bead0.setTranslation([0.0, 0.0, 0.0])
    scene_parts.append(bead0)

if bd2_tip_onode is not None:
    bead1: ProxyScenePart = ProxyScenePart("bead_tip_onode_visual", scene=proxy_scene, geometry=bead_geom, material=gold)
    bead1.attachTo(bd2_tip_onode)
    bead1.setTranslation([0.0, 0.0, 0.0])
 
# Propagator
sp: StatePropagator = StatePropagator(
    mb,
    IntegratorType.RK4,
    None,
    None,
    SpSolverType.TREE_AUGMENTED_DYNAMICS,
)
integrator = sp.getIntegrator()

# Pins
bd1_pin: PinSubhinge = cast(PinSubhinge, bd1.parentHinge().subhinge(0))
bd2_pin: PinSubhinge = cast(PinSubhinge, bd2.parentHinge().subhinge(0))

# Models
hinge_limits: HingeSoftLimits = HingeSoftLimits("hinge_soft_limits", sp, bd1_pin, bd2_pin)
sp.registerModel(hinge_limits)

freeze_model: FreezeHingesModel = FreezeHingesModel("freeze_hinges", sp, bd1_pin, bd2_pin)
sp.registerModel(freeze_model)

tip_force_xy: TipForceXYToHingeTorque = TipForceXYToHingeTorque("tip_force_xy", sp,bd1_pin, bd2_pin, ROD_LEN)
sp.registerModel(tip_force_xy)

force_driver: ForceDriver = ForceDriver("force_driver", sp, tip_force_xy)
sp.registerModel(force_driver)

gravity_model: UniformGravity = UniformGravity("grav_model", sp, mb)
gravity_model.params.g = np.array([0.0, -9.81, 0.0], dtype=float)

_update_scene: UpdateProxyScene = UpdateProxyScene("update_proxy_scene", sp, proxy_scene)
_sync_rt: SyncRealTime = SyncRealTime("sync_real_time", sp, 1.0)

# Hard clamp last so it always wins (prevents self-overlap)
hinge2_clamp: HardClampHinge = HardClampHinge("hinge2_clamp", sp, bd2_pin, HINGE2_Q_MIN, HINGE2_Q_MAX)
sp.registerModel(hinge2_clamp)


# -----------------------------
# Helpers
# -----------------------------
def push_state() -> None:
    sp.setState(sp.assembleState())
    proxy_scene.update()


def set_hinges(q1: float, u1: float, q2: float, u2: float) -> None:
    bd1_pin.setQ(q1)
    bd1_pin.setU(u1)
    bd2_pin.setQ(q2)
    bd2_pin.setU(u2)


def set_modes(q: np.ndarray, u: np.ndarray) -> None:
    nm: int = int(bd2.nU())
    bd2.setQ(np.asarray(q, dtype=float).reshape((nm, 1)))
    bd2.setU(np.asarray(u, dtype=float).reshape((nm, 1)))


def enable_modal_damping(c0: float) -> None:
    nm: int = int(bd2.nU())
    bd2.setDampingVector(np.full((nm, 1), c0, dtype=float))


def zero_modes() -> None:
    nm: int = int(bd2.nU())
    set_modes(np.zeros((nm, 1), dtype=float), np.zeros((nm, 1), dtype=float))


def pluck_mode0(q_amp: float, u_amp: float) -> None:
    nm: int = int(bd2.nU())
    q: np.ndarray = np.zeros((nm, 1), dtype=float)
    u: np.ndarray = np.zeros((nm, 1), dtype=float)
    q[0, 0] = q_amp
    u[0, 0] = u_amp
    set_modes(q, u)


def run_phase(
    label: str,
    duration_s: float,
    pre_step: Optional[Callable[[], None]] = None,
    print_every_s: float = 0.10,
) -> None:
    print(f"\n===== {label} =====")

    if pre_step is not None:
        pre_step()
        push_state()

    t0_s: float = float(integrator.getTime()) / 1e9
    t_end_s: float = t0_s + duration_s

    def cb(_: object) -> None:
        t_s: float = float(integrator.getTime()) / 1e9
        q2: float = _scalar(bd2_pin.getQ())
        u2: float = _scalar(bd2_pin.getU())
        mode0_q: float = _scalar(bd2.getQ()[0])
        mode0_u: float = _scalar(bd2.getU()[0])
        fx: float = float(tip_force_xy.force_xy[0])
        fy: float = float(tip_force_xy.force_xy[1])
        print(
  f"t={t_s:7.2f} mode={FORCE_COMPONENT_MODE:<6} phase={force_phase:<4} "
  f"hinge2_q={q2:+.3f} hinge2_u={u2:+.3f} "
  f"mode0_q={mode0_q:+.6f} mode0_u={mode0_u:+.6f} "
  f"Fxy=[{fx:+.1f},{fy:+.1f}]"
)

    h_ns: int = int(print_every_s * 1e9)
    now_ns: int = int(integrator.getTime())
    first_fire: np.timedelta64 = np.timedelta64(now_ns + h_ns, "ns")

    ev: TimedEvent = TimedEvent(f"phase_print_{label}", first_fire, cb, False)
    ev.period = np.timedelta64(h_ns, "ns")
    sp.registerTimedEvent(ev)

    sp.advanceTo(t_end_s)
    del ev


# -----------------------------
# Initial pose sanity checks
# -----------------------------
sp.setTime(np.timedelta64(0, "ns"))
sp.setState(sp.assembleState())

print("\n--- BUILD NOTE ---")
print("This build does NOT consume Node/body external wrenches from Python.")
print("We fake tip forces via hinge torque")
print(f"Hinge2 is limited to [{HINGE2_Q_MIN:+.2f}, {HINGE2_Q_MAX:+.2f}] with soft stops + hard clamp.\n")

set_hinges(0.0, 0.0, 0.0, 0.0)
zero_modes()
push_state()
input("Pose check 1: straight 2-link line. Press Enter...")

bd2_pin.setQ(pi / 3.0)
push_state()
input("Pose check 2: link2 rotates about end of link1. Press Enter...")

set_hinges(0.2, 0.0, 0.5, 0.0)
push_state()
input("Pose check 3: both rods point +X in their own frames. Press Enter...")

pluck_mode0(q_amp=0.2, u_amp=0.2)
enable_modal_damping(0.50)
set_hinges(0, 0, 0, 0)
push_state()
input("Modal check: bead should wiggle + visibly offset in +Y (visual bend hack). Press Enter...")


# -----------------------------
# Phases
# -----------------------------
def phase_a() -> None:
    global freeze_hinges_enabled, force_phase
    freeze_hinges_enabled = False
    force_phase = "NONE"
    gravity_model.params.g = np.array([0.0, -9.81, 0.0], dtype=float)

    set_hinges(q1=0.0, u1=0.0, q2=0.0, u2=0.0)
    zero_modes()
    enable_modal_damping(0.0)

    tip_force_xy.enabled = False
    tip_force_xy.force_xy[:] = 0.0


def phase_b() -> None:
    global freeze_hinges_enabled, force_phase
    freeze_hinges_enabled = True
    force_phase = "NONE"
    gravity_model.params.g = np.array([0.0, 0.0, 0.0], dtype=float)

    set_hinges(q1=0.0, u1=0.0, q2=0.0, u2=0.0)

    # modal-only, exaggerate
    enable_modal_damping(0.01)
    pluck_mode0(q_amp=0.2, u_amp=0.0)

    tip_force_xy.enabled = False
    tip_force_xy.force_xy[:] = 0.0


def phase_c() -> None:
    global freeze_hinges_enabled, force_phase
    freeze_hinges_enabled = False
    force_phase = "NONE"
    gravity_model.params.g = np.array([0.0, -9.81, 0.0], dtype=float)

    set_hinges(q1=0.2, u1=0.0, q2=0.0, u2=0.0)
    enable_modal_damping(0.5)
    pluck_mode0(q_amp=0.05, u_amp=0.05)

    tip_force_xy.enabled = True
    tip_force_xy.force_xy[:] = [0.0, 10.0]


def phase_d() -> None:
    global freeze_hinges_enabled, force_phase
    freeze_hinges_enabled = False
    force_phase = "D"
    gravity_model.params.g = np.array([0.0, 0.0, 0.0], dtype=float)

    set_hinges(q1=0.0, u1=0.0, q2=0.0, u2=0.0)
    zero_modes()
    enable_modal_damping(0.02)

    tip_force_xy.enabled = True
    tip_force_xy.force_xy[:] = 0.0


def phase_e() -> None:
    global freeze_hinges_enabled, force_phase
    freeze_hinges_enabled = False
    force_phase = "E"
    gravity_model.params.g = np.array([0.0, 0.0, 0.0], dtype=float)

    set_hinges(q1=0.0, u1=0.0, q2=0.0, u2=0.0)
    zero_modes()
    enable_modal_damping(0.02)

    tip_force_xy.enabled = True
    tip_force_xy.force_xy[:] = 0.0
def phase_f_fx_only_constant() -> None:
    global freeze_hinges_enabled, force_phase, FORCE_COMPONENT_MODE
    freeze_hinges_enabled = False
    force_phase = "NONE"
    FORCE_COMPONENT_MODE = "X_ONLY"
    gravity_model.params.g = np.array([0.0, 0.0, 0.0], dtype=float)

    # IMPORTANT: make Fx create torque by rotating the rod away from +X
    set_hinges(q1=pi / 2.0, u1=0.0, q2=0.0, u2=0.0)

    zero_modes()
    enable_modal_damping(0.02)

    tip_force_xy.enabled = True
    tip_force_xy.force_xy[:] = [PHASE_F_FX_CONST, 0.0]


def phase_g_fy_only_constant() -> None:
    global freeze_hinges_enabled, force_phase, FORCE_COMPONENT_MODE
    freeze_hinges_enabled = False
    force_phase = "NONE"
    FORCE_COMPONENT_MODE = "Y_ONLY"
    gravity_model.params.g = np.array([0.0, 0.0, 0.0], dtype=float)

    # For Fy, rod can be horizontal (theta ~= 0) and you get tau_z ~= L*Fy
    set_hinges(q1=0.0, u1=0.0, q2=0.0, u2=0.0)

    zero_modes()
    enable_modal_damping(0.02)

    tip_force_xy.enabled = True
    tip_force_xy.force_xy[:] = [0.0, PHASE_G_FY_CONST]


input("\nREADY CHECK: press Enter to start the simulation phases...")

run_phase("A_rigid_only", duration_s=10.0, pre_step=phase_a)
run_phase("B_modal_only_frozen_hinges (exaggerated)", duration_s=10.0, pre_step=phase_b)
run_phase("C_coupled", duration_s=15.0, pre_step=phase_c)
run_phase("D_force_sweep_XY (slow)", duration_s=15.0, pre_step=phase_d)
run_phase("E_force_circle_XY (slow)", duration_s=15.0, pre_step=phase_e)
run_phase("F_constant_Fx_only (world)", duration_s=8.0, pre_step=phase_f_fx_only_constant)
run_phase("G_constant_Fy_only (world)", duration_s=8.0, pre_step=phase_g_fy_only_constant)

input("Done. Press Enter to quit...")
sp.dump("sp")


# -----------------------------
# Cleanup
# -----------------------------
def cleanup() -> None:
    global force_marker, proxy_scene, web_scene, scene_parts
    try:
        force_marker = None
    except Exception:
        pass

    try:
        for p in scene_parts:
            try:
                del p
            except Exception:
                pass
        scene_parts = []
    except Exception:
        pass

    try:
        del web_scene
    except Exception:
        pass
    try:
        del proxy_scene
    except Exception:
        pass

    try:
        cleanup_graphics()
    except Exception:
        pass

    try:
        discard(sp)
    except Exception:
        pass
    try:
        discard(mb)
    except Exception:
        pass
    try:
        discard(fc)
    except Exception:
        pass


atexit.register(cleanup)
