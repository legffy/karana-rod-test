# karana-rod-test# Karana Rod / Modal Sandbox

This repo tests the Karana dynamics library on a simple multibody:
- rigid rod (bd1) -> modal rod (bd2) -> tiny tip body
- hinge limits to avoid self-overlap
- modal “wiggle” 
- attempted external force at tip (workaround used)

## What works
- Multibody builds and sim runs
- Modal coordinate excitation produces visible motion (phase B)
- Hinge limits prevent collision/overlap

## Known limitation (important)
This Karana build does **not** appear to consume external wrenches/forces from Python (at least via the APIs available here).
So the code uses a workaround: it converts an intended tip force into hinge torque:
- `tau_z = L * Fy` (for r=[L,0,0], F=[Fx,Fy,0])

## Run
Prereq: Karana must be installed and importable 
Guide to getting a license and installation can be found here: https://portal.karanadyn.com/docs/latest/getting_started.html

```bash
python3 rod.py
