# Dart De-orbit through VLEO (2026-11)

How would a dart, dropped from 500 km, fall through VLEO (300 km down to through 80-120km, then down to the ground)?

* `vleo_dart_sim.py`: 6-DOF model (3D orbit at any inclination, full rigid-body attitude with
  quaternion + body rates, gravity-gradient torque vector, roll/yaw aero incl. optional fin cant),
  plus a Monte Carlo (`mc`) over release attitude and tip-off rates.
* `vleo_dart_sim_v1.py`: earlier planar checkpoint (equatorial orbit, pitch only).

```sh
uv run vleo_dart_sim.py run --incl 51.6 --aoa0 90 --yaw0 10 --rates0 0.5 -0.5 1
uv run vleo_dart_sim.py plot && uv run vleo_dart_sim.py video
uv run vleo_dart_sim.py mc --alts 500 300 200 --n 32 --reentry-from 130
```

Things the 6-DOF model shows that the planar one could not:

* Large planar pitch swings with the fins in "+" orientation are unstable to out-of-plane
  motion (pitch and yaw share a natural frequency, and the cruciform's yaw stiffness varies over
  the swing), so energy pumps from round-off into yaw within about an orbit. In "×"
  orientation, or with no fins, it does not happen.
* Once the (balsa) fins burn off, the bare body trims at ~12° total angle of attack. In the planar
  model that lift always points up and the dart skips; in 6-DOF its direction is free.
* Fin cant spins the dart up toward cant × V / fin radius (hundreds of rad/s at orbital speed),
  which makes canted-fin release-to-ground runs very slow.
