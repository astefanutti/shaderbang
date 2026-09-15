# Plasma globe — design record

Physically-based, interactive, GLSL-path-traced plasma globe (`examples/plasma_globe.py`), built
like `examples/cloth.py`: NVIDIA Warp physics driven through shaderbang's `Input` lifecycle,
evdev touch/mouse/trackpad/keyboard, one CUDA graph per frame, Warp↔GL interop, GLSL 4.60
compute rendering with a 2× temporal upscale. No OptiX.

The approved design (physics, rendering passes, orchestration, milestones, validation matrix,
risks) is the plan of 2026-09-13; this document records what was **measured and decided** as the
milestones land. Numbers are from the target box: RTX 5090 (sm_120, 32 GB), driver 575.57.08,
CUDA 12.9, free-threaded CPython 3.13.3t.

## Operating point (provenance-tagged)

| Symbol | Value | Tag / source |
|---|---|---|
| R1 electrode bulb | 0.015 m | MEASURED — PPPL-4485 (Campanell et al. 2010): 1.5-2 cm |
| R2i / R2o glass | 0.075 / 0.0775 m (2.5 mm soda-lime, IOR 1.52, ε_r 6) | PPPL Table 1 / CHOSEN thickness |
| Gas | Ne + 2 % Xe, 740 Torr, T0 = 300 K | MEASURED — PPPL-4485 |
| Drive | 26 kHz, 5 kV peak, ~1 mA total | MEASURED — PPPL-4485 |
| Filaments | 6-8 cm × ~1 mm, 10-30 simultaneous, rise ~1 cm/s | MEASURED — PPPL-4485 |
| ρ0, cp, k, ν, α | 0.798 kg/m³, 1030 J/kgK, 0.0491 W/mK, 3.97e-5 m²/s, 5.97e-5 m²/s | DERIVED (Ne, 740 Torr, 300 K) |

Why near-atmospheric matters: Ra ∝ p². At 5 Torr Ra ≈ 60 (no convection) and a 1 mm channel
cools in 24 µs; at 740 Torr Ra ≈ 1e6 and the hot channel persists 1-4 ms ≈ 40-200 drive
half-cycles — the thermal channel is the re-strike memory.

## Milestone log

### M0 — Toolchain (2026-09-13)

- **ABI mismatch fixed.** The installed `shaderbang/_shaderbang.so` had been built from branch
  `pdt-collision` (80-byte `struct options` with a trailing `bool triple_buffer`) while `main`'s
  ctypes `OPTIONS` was 72 bytes → `init()` read one byte past the allocation. Commit `70f2182`
  ("Add opt-in triple buffering to the DRM render loops") was ported to this branch and the
  library rebuilt with `make`; `ctypes.sizeof(OPTIONS) == 80`.
- **Warp pinned to 1.17.0** (PyPI wheel, CUDA 12.9 / driver ≥ 525; never the `+cu13` wheel,
  which needs driver 580+). Verified in a throwaway venv, then upgraded `.venv`: imports on the
  free-threaded interpreter with the GIL disabled, toolkit (12, 9) / driver (12, 9), conditional
  graphs supported, `wp.capture_while` runs (5 iterations, correct result), and an allocation
  inside a conditional body now raises `Conditional body graph contains an unsupported operation
  (memory allocation)` instead of silently corrupting (the GH-1641 class of failure). New APIs
  present: `wp.GLTextureResource`, `wp.Texture3D`, `HashGrid.build(points, radius, groups=None)`,
  `Volume.allocate_by_voxels(..., rebuildable=...)`.
- **Script A — CUDA-graph node cost** (200 replays, graphs of K identical kernel nodes):

  | kernel | 1.13.0 slope | 1.17.0 slope | intercept |
  |---|---|---|---|
  | no-op, dim 1 | 0.57 µs/node | 0.53 µs/node | ~0-5 µs/launch |
  | 64k-thread early-out kernel | 1.00 µs/node | 0.92 µs/node | 2-4 µs/launch |

  A ~170-node frame graph therefore costs ~0.1-0.2 ms of launch floor → **S_MAX = 16** growth
  steps per frame is affordable (the plan's 2-4 µs/node assumption was pessimistic by 3-4×).
- **Headless GL for tests and benchmarks.** GNOME (gdm3, Wayland) owns the DRM device on this
  box, so shaderbang apps cannot be run from an agent session. A desktop GL 4.6 compatibility
  context is available without DRM master through `EGL_EXT_platform_device` +
  `EGL_KHR_surfaceless_context` (PyOpenGL, `setPlatform("egl")`): `GL_VERSION 4.6.0 NVIDIA
  575.57.08`, GLSL 4.60, `GL_MAX_IMAGE_UNITS 8`, 1024 compute invocations/group; a 1920×1080
  RGBA16F FBO is complete; `GL_TIME_ELAPSED` queries work (clear = 9 µs); a
  `wp.RegisteredGLBuffer` over an SSBO round-trips through Warp. This is the substrate for
  `examples/plasma_globe_bench.py` (M1) and for headless render tests (M5).
- Displays: DP-2 (native 6144×3456) and HDMI-A-1 (3840×2400) both offer **3840×2160** → the
  example is run with `--mode 3840x2160`; internal resolution = mode / 2.

### Circuit, surface charge, lifecycle — `examples/plasma/circuit.py` (2026-09-13)

- **Source capacitance calibrated once:** C_e = 3.32 pF so that 20 untouched 6 cm filaments at
  5 kV / 26 kHz draw exactly 1.000 mA (PPPL Table 1). C_d = 21.2 nF/m², σ_sat(5 kV) = 106 µC/m².
- **Emergent filament count** (admission while the projected share of one more filament ≥ the
  strike current, I_str = 40 µA at 26 kHz):

  | V (kV) | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 |
  |---|---|---|---|---|---|---|---|---|
  | N | 0 | 0 | 0 | 0 | 14 | 27 | 29 | 29 |

  Threshold ≈ 3 kV (60 % of nominal — PPPL: breakdown at ~70 % of the knob, sustain ~55-60 %),
  linear rise, saturation at 29 set by the 1.2 mA supply limit. Nothing prescribes N.
- **Touch:** one finger on a foot (×214 termination admittance) → that filament carries 80 % of
  I_tot (1.2 mA) and every other filament falls to 12.7 µA < I_sustain = 32 µA, so they retract
  through the 3-frame latch — the measured "touch brightens one, dims the rest".
- **DECISION — surface charge as analytic per-foot records, not a 256×128 texture.** Each
  attached foot owns a record {direction, amplitude (units of σ_sat), ring radius, age}: the
  amplitude relaxes exactly to r/(r + d) (saturates ≈ 1 within a second; the finger drains it
  with τ_finger = 5 ms), the activator core has a fixed width of 1.5 foot radii, and the
  inhibition ring starts at r_inh = 8 mm and expands as √(r_inh² + 2·D_s·t) up to 15 mm (Burin's
  expanding circular structures); detached feet move to an orphan ring buffer and decay with
  τ_env = 90 ms. Reasons: explicit diffusion on a 256×128 equirect map is unstable at dt = 1/60
  (D·dt/dx² ≈ 1.2), the map is singular at the poles where filaments die, and with ≤ 64 feet the
  growth weight s(x) = 1 + A·act − B·ring is an exact analytic sum. Measured probe on a saturated
  foot: s = 1.17 at the footprint, 0.39 on the ring, 1.04 at 4.5 cm, 1.00 in the interior.
- **Lifecycle marking** verified on a synthetic tree (100-node chain + 20-node side branch):
  a re-route at u = 0.5 flags exactly the 50 main-channel nodes beyond the fork plus the whole
  side branch (70 nodes) as decaying, records the fork node, and the decaying pool frees them
  after 0.1 s; retraction advances the cut by 2 cm per frame.

### Publish stage — `examples/plasma/publish.py` (2026-09-13)

- One 80-byte segment record per live node (stable ids → exact previous positions for motion
  vectors), Kim-Lin channel classes from a per-tree main-channel walk + descendant counting
  (main / secondary / side = 1 : 0.25 : 0.06), a 96³ uniform-grid CSR (count → exclusive scan →
  scatter, dilation 1 cell, no sort), per-tree 8-span polyline lights carrying the summed member
  power (conserved to 1e-6), a wide-Gaussian ambient splat and the float16 RGBA volume pack.
- **DECISION:** the render buffers stay separate typed `wp.array`s, each backing its own GL
  buffer, mapped with one batched `cuGraphicsMapResources` per frame (0.19 ms measured for 8
  resources) instead of one interleaved SSBO.
- Measured on a synthetic 13×150 star (853 segments): whole publish graph **0.48 ms/frame**;
  CSR occupancy 1.33 % of cells, ≤ 10 segments per cell (mean 3.0), 35k items of a 221k cap.

### Tracer — `examples/plasma/shaders/trace.comp` + `examples/plasma/headless_render.py` (2026-09-13)

- Headless render of the synthetic 13×150 star (853 segments) at 1920×1080 through the EGL
  device context: **trace.comp median 0.96-1.04 ms** (deterministic 3-branch glass, DDA over the
  96³ CSR, closed-form capsule emission, 32-step volume march, polyline line lights, layer ids +
  motion vectors). Candidates per interior ray: **mean 4.5, p99 12**, the 270 cap never hit; no
  NaNs. `python -m plasma.headless_render --out /tmp/plasma_trace.png` writes beauty / emissive /
  layer PNGs and the candidates-per-ray histogram.
- **Two pitfalls found and fixed:**
  1. The naive antiderivative of the capsule emission integral (terms in `(2At+B)/(D·R^k)`)
     reaches ~1e13 for 0.5 mm filaments and cancels catastrophically in float32 — thin cores
     rendered as centimetre-wide fog. The shader now evaluates the integrals from the closest
     approach with the complements `T3(x) = 1/(q(q+x))`, `T5(x) = (2q+x)/(3q³(q+x)²)`
     (`q = √(x²+1)`, `x = s·√(A/P)`), which never cancel; verified against brute-force
     quadrature to 1.6e-4 relative error in float32 over 4000 random rays.
  2. Passing a `uint16` view of a float16 volume to PyOpenGL's `glTexSubImage3D` makes the
     binding convert the integers numerically (the mask's 1.0 became 15360) — upload the float16
     array itself.
- **DECISION:** the core uses the compact 5/2 profile `e ∝ 3ε³/(2π)·(d²+ε²)^{-5/2}` (closed form:
  `2(2At+B)/(3DR^{3/2}) + 16A(2At+B)/(3D²√R)`, then stabilised as above) and only the faint
  sheath (2 %, radius 3ε) keeps the 3/2 profile: the 3/2 tail (~1/d²) summed over hundreds of
  segments filled the globe with fog.
- Interior nodes receive exactly one hemispherical cap (the end cap of their incoming segment);
  the start cap is added only at chain starts, which removed the periodic beading at node
  spacing. Residual beading at sharp kinks is a publish-stage polyline-smoothing item (M7).

### Renderer pipeline — `examples/plasma/renderer.py` (2026-09-13, headless, synthetic star)

`python -m plasma.headless_render --full` drives the real `Renderer` Input into an offscreen
3840×2160 "display" framebuffer (internal 1920×1080), orbiting the camera 0.5°/frame:

| pass | GPU time |
|---|---|
| P0b upload (7 separate `wp.RegisteredGLBuffer` map/unmap pairs + PBO → 3D texture) | 1.10 ms |
| P1 trace.comp | 0.99 ms |
| P2 glow pyramid (6 levels) | 0.06 ms |
| P4 exposure (reduce + resolve) | 0.01 ms |
| P3 TAAU 1080p → 4K | 0.26 ms |
| P5 present (AgX + dither, 4K) | 0.04 ms |
| **total** | **2.46 ms** (wall 2.53 ms incl. `glFinish`) |

The upload is now the largest render-side cost, exactly the per-pair interop floor the research
measured (~0.145 ms × 7); the batched `cuGraphicsMapResources` path (M1 GL module) is the fix.
Also found: the venv's editable `shaderbang` install pointed at a deleted checkout
(`~/Development/kms-glsl`), so `import shaderbang` failed in `.venv`; re-installed with
`uv pip install -e .` from this repository.
- With the GL module's batched `cuGraphicsMapResources` (one map/unmap for 6 SSBOs + the 3D
  texture, `Texture3D.copy_from` for the volume): upload **1.10 → 0.28 ms**, render total
  **1.63 ms GPU / 1.93 ms wall** per 4K frame on the synthetic star.

### Coupled simulation — `examples/plasma/globe.py` (2026-09-14, headless `python -m plasma.headless_render --sim`)

Frame order (one CUDA graph, captured once): params upload → gas control → gas step → circuit →
engine hooks (fingers, admission, per-tree current) → candidate inputs (σ weight, gas T) →
`Dbm.step()` (link subdivision, prev_pos snapshot, lifecycle, re-base, 16 growth steps, CG) →
node advection → tree geometry (foot/root dirs, chord, stretch) → node alpha / heat sources /
x_ion → publish (segments, CSR, lights) → gas volume pack → counters (pinned).

Problems found in the first coupled runs and what fixed them (all in the physics, none cosmetic):

| Symptom | Cause | Fix (CHOSEN) |
|---|---|---|
| Growth froze, candidate potentials ~1e11, admissions stopped | advection packed same-tree nodes closer than the conductor radius a (0.375 mm): the 1/r system lost positive definiteness and the warm-started PCG diverged (\|q\| 1e-4 → 1e8 in 15 frames) | node charge kernel = inverse multiquadric 1/sqrt(r² + a²) (self term 1/a unchanged, far field unchanged, strictly positive definite for any node set); lab check 2 (incremental Kim Eq.11) passes at 2.5e-7 |
| Boom-bust population (10 → 0 → 12 → 0) | a re-route moved the tree to GROW (no current) and cohorts seeded together re-routed together; ~27 seeds while nothing was attached | re-route *in place*: the tree stays ATTACHED (current, heat, foot) while a leader grows from a fork; the old distal channel fades only when the leader reaches the glass; admission is sequential (a new strike only while < 2 trees are growing, `MAX_GROWING`) |
| Leaders never attached once channels were long; stretch 20–1000, main channels of 1000 nodes, N_MAX hit | the still-charged folded old channel screened every fork; the hand-over had no bound | grace period `REGROW_TIMEOUT` 0.25 s: after it the stretched channel extinguishes anyway (pruned, tree → GROW), a starved leader is re-seeded from a new fork instead of giving up |
| Trees kept 2/3 of their nodes as side branches that stretched forever | side branches carry no current, so nothing ever removed them | the frame after any attachment every charged node off the main channel enters the decaying pool (only the conducting channel persists; the strike's branches fade over 100 ms) |
| Straight centimetre-long chords (up to 9 cm) across the globe | the polyline nodes were advected apart in the shear layers; the CSR rasteriser then paid 7 ms for their AABBs | links longer than 2 h are split at their midpoint each frame (material line), stencil radius shrinks to the wall distance near the electrode/glass, CSR rasterises long segments by sampling (cap 400 nodes/tree) |
| Re-routes forked distal of the stretched part, leaving 3-node main channels | fork_s drawn in [0.35, 0.75] L regardless of where the stretch was | stretch trigger forks at the most distal main-channel node whose root-side geometric length ≤ 1.2× its arc length at birth, else re-strikes from the root |
| Trees seeded next to attached ones stalled | propagation gate 0.25·E_bd0 too high in the screened field | `E_PROP_FACTOR` 0.05 |
| Lightning-like bushes at η 3 | the lab's D 1.7 regime | globe default η 6 (knob `E`); the post-attachment prune leaves the rope |

Measured (RTX 5090, 64³ gas, 8192 nodes, 32 trees, S_MAX 16, seed 1, 600 frames):

| Quantity | Value |
|---|---|
| Sim graph (replay, synced) | 1.55 ms median |
| Whole 4K frame (sim + render + upload, headless FBO) | 3.5 ms median, 5.0 ms p99 |
| Trace / TAAU / upload | 0.98 / 0.24 / 0.37 ms |
| Attached filaments at 4 / 5 / 6.5 / 8 kV (frame 600) | 9 / 12–13 / 19 / 25 (I_tot 0.45 / 0.79 / 1.20 / 1.20 mA: supply-capped from 6.5 kV) |
| Main-channel stretch in steady state | 1.0–1.6 (trigger 1.5), links ≤ 4.6 mm, 0 dangling parents |
| Nodes per attached filament | 64–104 (main channel only) |
| Deaths in 900 frames at 5 kV | 0 retractions; 2 grace time-outs |
| Touch (finger at +z at frame 300) | 12 filaments → 1 within 60 frames carrying 1.18 mA, then 2 at 1.20 mA (others fall under I_sus and retract) |
| Peak gas speed | 0.12 m/s (merged plume; channels re-strike every ~0.5–2 s) |

Emergent, not prescribed: the count law above (field screening at the electrode limits N at 5 kV,
the 1.2 mA supply from 6.5 kV), the touch capture, and the re-strike cadence. Open for the look
phase: filament smoothness (1.5 mm polyline with Gumbel jitter reads jagged), foot-glow gain
(disc ×1.5 now), gas heating power vs the ~1 cm/s observed rise (plume 3–12 cm/s here).


Validation toggles (headless `--sim`, 600 frames, seed 1, 5 kV):

| Case | Attached | Mean live-node y | Heat centroid y (box fraction, +up) |
|---|---|---|---|
| upright | 12 | +0.58 cm | +0.083 |
| `--invert` (gravity flipped in globe space = globe upside down) | 13 | −0.55 cm | −0.083 (heat still rises in the world frame) |
| `--ice` (20° cap at +y, T_amb − 20 K) | 12 | −0.58 cm | +0.071 (channels bent away from the cap, weaker plume) |

Stability: 3600 frames (60 s) at 5 kV: 12 → 16 attached, 0 NaN, sim 1.63 ms median, whole 4K
frame 3.34 ms median / 4.99 ms p99 (headless FBO). `plasma_globe.py --test` runs the same
invariants in the app every 60 frames (finite positions, node pool, SEG_MAX, dangling parents,
candidate cap < 0.1 %, engine/counter agreement) and exits 2 with a state dump on the first
violation. The app initialises end to end on the real GL context (graph capture + renderer) up
to the DRM mode set, which needs a bare VT.

Rendering: filament polylines are corner-cut (one Chaikin pass in `k_segments`, previous
positions cut the same way for the motion vectors), which halves the 1.5 mm growth zigzag
without touching the physics.

### Second pass after the first display review (2026-09-14)

Review on the 4K display: "paths not smooth enough, the root should move with the convection,
several fingers do not each get a path, the glow at the electrode and on the sphere is not
convincing, the old procedural shader looks closer to a real globe". Root causes found and
what changed (physics first, then rendering):

| Report | Cause (measured) | Change (CHOSEN) |
|---|---|---|
| Channels 1.6–1.9× longer than their chord at *any* η (6, 10, 14 gave 1.75–2.06) | two things: the K = 8 candidates of a node were the octant directions of a *random* rotation, so a straight continuation rarely existed (mean turn 43°, p90 80°); and a tip's straight candidate exceeds its 35° neighbour by only ~1 % of the pool-normalised potential (candidate dump: Φ 0.99 / 0.98 / 0.94), which the Gumbel noise swamps below η ≈ 100 | candidates now span a forward cone about the growth direction (straight, 3 × 35°, 4 × 80°, random azimuth phase, `cone_dir`); globe η = 120 with the physical γ = 1, so the weight is (E·T/T0)^η: a streamer resolves 1 % field differences because ionisation is exponential in the field. Lab: η 120 → tortuosity 1.09, single 44-node channels, turn median 0°; η 300 → 1.00; η 3 stays the D 1.7 lightning regime (knob `E`) |
| Every channel sucked into the plume core above the electrode; 10–26 cm channels | the material-line advection was uniform, but near the electrode the field (∝ 1/r², 25× the glass value) dictates the re-struck path; the ~3 % breakdown advantage of the hot channel cannot hold it off the field lines there | channel drift weighted by (r/R2)³ (`DRIFT_EXPONENT`); roots slide on the electrode with the flow sampled 4 h above it; gas heating 0.4 W at 1 mA; peak gas speed 3.3 cm/s and mid-gap channel drift ~1 cm/s (PPPL's rise) |
| A second finger gets no filament | the touched tree took ~all the budget and the nominal admission (untouched projection) failed; nearby feet were also excluded by their own σ rings | unserved fingers (no attached foot within the contact half width) admit a strike on the touched admittance (`circuit[7]`); attached trees within 0.6 rad of an unserved finger get a re-route request (engine trigger b, `t_reroute_req`); the σ inhibition fades under a contact; `I_SUSTAIN_RATIO` 0.2 so the other filaments dim to ~10 % instead of vanishing. Test: two fingers at +z and +x → feet 1.8° and 2.9° away, 406 / 411 µA |
| Hairy strike bushes | streamer branches carry displacement current only | classes main / secondary / side = 1 / 0.10 / 0.02 |
| Electrode and sphere glow | a 1.5 mm corona shell and a grey volume splat | electrode halo ∝ I_tot · (R1/r)⁴ (j·E around a sphere) with a 1 mm sheath layer and an emissive electrode surface (grazing-brightened), filament sheath 2.2 mm carrying 35 % of the light, violet-tinted ambient splat (8 mm) plus a faint background discharge ∝ I_tot, softer 2.5 mm foot discs, glow gain 1.5, exposure bias +0.5 EV, `gas.x` in the UBO carries I_tot (mA) |

Measured after the pass (seed 1, 600 frames, 5 kV unless noted): 12 attached, 653 nodes
(52 per channel), tortuosity median 1.31 (advection + re-routes; 1.09 at birth), ~6 re-routes/s
(timer-dominated), no dangling parents, render 1.44 ms per 4K frame; N(V) = 8 / 12 / 15 / 18 at
4 / 5 / 6.5 / 8 kV (supply-capped at 1.2 mA from 6.5 kV); all four engine lab checks and the
circuit / publisher self-tests pass. Reference frame of the old shader for comparison:
`/tmp/plasma_dbg/ref_render.py` renders `examples/plasma_globe.glsl` headlessly.

### Third pass — matching the reference footage (2026-09-14)

Screenshots of a real globe (`~/Pictures/Screenshots/`, from the user's reference video) fixed the
targets: thin blue-violet filaments with tight halos and a wide brightness spread, the touched
filament white and ~3x thicker ending in a spray of pink branches under the fingers, a pink
translucent electrode with a bright rim, small pink tips at the glass, and almost no interior
haze. Changes, all driven by the per-filament current:

| Element | Change |
|---|---|
| Core radius | `r = 0.45 mm sqrt(I / 50 uA)`, clamped 0.2–2 mm (the old law went as I^-1/4, the wrong way); sheath radius = 4 r, 25 % of the light |
| Foot | size `1.5 mm sqrt(I / 50 uA)` (1.5–8 mm) and brightness ∝ I from the record's current (`sig_I`, packed as uA); orphan records keep their charge but no glow |
| Foot brush | branches that leave the main channel within 12 mm of the glass survive the post-attachment prune (`FOOT_BRUSH`) and render as secondary class 0.3 |
| Beading | interior joints no longer get end caps (`capsuleEmission(..., capA, capB)`, caps only at feet): the cap of span k and the body of span k+1 double-counted a sphere at every node |
| Electrode | pink emissive shell with the limb law `0.06 + 3 (1 - mu) + 3 (1 - mu)^4` (dark ball, bright rim: rim/centre ~2x after AgX), violet (R1/r)^4 halo reduced, pink 1 mm sheath layer |
| Haze | ambient gain 4e-3, background discharge 0.004 I_tot, violet tint; glow kernel 24 px at gain 0.6 (a tight camera PSF; the halo is the physical sheath) |
| Colours | shafts x_ion 0.6–1.0 (bluer), feet magenta |
| Dynamics | drift (r/R2)² with 0.5 W heating (the previous pass was judged too slow), I_STRIKE 30 uA |

Two-finger check unchanged (feet 4.2° from both fingers, 537 / 516 uA); 13 filaments at 5 kV,
tortuosity 1.46 median, render 1.45 ms per 4K frame, lab / circuit / publisher self-tests pass.

### Fourth pass — halo, one filament per finger, hands (2026-09-14)

| Report | Cause | Change |
|---|---|---|
| Blocky ("pixelised") halo, independent of glow and upscale | a segment is only visited by rays crossing its 1-cell dilated footprint (1.56 mm) while the 4× sheath reaches ~2 mm: the halo was clipped at cell boundaries | CSR dilation 2 cells (`ITEMS_MAX` 125/segment, mailbox 16) and the sheath fades to zero with `smoothstep` at `VIS_RADIUS` = 2 cells from the ray–segment distance (`raySegmentDist`); candidates/ray mean 5.5, trace 1.2 ms |
| Two filaments per finger | a re-routed tree and a strike arrived together; both counted as served | ownership in `k_fingers`: the attached filament with the largest contact coverage owns the finger and alone gets the touched termination; another attached filament with its foot within two half widths of an owned finger has its admittance ×0.01 (barrier saturated by the owner) and retracts; touched strikes only while nothing else grows |
| Closed fingers each got their own filament from the electrode | no multi-foot channels | brush leaders: an unserved finger within 0.8 rad of a touched filament's main foot asks that filament for a leader forked `BRUSH_FORK` = 2 cm before its foot; the leader becomes a *secondary foot* (`t_foot2`, up to 4) instead of replacing the channel, its chain is marked with the main channel (kept by the prune, rendered as main), dropped when no finger covers it for 0.5 s, on re-route and on retraction |
| Brush leaders kept landing next to the finger already served | the served finger's charge still pulled them, the drained inhibition let them land | a served finger's charge is screened by `SERVED_SCREEN` = 85 % (the barrier under it charges up once fed); the σ inhibition fades only under *unserved* fingers; the charged footprint is a plateau of radius `sig_radius` (8→15 mm), not an annulus; contact half width 8 mm (`FINGER_HWHM` 2.4 × standoff: a fingertip pad through the glass); ownership coverage ≥ 0.15 (~1 cm) |
| Touched filaments wandered | the Poisson re-route timer also ran for pinned filaments | `t_hold` from the app: no timer re-route while touched (stretch and requests still apply) |

Headless scenarios after the pass (finger directions on the glass, frame 600):

| Fingers | Result |
|---|---|
| one | one owner filament, 958 µA |
| two, 90° apart | one filament each (640 / 489 µA) |
| two, 25° apart | one filament each (511 / 571 µA) |
| three within 35° (a hand) | **one** filament (1028 µA) with a brush foot at each finger (4.4°, 5.1°, 1.6°) |

Self-tests (engine lab 4/4, circuit, publisher) pass; 13 filaments at 5 kV without fingers.

Brush branches render with their share of the channel: 1/(feet) of the power and a core scaled
by that share to the 0.4 (`NODE_BRUSH`, `tree_nfeet`); the core law is r = 0.45 mm (I / 50 uA)^0.4
clamped at 1.5 mm. 3600-frame run after the pass: 13 → 19 filaments, 0 NaN, sim 1.23 ms, 4K frame
3.1 ms median.

### Experiment — the old shader's look on the simulated filaments (2026-09-14)

`M` in the app (`--nimitz` headless) switches the tracer to a second shading of the same data:
nimitz's `plasma_globe.glsl` recipe with his 13 procedural rays replaced by the simulated
segments found through the CSR grid. Per 0.75 mm march step the nearest segment plays his ray:
tube radius = 2 × core / (ins·outs) (thin mid-gap, flaring at the electrode and at the glass),
soft interior (`smoothstep` over 1.1 mm of his halved distance), his position palette, noise
flicker, max over segments at a point, sum along the ray scaled to his 0.03-radius step, his
flow-noise reflections on the sphere, a pink luminous ball for the electrode, no tonemapping
(the present pass clamps). Cost 0.2 ms. It is display-referred and not physical; it exists to
compare the two looks live on the same simulation.

### Fifth pass — calibration against the reference footage (2026-09-14)

Measured on the user's screenshots (globe ~1500 px across, 10 px/mm): filament FWHM 1.1–1.9 mm
with a faint skirt out to ~8 mm (10 % level), peak colour ≈ (120, 110, 210) 8-bit (violet-blue,
not clipped), ~35 filaments, background (3, 5, 4), electrode ~1/7 of the globe diameter with a
dark centre (26, 8, 27), a lavender rim (173, 147, 255) and bright attachment points.

| Difference | Change |
|---|---|
| Electrode too large (1/5 of the globe) | R1 = 1.1 cm everywhere (params, engine, gas, shaders, publisher); the app configures the engine and the gas solver with it |
| Colour lavender-white vs violet-blue | `video` colour preset (default, `T` cycles to the spectral ones): shaft (0.20, 0.14, 0.78), feet/electrode pink (1.0, 0.30, 0.62); shaft x_ion 0.75–1.0, feet 0.45 |
| Milky, cores not saturated, halos too soft | camera tonemap (clip + sRGB OETF, knob `tonemap` 1; AgX at 0), exposure bias +0.2, glow 0.4, sheath 3 × core carrying 40 % (the skirt), ambient 0, background discharge 0 |
| Electrode a flat lit disc | bulb albedo 0.02, gloss ×0.2, shell 0.025 with a strong limb law, halo ×0.3 |
| Feet too big | 1.2 mm σ, brightness 0.08 × (I / 50 uA) |
| 13 filaments vs ~35 | strike field 110 kV/m (V_th ≈ 1.0 kV at this electrode; 130 → 21, 100 → 28 after 30 s), I_STRIKE 20 µA. Remaining gap is model over-screening (channels held at the electrode potential); the resistive channel φ_ch(s) is the physical follow-up |

Result: bright-pixel mean (147, 115, 195) vs the footage's (120, 110, 210); side-by-side in
`/tmp/plasma_compare.png`. The nimitz look mode (`M`) stays as a second reference.

### Sixth pass — drift speed and finger following (2026-09-14)

* Drift: (r/R2)^1.5 weighting and 0.7 W heating (peak gas 6 cm/s); the stretch re-route trigger
  moves from 1.5 to 1.8 (`STRETCH_TRIGGER`) so channels drift longer between re-strikes
  (≈ 12 re-routes/s for 24 filaments, the 2 s timer now dominates).
* Finger following: the circuit publishes, per foot (main and brush), the direction of the finger
  it serves (`tree_targets`); the advection kernel pulls the foot and the last 2 cm of the channel
  towards it with a 50 ms time constant (weight (depth/2 cm)²), so a dragged finger is followed
  instead of lost and re-struck. Headless `--finger-speed` rotates the fingers about +y:
  at 2.2 cm/s the nearest foot stays 1.6° from the finger on average (1.1° p90), at 7.5 cm/s
  3.6° (3.0° p90); the two >10° samples per 10 s are re-strikes of a channel wrapped around the
  globe by the drag (stretch trigger), which is expected.

### Seventh pass — resistive channel, sheath march, re-strikes (2026-09-14)

* **Resistive channel** (engine parameter `E_ch`, V/m; the globe sets V_CH / 6 cm = 33 kV/m, the
  lab keeps 0). A node at arc length s is held at φ = E_ch s / V instead of 0: the conductor
  right-hand side is φ_ch(s) − u_ext, candidate potentials are taken relative to their parent's
  channel potential, the Born charge targets φ_ch(s_new). The attached channels then screen the
  electrode far less: the electrode pool has ~130 of 247 candidates above the strike field
  instead of 0.6, strikes run at 4.5/s and the population settles at **31 filaments** at 5 kV
  (was 17), which is the footage's count.
* **Sheath march.** Summing per-span sheath integrals left a wedge gap on the outside of every
  joint (transverse ribs at the sheath's width). The core stays closed-form per span; the sheath
  is now marched (0.6 mm steps) over the distance to the nearest segment of each cell, i.e. the
  union of the capsules. Trace 3.4 ms at 30 filaments (whole 4K frame 3.9 ms).
* **Re-strikes.** 30 % of the timer events retract the filament instead of re-routing it
  (`RESTRIKE_P`); the count law strikes the replacement where the electrode is least screened,
  so roots stop collecting at the top of the bulb (13 % above y/r = 0.7 after 20 s, was 74 %);
  roots also sample the flow 2 h above the bulb instead of 4 h; admission allows 4 growing trees.

Final calibration numbers of the pass (frame 900, 29–32 filaments): bright-pixel mean
(164, 127, 223) vs the footage's (120, 110, 210); bulb centre (26, 18, 33) vs (26, 8, 27), bulb rim
(159, 130, 174) vs (173, 147, 255); exposure bias −0.4 EV, bulb shell 0.03.

### Eighth pass — mitre-clipped capsules (2026-09-14)

The remaining transverse ribs came from the core: two consecutive capsules meet at a bend with a
wedge gap outside and a double count inside every joint, and a 0.45 mm core cannot be marched
without ~0.1 mm steps. Exact fix instead: each segment record now carries the bisector planes of
its two joints (`n_start`, `n_end`; record = 7 vec4, `SEG_STRIDE`), and `capsuleEmission` clips
the body integral to the slab between them. On a bisector the distances to the two lines are
equal, so consecutive spans tile space with no gap and no overlap while the closed forms stay
exact; the sheath is back to closed form too (the march is gone). Trace 3.1–3.7 ms at 30
filaments; `MAX_CANDIDATES` 1024 (rays through the root cluster at the bulb exceeded 512).

### Ninth pass — exact segment dedupe, filament hue (2026-09-14)

* The last transverse steps were double counts: with the 2-cell dilation a segment sits in many
  consecutive cells along a ray and the 16-entry mailbox missed revisits once ~30 filaments
  shared a region, so whole spans were integrated twice, stepping at cell boundaries. Each
  segment is now integrated exactly once, in the cell that contains the ray's closest approach
  to it (always a listed cell); no mailbox. Candidates/ray 15 → 10, capped rays 0, trace
  3.7 → 2.1 ms at 30 filaments (whole 4K frame 2.7 ms).
* Mid-gap hue measured against the footage (peak R/B, G/B): footage 0.57–0.69 / 0.51–0.62, ours
  before 0.60 / 0.46 with pure deep blue in the dim parts. `video` ion colour (0.22, 0.17, 0.75),
  shaft x_ion 0.7–0.92 (10–30 % neutral pink), sheath tinted (1.30, 0.95, 0.85) relative to the
  core (the cooler outer layer radiates more neutral lines): now 0.69–0.70 / 0.52.

### Tenth pass — hue striping, the meter (2026-09-14)

The ninth-pass render showed filaments striped in blue and purple along their length although
the published segment colours are uniform along a shaft (checked: normalised rgb constant to the
last 8 mm, where x_ion blends to the pink foot). Sampling the rendered hue along the projected
main chains (peak pixel within 6 px of each node) showed R/B swinging 0.70 → 0.0 in sections,
correlated with brightness (+0.8) and present without TAAU and without the glow, so it came from
the tracer:

* **Negative light.** The foot renderer subtracted a grey "dark annulus" around every foot on the
  far wall. In front of a ring a dim filament pixel lost the same amount from R, G and B; the
  clamp at zero removed red and green first and left pure blue. Rings covered a fair part of the
  back wall (radius up to 1.5 cm, ~30 feet), so filaments crossed in and out of them: the
  striping. The term is gone (`footGlow` is non-negative). Per-tree hue spread along the shaft
  (p90 − p10 of R/B): 0.29 → 0.05; pure-blue pixels 24 801 → 0.
* **Sheath tint.** The ninth-pass pink sheath made the pixel hue depend on the local core/sheath
  ratio; the sheath now has the core's colour (one colour per segment, from x_ion alone).
* **Tone curve.** The per-channel clip of the camera tonemap turned bright violet pink; the largest
  channel is now rolled off with a tanh shoulder above 0.7 with all channels scaled alike, and
  only extreme highlights (> 2, the touched filament) desaturate towards white.
* **Meter.** Removing the negative term re-exposed the frame by ~1.5 EV: the log-average meter
  was reading the clamped black floor, not the filaments. Replaced by a highlight-priority meter
  (`exposure.comp`): 64-bin log₂ histogram, the luminance below which 99.7 % of the pixels lie is
  exposed to 0.18 (display-linear; the ordinary cores end just under clipping), same asymmetric
  adaptation; the bias knob is still in EV and defaults to 0. Room radiance ×0.35 and the bulb's
  reflected/lit terms ×0.3 to keep the footage's black background and dark ball at the new
  exposure; bulb limb law `0.004 + 0.5(1−μ) + 8(1−μ)⁴` (thin rim).

Frame 400, 30 filaments: shaft R/B 0.70 (p10 0.69, p90 0.72), bright-pixel mean (102, 75, 138)
vs the footage's (120, 110, 210), bulb centre (33, 24, 37) vs (26, 8, 27), bulb rim p95
(230, 189, 247) vs (173, 147, 255), background (2, 2, 3) vs (3, 5, 4); metered exposure 0.143.

### Eleventh pass — matching the recording (2026-09-14)

Reference: the user's screen recording of a real globe (`~/Videos/Screencasts/Screencast from
2026-09-14 17-23-29.webm`, 1683×1469, 29.33 fps, 101 s; frames extracted to `/tmp/plasma_vid/`).
Measured on it: the bright-filament mask (B > 150) keeps only 0.28–0.49 of its pixels one frame
later and, dilated by 1.6 mm, 0.67–0.75 / 0.47–0.55 / 0.41–0.47 / 0.29–0.35 after 34 / 68 /
100 / 200 ms — the filaments jump to nearby paths several times a second rather than sliding; the
overlays of consecutive frames show nearly straight radial spokes, wavier towards the glass.
Ordinary cores peak at (138–156, 116–135, 224–255); the touched channel is a 2 mm saturated white
core with a blue fringe, and 15–20 thin filaments stay lit around it; the bulb is a dark magenta
ball ((33,13,41) at the centre, (76,41,107) on the face, (118,71,157) at the rim) with a black stem
below it and pink-white flares where the filaments leave it; under a touch the whole ball glows
pink with a white blob at the touched root. Our sim before this pass kept 0.87 / 0.68 / 0.56 of
the dilated mask after 33 / 100 / 200 ms and carried long, bowed channels.

* **Dynamics.** Poisson re-route timer mean 2 s → 0.25 s (`REROUTE_MEAN`), fork at 10–60 % of
  the arc instead of 35–75 % (most of the channel regrows: a whole-path jump guided by the hot
  channel), `STRETCH_TRIGGER` 1.8 → 1.3. Re-routes 0.7 → 3.5 per filament per second; dilated
  overlap 0.77 / 0.61 / 0.53 / 0.24 at 33 / 67 / 100 / 200 ms (the recording's 0.67–0.75 / 0.47–
  0.55 / 0.41–0.47 / 0.29–0.35); tortuosity p90 1.34 → 1.19, turning in the proximal third 56 →
  34°/cm; the downward filaments no longer collect at the bottom (they no longer live long enough
  to be swept there by the return flow). A touched filament now re-routes too (timer at ¼ speed)
  and never re-strikes, so it keeps changing shape but never vanishes.
* **Circuit.** `R_CH` 3.3 → 25 MΩ/m (differential): with the 215× touched termination a touched
  filament takes ~15× an ordinary one (share 33–44 %) instead of 60–80 %, so the others stay lit
  (29 attached under a touch, was 6); `I_SUPPLY` 1.5 mA. Self-test thresholds updated (share ≥
  30 %, others dimmed but above sustain).
* **Look.** Radiance ∝ I^1.5 (`publish.k_segments`) and the camera curve goes white by ~4× the
  knee, so the touched core saturates (white width 19–21 px at 4K ≈ 2 mm, blue fringe) while an
  ordinary one is a violet line; core radius ∝ I^0.3; shaft x_ion 0.85–0.97 (bluer). Root flares
  on the first millimetres of each channel (×3.5 at the bulb, 2.5 mm e-fold). Bulb: magenta
  `ELECTRODE_RGB` (0.55, 0.17, 1.0), limb law ×0.012·I_tot, sheath gain Σ(I_k/50 µA)^2.5 / 20 (the
  ball turns pink all over under a touch), root records (32 per frame after the foot records in
  the sigma SSBO) drawn as ~I^3 spots and halos (only a touched root shows), a 4 mm black stem
  below the bulb (occludes both interior paths), soft filament reflections. Touched foot glow
  ~6 mm wide. Meter: 98th percentile → 0.06 (a touched filament, ~1 % of the frame, only pushes
  L_p up the ordinary population; ordinary cores then peak at B ≈ 215–250 with R/B 0.64 and the
  touched one clips white, as the phone camera does). Room ×0.35, table albedo 0.12.

Frame 400 (30 filaments) vs the recording: ordinary peaks (141–148, 114–123, 214–233) vs (138–156,
116–135, 224–255); bulb face (61,41,80) vs (76,41,107), rim (138,97,188) vs (118,71,157); touched
bulb face (158,110,201); touched core white 19 px. Solo timing (headless, 4K, 30 filaments): trace
1.74 ms, render total 2.32 ms, sim graph 1.76 ms (was 1.23: the faster lifecycle), whole frame
median 4.13 ms, p99 5.72 ms. Not yet matched: the recording's hand is a whole palm (a 3–5 cm pink
patch on the glass) where ours is one finger point, its bulb is translucent with the support rod
visible inside, and its filaments are brighter and bluer in the first third than near the glass.

### Twelfth pass — roots, room lighting, second recording, re-strike rate (2026-09-14)

* **Roots.** The channels hovered a gap above the bulb: the root node sits at R1 + h and the first
  span started there. The first span now starts on the bulb surface (`publish.k_segments`, parent
  = root), the core widens ×1.6 into the bulb over 3 mm (`ROOT_WIDEN`) on top of the ×3.5 flare, and
  the glow layer around the bulb was recoloured from the recording's radial profile (medians per
  mm from the surface: rim (113,52,147) → (118,63,180) at the surface → (85,45,167) at 1 mm →
  (77,52,172) at 3 mm → (48,29,124) at 8 mm): the electrode sheath is violet-magenta (0.45, 0.10,
  1.0) ×0.6, the (R1/r)⁴ halo deep violet (0.20, 0.08, 1.0) ×0.15, `ELECTRODE_RGB` (0.55, 0.12, 1.0).
  Ours after: (152,90,213) / (109,81,169) / (95,72,150) / (77,60,122) / (56,44,92) — the levels match,
  the halo between filaments is still greyer than the recording's (G/B 0.49 vs 0.30). Wikipedia's
  plasma-globe article (fetched; the PPPL report is not in `papers/`) adds only that each tendril
  competes for a footprint on the inner electrode with a thin dark boundary around it.
* **Room.** Three rectangular panel lights (key, fill, rim: centre / half-axes / radiance constants
  in `trace.comp`) replace the soft window: mirrored in the glass (branch A), diffuse on a dark
  glossy table (`tableShade`: small-source form factor with the globe's base and bulb as shadow
  casters and the glass at 55 % transmission; Fresnel-weighted gloss = blurred panels, the globe's
  dark body, and the filaments as glossy line-light reflections), a 5 cm black base under the
  globe (`baseHit`/`baseShade`). Trace 1.74 → 2.55 ms at the default view (the table shading).
* **Second recording** (`Screencast from 2026-09-14 21-07-51.webm`, 1328×1201, variable rate,
  55 s; the same globe from further back). Dilated (1.5 mm) overlap 0.56–0.68 / 0.45–0.48 /
  0.30–0.37 at ~67 / 100 / 200 ms, consistent with the first recording and with the sim's
  0.61 / 0.53 / 0.24. Its colours are pinker and dimmer (ordinary peaks R/B ≈ 0.8 vs 0.6): a
  different white balance/exposure of the same globe — the first, closer recording stays the
  colour reference.
* **Re-striking every frame.** Tried by shortening the Poisson timer: mean 0.25 s → 30 attached,
  3.5 re-routes per filament per second; 0.1 s → 16 attached; 0.05 s → 6 attached. The total
  re-route throughput saturates at ~90 per second whatever the timer, because a regrowth takes 2–3
  frames at S_MAX = 16 steps (2.4 cm per frame) and a regrowing tree carries no current, so the
  population collapses. Per-frame re-striking needs same-frame regrowth: either S_MAX ≈ 48 (a 6 cm
  channel in one frame; growth is ~half of the 1.8 ms sim graph, so ~+2 ms), or a dedicated
  "regrow in place" kernel that rebuilds a channel along its hot-channel memory in one launch and
  swaps it, keeping node ids for the motion vectors. Not done in this pass.

Per-pass render cost at the default view: trace 2.55 ms, TAAU 0.25 ms, total 3.11 ms (solo run).

### Thirteenth pass — root funnels, zoom-invariant look, soft shadows (2026-09-14)

Reference for the bulb: 6× zooms of the recording and of the screenshots (`/tmp/plasma_vid/zoom_*`).
The sharpest screenshot (10-43-23, bulb radius 172 px) shows a thin saturated pink rim line, a
dim purple haze between the roots (dimmest quartile (65,30,61) sRGB at 0–4 mm, (26,17,40) at
10 mm; B p25 68–80 within 4 mm, 53 at 10 mm), and every filament leaving the rim as a soft funnel
(half-max footprint 0.7–0.8 mm per crossing at 0.5–1 mm from the bulb vs 0.42 mm at 2–3 mm) whose
skirt is several mm wide; 10-45-48 shows the funnels saturating within a bulb radius while the
haze stays dim, a luminous face (125,82,168) and root flares covering ~40 % of the disc.

* **Funnels.** Each segment record now carries a root factor `rootF = exp(−(r − R1)/4 mm)` and
  its own CSR dilation (+2 cells within ~3.6 mm of the bulb) in the spare `.w` of the two mitre
  normals; the tracer widens the sheath ×(1 + 2.5·rootF) and brightens it ×(1 + 3·rootF), the
  core widens ×2.5 into the bulb (`ROOT_WIDEN`), the first span starts on the surface, and the
  root colour blends to the neutral pink lines (`X_ION_ROOT` 0.35 over 5 mm: the flares are
  pink-white in the footage). Bulb limb law `0.40 + 0.25(1−μ) + 10(1−μ)⁸` (luminous face, thin edge
  line), `ELECTRODE_RGB` (0.52, 0.20, 1.0), halo ×0.05 blended magenta→violet with distance. The
  dimmest-quartile level near the bulb is insensitive to the halo, the funnel width and even the
  glow (one-term-off renders: 0.67 → 0.62 without half the sheath, → 0.56 without the glow): within
  4 mm the ring is fully covered by ~30 filaments' sheaths, so it measures the sheath, not a haze.
* **Zoom.** Two things were defined in screen pixels: the meter (top 2 % of the frame) and the
  glow kernel (24 internal px), so zooming out re-exposed the picture and kept a touched
  filament's bloom the same width. The meter now takes the luminance exceeded by 9.4 % of the
  globe's *projected* pixels (`refArea` from the camera), and the glow kernel and gain scale with
  the projected globe radius (weights interpolated in a table fitted once for 24–192 px).
  Measured: touched core white width 16.6 px at the default distance, 7.3 px at twice the
  distance (0.44; proportional would be 0.5); bulb face (98,59,133) vs (99,60,135), rim 218 vs 210.
* **Soft shadows.** `rectIrradiance` takes 4 jittered points per panel per frame (pcg4d seeded
  by pixel and frame; TAAU integrates them), each tested against the glass sphere (55 %), the bulb
  and the base. Trace 2.55 → 2.85 ms at the default view.
* **Research (agent on another model; web search is blocked here, the agent used OSTI, Crossref
  and direct PDFs).** PPPL-4485 is Campanell, Laird, Provost, Vasquez, Zweben, *Measurements of the
  Motion of Plasma Filaments in a Plasma Ball* (2010), https://www.osti.gov/servlets/purl/973080:
  Ne+Xe 740 Torr, 26 kHz, 5 kV, ~1 mA, 6 cm × 0.5–1 mm filaments that start at the electrode
  surface and emerge perpendicular to it; at 400 ns exposure the instantaneous intensity
  *increases* with radius — the bright thick root is a time-integration effect, which is what a
  camera and this renderer show; the discharge starts as a diffuse glow around the electrode out
  of which the filaments grow; dying filaments retract into the electrode in ~0.5 ms; no footprint
  size or halo ratio is given. Brandenburg 2017 (PSST 26 053001): anode glow + 30 µm cathode
  layer per microdischarge, channels broaden on the dielectric. nimitz's Shadertoy XsjXRm: tendril
  half-width ∝ 1/ramp near the electrode with brightness ∝ ramp, i.e. the flare *is* the halo, no
  separate additive term — the same construction as the funnels above. Kim & Lin 2004: the
  apparent thickness is HDR bleaching of the APSF glow.

### Fourteenth pass — trumpet roots, root-lit anode, feet that re-strike upward (2026-09-15)

* **Roots.** The root factor is now a bell, `(s0 / (s + s0))²` with s0 = 1.5 mm (1 at the bulb, ¼
  at 1.5 mm, 1/16 at 4.5 mm): the sheath flares ×5 and the core ×3 at the mouth and the flare
  closes within a few millimetres, like the end of a trumpet (`publish.k_segments`, tracer
  `FUNNEL_WIDEN` 4). The anode is lit by its roots: each root casts a soft pool of glow on the
  sphere (σ 1.8 mm, ∝ I, `POOL_GAIN` 0.008, `rootGlow`) over a dim uniform face (limb law
  `0.05 + 0.25(1−μ) + 10(1−μ)⁸`), so the sphere is bright where filaments attach and dark between.
  Face mean 0.53 of the filament-core level (screenshot 0.49), face p10/p50/p90 0.32/0.49/0.82
  (was 0.70/0.83/1.05 uniform with wider, brighter pools).
* **Feet.** Measured over 4 s at frame 500: roots drift up at +1.6 mm/s (1 % down) but the feet
  on the glass drifted down at −1.8 mm/s (100 % down) — the return flow along the cold wall,
  while the channel's middle rose: the arch the user saw. In the recordings the whole channel
  rises, breaks and re-strikes with its foot higher. The channel no longer rides the gas within
  8 mm of the glass (`FOOT_PIN_LEN`, smooth ramp): the foot is held by its surface-charge
  footprint, the rising middle stretches the channel, the re-route regrows it along the risen hot
  channel. After: foot drift −0.02 mm/s, and the feet's re-strike jumps (254 in 4 s) are upward
  57 % of the time with a median of +4.8 mm. Population 28, re-routes 2.9 per filament per
  second (was 3.5), tortuosity p90 1.17.

### Fifteenth pass — root fans, channels never sink (2026-09-15)

* **How the root was drawn until now** ("artistic"): one constant radius per 1.5 mm segment
  scaled by a bell factor, so the flare was a fatter first cylinder, not a fillet. **Now**: a
  time-integrated model of the attachment. Within a frame (~900 half-cycles) the streamer's
  attachment point wanders over the root's footprint on the electrode (Wikipedia: each tendril
  has a footprint on the orb; PPPL-4485: roots slide on the conductive paint), so what a camera
  integrates is a bundle of sub-channels fanning out of the channel onto the surface. `k_root_fans`
  writes, per attached tree, `FAN_K` = 6 sub-channels from points spread over a footprint of radius
  2 mm·(I/40 µA)^0.3 (fixed per root node: they change only when the channel re-strikes), each
  leaving the surface perpendicularly for 0.8 mm (PPPL: filaments emerge normal to the bulb) and
  merging into the channel 1 mm beyond the root node; two segments each, wide soft tubes (1.5× the
  core radius, so they overlap into one bell), root colour, 1.5 ordinary segments' worth of light
  shared between them; `SEG_MAX` = N_MAX + 384 fan slots, CSR dilation +2. The per-segment bell
  is kept small (core ×1.8, sheath ×3 at the mouth). Root pools on the anode `POOL_GAIN` 0.005.
  Face 0.68 of the core level, p10/p90 0.42/1.02. Trace 2.85 → 3.5 ms (the fan slots). What a
  fully physical model would need (the sheath and glow layer of the attachment at µm scales) is
  out of reach in real time; the fan is the integrated appearance of the physics.
* **Feet.** Pinning the foot node was not enough: measured per band of distance to the glass, the
  channel 5–20 mm from the glass still sank at 6–10 mm/s (the convection cell's return flow along
  the cold wall on the 2.5 mm gas grid) while the middle rose at 8 mm/s, so the end drooped into
  a hook. A real channel sits inside its own buoyant plume (ΔT ~100 K over ~1 mm: cm/s relative to
  the ambient) that the grid cannot resolve, so it never sinks: the vertical component of a
  channel node's drift is now at least `PLUME_MIN_RISE` (1 mm/s, before the drift weight;
  gravity-sign aware), and the foot pin is 4 mm. After: no band sinks (0–2 mm +0.05, 2–20 mm
  +0.8 to +0.9, 20–50 mm +4 to +8 mm/s), the channel stretches and re-strikes with the foot higher
  (as measured in the previous pass). Population 27, re-routes 3.0 per filament per second.

### Sixteenth pass — continuous root bells (2026-09-15)

The six-sub-channel fan read as dots. Replaced by one continuous fillet per root: `k_root_bells`
writes a single record per attached tree (slot N_MAX + t·12, flag `NODE_BELL` = 1024) — a segment
from the electrode surface to 1.2 mm beyond the root node — and the tracer renders that record
with `bellEmission` instead of a capsule: a surface of revolution about the axis with radius
ρ(s) = r_c + (R_b − r_c)(1 − √(1 − (1 − s/L)²)), a quarter ellipse tangent to the sphere at the
base (ρ = R_b, vertical tangent) and to the channel at the throat (ρ = r_c, zero slope), density
(1 − (d/ρ)²)²/ρ² so the light per unit length is constant (nimitz's width × 1/ramp, brightness ×
ramp), marched with 12 midpoint samples over the ray's overlap with the bounding cylinder (only
rays within R_b of a root). R_b = 3 mm at the reference core radius, ∝ r_c ∝ I^0.3. `BELL_GAIN`
was calibrated by rendering (1.8e-6: the throat matches the channel; 1e-5 saturates the bulb
region) because the analytic estimate against the capsule normalisation (≈1/ε⁴ on the axis)
was off by ~10⁶. Root pools lowered to 0.002 (the bells light the base). Trace 3.5 → 2.7 ms (the
fan's 12 records per tree are down to 1).

An adversarial review (3 lenses, each finding verified by an independent refuter; 7 of 11
findings confirmed) then found the 10⁶: the integrand (1 − (d/ρ)²)²/ρ² already integrates to π/3
per unit length, so the extra `/ rc²` made the bell's brightness scale as 1/rc² across channels
(thin channels ~5× too bright relative to their channel, the touched one ~6× too dim) and the
"calibrated" gain silently carried m². Fixed: `BELL_GAIN` is dimensionless (4.0: light per unit
length = 4.0·π/3 × the record's power at the base, fading to zero over the last 30 % of the
height so the channel's own capsule takes over at the throat, no bead). The review also showed the
mouth was a flat disc perpendicular to a possibly tilted axis (up to ~35° off the normal, since
the root's first link is one of the cone directions), half buried in the ball and half floating:
the fillet radius is now a function of the height above the sphere, |x| − R1, so the mouth hugs
the ball whatever the tilt; the mouth radius is clamped to the record's 4-cell CSR reach. Face
0.61 of the core level, p10/p90 0.36/1.02.

### Seventeenth pass — the rolling motion, measured against a third recording (2026-09-15)

Reference: `Screencast from 2026-09-15 10-42-29.webm` (1162×1045, 10 s, 11 distinct frames per
second). A ring tracker (`/tmp/plasma_dbg/track_ring.py`: intensity vs angle on a ring, peaks,
nearest-neighbour tracks; vertical velocity in mm/s from the frame spacing) was run on rings
around the anode (4.3 and 7.6 mm out) and 5 mm inside the glass, on the recording and on 5 s of
the simulation rendered at 30 fps, with the same code:

| ring | recording | sim before | sim after |
|---|---|---|---|
| anode +4.3 mm: median v_y, fraction up | +1.65 mm/s, 0.69 | +0.88, 0.70 | +1.05, 0.77 |
| anode +7.6 mm | +2.2 mm/s, 0.72 | +1.47, 0.78 | +1.5 to +1.95, 0.79 |
| anode track lifetime | 0.45–0.55 s | 0.33–0.37 s | 0.40–0.43 s |
| glass: median v_y | +0.57 mm/s, 0.57 up | 0.00 (pinned) | creep +0.1 (array), jumps 59 % up, +5.5 mm |

So the recording's "rolling" is a coherent upward walk of the roots over the anode at ~2 mm/s
(70 % of crossings up, births and deaths balanced top/bottom: a walk, not a conveyor) with the
feet creeping slowly and jumping up on re-strike. The sim had the walk at ~60 % of the speed and
the feet pinned. Changes: roots ride the plume over the bulb with `ROOT_DRIFT_GAIN` 1.8 and the
same plume floor as the channel; the foot pin becomes a creep (`FOOT_CREEP` 0.5 of the drift
inside 4 mm of the glass); `STRETCH_TRIGGER` 1.3 → 1.5 because the faster walk stretched the
channels sooner (population had slipped 28 → 23; back to 28, re-routes 2.5 per filament per
second). Bells: base radius 3 → 4.5 mm, gain 4 → 2 (wider, dimmer mouths).

### Eighteenth pass — feet that walk, key logging, wider bells (2026-09-15)

* **Feet.** The creep was invisible (+0.1 mm/s in the arrays, 0 in the image tracker) and an
  attempt to make the foot follow the extrapolated tilt of the channel's last two centimetres
  walked the feet *down* (−1.1 mm/s, 69 % down: with the middle risen and the end lagging, the
  last centimetres point downward, so their extrapolation meets the glass below the foot).
  Dropped. What the recording shows is the whole channel rising, foot included (feet: median
  +0.6, upper quartile +5.9 mm/s; roots ~+2 mm/s), i.e. the channel rides its own plume all the
  way to the wall. `PLUME_MIN_RISE` 1 → 4 mm/s (the floor applies to every node, roots
  included) and the foot pin is gone (`FOOT_CREEP` 1). Measured over 4 s: feet +2.5 mm/s while
  attached (92 % up, was 0), re-strike jumps 59 % up with a median of +7.6 mm, roots +2.9 mm/s;
  population 26, re-routes 2.3 and retracts 1.4 per filament per second (the channels break more
  often because the ends now walk).
* **Keys.** Every key that changes a setting prints `[keys] <setting> <new value>` (`log_key` in
  `plasma_globe.py`): pause/run, reset, gravity, ice, voltage, frequency, η, γ, finger charge,
  gas preset, quincunx, hybrid, lights, glow on/off and width, exposure bias, temporal upscale,
  wireframe, look, debug view, state dump.
* **Look.** Bell base radius 4.5 → 6 mm (the record's CSR dilation +3 cells, mouth clamped to 7 mm);
  default glow width 24 → 32 internal px.
