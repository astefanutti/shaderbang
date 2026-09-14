#!/usr/bin/env python

# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Headless growth lab for the plasma globe (plan milestone M2): no DRM/KMS, no OpenGL.

Modes (each prints its measurements and PASS/FAIL against the M2 thresholds)

  grow       single tree (or --trees F concurrent pools) in the fixed field: fractal dimension
             (box counting + mass-radius), reach at 350 steps, distinct tips, shell contacts
  pools      F = 13 concurrent pools vs F = 1 sequential (same total nodes): |dD|, min-distance
             histograms (KS statistic), and the screening halo around a frozen filament vs the
             grid reference
  reference  grid Laplace reference (examples/plasma/reference.py) vs the engine's candidate
             potentials at a snapshot: Spearman + absolute RMSE for a/h in {0.15, 0.25, 0.35};
             Born-vs-resolved charge error; warm-CG residual
  strike     strike gate: seeding vs V at T0 and with a hot channel (T/T0 = 1.1)
  touch      one finger on the glass, q_f sweep: shell contacts within 0.5 rad, maxr
  bench      Script B: microseconds per growth step and per frame, captured graphs replayed from
             a restored snapshot at n in {1k, 4k, 5k, 20k} nodes x C_MAX in {8k, 32k, 65k}
  replay     capture once, replay 600 frames twice, assert bit-identical; captured reset round trip
  nan-scan   10k frames with re-routes enabled, NaN/Inf, count and topology invariants
  check      bookkeeping invariants: root ring / K-1 candidates, exact incremental potentials with
             idle trees (n_cg 0, small n_max), re-route topology and timer arming

Examples
  .venv/bin/python examples/plasma_globe_headless.py grow --seeds 1,2,3 --eta 3
  .venv/bin/python examples/plasma_globe_headless.py grow --r1-over-r2 0.12 --eta 3
  .venv/bin/python examples/plasma_globe_headless.py grow --device cpu --nodes 150 --seeds 1
  .venv/bin/python examples/plasma_globe_headless.py reference --n-ref 128
  .venv/bin/python examples/plasma_globe_headless.py check
  .venv/bin/python examples/plasma_globe_headless.py bench
"""

import argparse
import math
import os
import statistics
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plasma import dbm, harness, reference  # noqa: E402
from plasma.harness import verdict  # noqa: E402

wp.config.log_level = wp.LOG_WARNING
wp.init()


# --- common helpers ---------------------------------------------------------------------------

def make(args, **overrides):
    r2 = dbm.R2I
    r1 = args.r1_over_r2 * r2
    kw = dict(device=args.device, r1=r1, r2=r2, a_over_h=args.a_over_h, n_cg=args.n_cg,
              images=args.images, seed=args.seed, mid_cg=tuple(int(v) for v in args.mid_cg.split(",")),
              s_max=args.s_max)
    kw.update(overrides)
    g = dbm.Dbm(**kw)
    g.configure(eta=args.eta, gamma=args.gamma, V=args.voltage, global_norm=args.global_norm,
                grow_after_attach=args.grow_after_attach, born_gain=args.born_gain)
    return g


def tree_metrics(g, pos, flags, r2):
    r = np.linalg.norm(pos, axis=1)
    roots = pos[(flags & dbm.ROOT) != 0]
    root = roots[0] if len(roots) else np.zeros(3)
    out = dict(n=len(pos), maxr=float(r.max() / r2) if len(r) else 0.0)
    enough = len(pos) > 20
    out["Db"] = harness.box_counting_dimension(pos, dbm.H)[0] if enough else float("nan")
    # the dossier's acceptance estimator: n(R) about the globe centre over R = 0.2-0.6 (R2 = 1,
    # R1 = 0.12), i.e. 9-55 % of the gap; mapped to this geometry's gap so it does not start on
    # the electrode when R1/R2 = 0.2
    r1 = g.param("r1")
    lo, hi = r1 + 0.0909 * (r2 - r1), r1 + 0.5455 * (r2 - r1)
    out["Dm"] = harness.mass_radius_dimension(pos, (0.0, 0.0, 0.0), lo, hi)[0] if enough else float("nan")
    out["Dm_root"] = harness.mass_radius_dimension(pos, root, 0.15 * r2, 0.6 * r2)[0] if enough else float("nan")
    out["tips"] = harness.distinct_tips(pos, 0.8 * r2)
    out["contacts"] = harness.shell_contacts(pos, r2, g.param("a"))
    return out


def grow_until(g, nodes, max_frames=2000, r2=dbm.R2I, on_frame=None):
    """Runs frames until `nodes` live nodes; tracks maxr at 350 nodes and nodes at first
    r >= 0.94 R2 and at first attachment."""
    track = dict(maxr350=float("nan"), n94=None, n_attach=None)
    for _ in range(max_frames):
        g.run(1)
        n = g.node_count()
        if math.isnan(track["maxr350"]) and n >= 350 or (track["n94"] is None and n >= 40):
            nd = g.nodes()
            maxr = np.linalg.norm(nd["pos"], axis=1).max() / r2 if len(nd["ids"]) else 0.0
            if math.isnan(track["maxr350"]) and n >= 350:
                track["maxr350"] = maxr
            if track["n94"] is None and maxr >= 0.94:
                track["n94"] = n
        if track["n_attach"] is None and (g.t_state.numpy() == dbm.ATTACHED).any():
            track["n_attach"] = n
        if on_frame is not None:
            on_frame(g)
        if n >= nodes:
            break
    return track


def print_table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


# --- grow -------------------------------------------------------------------------------------

def mode_grow(args):
    r2 = dbm.R2I
    seeds = [int(s) for s in args.seeds.split(",")]
    f_max = max(args.trees, 1)
    c_f = 16384 if f_max == 1 else 2048
    g = make(args, f_max=f_max, c_f=c_f)
    if args.device != "cpu":
        g.capture()
    rows, Db, Dm, Dm_root, maxr350, tips, n94 = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for seed in seeds:
        g.reset()
        g.configure(seed=seed)
        g.set_admit(1)
        seeded = [0]

        def admit_control(g, seeded=seeded):
            if f_max > 1 and seeded[0] == 0 and int((g.t_state.numpy() != dbm.FREE).sum()) >= f_max:
                g.set_admit(0)
                seeded[0] = 1
            elif f_max == 1 and seeded[0] == 0 and g.node_count() > 0:
                g.set_admit(0)
                seeded[0] = 1

        track = grow_until(g, args.nodes, max_frames=args.steps, r2=r2, on_frame=admit_control)
        nd = g.nodes()
        m = tree_metrics(g, nd["pos"], nd["flags"], r2)
        rows.append([seed, m["n"], g.frame(), f"{track['maxr350']:.3f}", track["n94"], track["n_attach"],
                     f"{m['maxr']:.3f}", f"{m['Db']:.2f}", f"{m['Dm']:.2f}", f"{m['Dm_root']:.2f}", m["tips"],
                     m["contacts"]])
        Db.append(m["Db"]); Dm.append(m["Dm"]); Dm_root.append(m["Dm_root"]); maxr350.append(track["maxr350"])
        tips.append(m["tips"])
        n94.append(track["n94"] if track["n94"] is not None else 10 ** 9)
        if args.save:
            np.savez(args.save.replace(".npz", f"_seed{seed}.npz"), **nd)
    print(f"[grow] device={args.device} trees={f_max} R1/R2={args.r1_over_r2} a/h={args.a_over_h} "
          f"eta={args.eta} gamma={args.gamma} V={args.voltage} global_norm={args.global_norm} "
          f"images={args.images} ({time.perf_counter() - t0:.1f}s)")
    print_table(rows, ["seed", "nodes", "frames", "maxr@350", "n@0.94", "n@attach", "maxr", "D_box",
                       "D_mass(centre, 9-55% gap)", "D_mass(root 0.15-0.6)", "tips", "contacts"])
    Db_m, Dm_m, Dr_m = float(np.nanmean(Db)), float(np.nanmean(Dm)), float(np.nanmean(Dm_root))
    print(f"  mean D_box {Db_m:.2f}  mean D_mass(centre) {Dm_m:.2f} |dD| {abs(Db_m - Dm_m):.2f}  "
          f"mean D_mass(root) {Dr_m:.2f} |dD| {abs(Db_m - Dr_m):.2f}  nodes to 0.94 R2: {n94}")
    print(f"  D_box in [1.6, 1.8]: {verdict(1.6 <= Db_m <= 1.8)}   box/mass(centre) agree within 0.1: "
          f"{verdict(abs(Db_m - Dm_m) < 0.1)}   maxr > 0.94 R2 at 350 steps (all seeds): "
          f"{verdict(all(m > 0.94 for m in maxr350))}   >= 5 tips (all seeds): "
          f"{verdict(all(t >= 5 for t in tips))}")
    return Db_m


# --- pools ------------------------------------------------------------------------------------

def mode_pools(args):
    r2 = dbm.R2I
    F = 13
    total = args.nodes
    per_tree = total // F
    # concurrent: 13 pools growing together
    g = make(args, f_max=F, c_f=2048)
    g.capture()
    g.reset(); g.configure(seed=args.seed); g.set_admit(1)
    seeded = [0]

    def admit_control(g):
        if seeded[0] == 0 and int((g.t_state.numpy() != dbm.FREE).sum()) >= F:
            g.set_admit(0)
            seeded[0] = 1

    grow_until(g, total, max_frames=400, r2=r2, on_frame=admit_control)
    nd_c = g.nodes()
    # sequential: one tree at a time, earlier trees frozen (charged, not growing)
    g.reset(); g.configure(seed=args.seed + 100)
    for t in range(F):
        # seed at a raised voltage so the frozen trees' screening does not stall the protocol
        # (growth itself runs at the nominal voltage)
        g.set_admit(1)
        g.configure(V=2.0 * args.voltage)
        waited = 0
        while (g.t_state.numpy() == dbm.FREE).sum() > F - 1 - t and waited < 200:
            g.run(1)
            waited += 1
        g.configure(V=args.voltage)
        g.set_admit(0)
        if waited >= 200:
            print(f"  sequential: tree {t} did not seed within 200 frames (screened electrode), stopping at {t} trees")
            break
        target = int(per_tree * (t + 1))
        last, stall = -1, 0
        for _ in range(args.steps):
            g.run(1)
            n = g.node_count()
            stall = stall + 1 if n == last else 0
            last = n
            if n >= target or stall >= 30:
                break
        g.set_tree_state(t, dbm.FROZEN)
    nd_s = g.nodes()
    D_c = harness.box_counting_dimension(nd_c["pos"], dbm.H)[0]
    D_s = harness.box_counting_dimension(nd_s["pos"], dbm.H)[0]
    Dm_c = harness.mass_radius_dimension(nd_c["pos"], (0, 0, 0), 0.3 * r2, 0.9 * r2)[0]
    Dm_s = harness.mass_radius_dimension(nd_s["pos"], (0, 0, 0), 0.3 * r2, 0.9 * r2)[0]
    md_c = harness.min_distance_between_trees(nd_c["pos"], nd_c["tree"])
    md_s = harness.min_distance_between_trees(nd_s["pos"], nd_s["tree"])
    md_c, md_s = md_c[np.isfinite(md_c)], md_s[np.isfinite(md_s)]
    ks, p = harness.ks_statistic(md_c, md_s)
    bins = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]) * 1e-3
    hc, _ = np.histogram(md_c, bins)
    hs, _ = np.histogram(md_s, bins)
    print(f"[pools] F=13 concurrent: nodes {len(nd_c['ids'])} D_box {D_c:.2f} D_mass(origin) {Dm_c:.2f} "
          f"tips {harness.distinct_tips(nd_c['pos'], 0.8 * r2)} contacts {harness.shell_contacts(nd_c['pos'], r2, g.param('a'))}")
    print(f"[pools] F=1 sequential:  nodes {len(nd_s['ids'])} D_box {D_s:.2f} D_mass(origin) {Dm_s:.2f} "
          f"tips {harness.distinct_tips(nd_s['pos'], 0.8 * r2)} contacts {harness.shell_contacts(nd_s['pos'], r2, g.param('a'))}")
    print(f"  |dD_box| {abs(D_c - D_s):.3f} -> {verdict(abs(D_c - D_s) < 0.1)}   min-distance KS D {ks:.3f} p {p:.3f} "
          f"-> {verdict(p > 0.05)}")
    print("  min distance to another tree, mm bins " + " ".join(f"{b * 1e3:g}" for b in bins))
    print("    concurrent  " + " ".join(f"{v / max(hc.sum(), 1):.3f}" for v in hc)
          + f"   median {np.median(md_c) * 1e3:.1f} mm")
    print("    sequential  " + " ".join(f"{v / max(hs.sum(), 1):.3f}" for v in hs)
          + f"   median {np.median(md_s) * 1e3:.1f} mm")
    # per-pool vs global min-max normalisation A/B (F = 13 concurrent)
    for gn in (0, 1):
        g.reset(); g.configure(seed=args.seed, global_norm=gn); g.set_admit(1)
        seeded[0] = 0
        grow_until(g, total, max_frames=args.steps, r2=r2, on_frame=admit_control)
        nd = g.nodes()
        print(f"  normalisation {'global' if gn else 'per-pool'}: D_box {harness.box_counting_dimension(nd['pos'], dbm.H)[0]:.2f} "
              f"maxr {np.linalg.norm(nd['pos'], axis=1).max() / r2:.2f} tips {harness.distinct_tips(nd['pos'], 0.8 * r2)} "
              f"contacts {harness.shell_contacts(nd['pos'], r2, g.param('a'))} trees {int((g.t_state.numpy() != dbm.FREE).sum())}")
    g.configure(global_norm=args.global_norm)
    mode_halo(args)


def mode_halo(args):
    """Screening halo around a frozen radial filament (R1 -> 0.75 R2 along +z) at r = 0.6 R2."""
    r2 = dbm.R2I
    r1 = args.r1_over_r2 * r2
    h = dbm.H
    n_nodes = int((0.75 * r2 - (r1 + h)) / h) + 1
    z = r1 + h + h * np.arange(n_nodes)
    fil = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    g = make(args, f_max=1, c_f=1024)
    g.reset()
    # inject the frozen filament as a tree
    n = len(fil)
    ids = np.arange(n)
    wp.copy(g.pos, wp.array(fil.astype(np.float32), dtype=wp.vec3, device="cpu"), count=n)
    wp.copy(g.flags, wp.array(np.full(n, dbm.ALIVE | dbm.CHARGED, np.int32), device="cpu"), count=n)
    wp.copy(g.tree, wp.array(np.zeros(n, np.int32), device="cpu"), count=n)
    wp.copy(g.cnt, wp.array([0, g.n_max - n, n], dtype=wp.int32, device="cpu"))
    g.set_tree_state(0, dbm.FROZEN)
    g.solve(200)
    angles = [3, 6, 10, 15, 22, 30, 45, 60, 90, 135, 180]
    prof_e, raw_e = harness.halo_profile(g.potential, 0.6 * r2, angles)
    ref = reference.Reference(96, r1, r2, device=args.device if args.device != "cpu" else "cpu")
    ref.setup(nodes=fil)
    ref.solve(1e-8)
    prof_r, raw_r = harness.halo_profile(ref.sample, 0.6 * r2, angles)
    dossier = {6: 0.280, 10: 0.528, 15: 0.689, 30: 0.855}
    print(f"[halo] frozen filament {n} nodes to 0.75 R2, probe r = 0.6 R2, min-max normalised over "
          f"theta in {angles} deg (CG residual {g.cg_residual()[0]:.1e}, ref N=96 {ref.iterations} it)")
    print("   theta      " + " ".join(f"{a:6d}" for a in angles))
    print("   engine     " + " ".join(f"{v:6.3f}" for v in prof_e))
    print("   reference  " + " ".join(f"{v:6.3f}" for v in prof_r))
    worst = 0.0
    for a_, d in dossier.items():
        i = angles.index(a_)
        e_ref = abs(prof_e[i] - prof_r[i]) / prof_r[i]
        e_dos = abs(prof_e[i] - d) / d
        worst = max(worst, e_ref)
        print(f"   {a_:3d} deg: engine {prof_e[i]:.3f} ref {prof_r[i]:.3f} ({e_ref * 100:.1f}%) dossier {d:.3f} ({e_dos * 100:.1f}%)")
    print(f"  halo within 15% of the reference at 6/10/15/30 deg: {verdict(worst <= 0.15)} (worst {worst * 100:.1f}%)")


# --- reference ----------------------------------------------------------------------------------

def mode_reference(args):
    """Engine candidate potentials vs the grid Laplace reference at a 13-tree snapshot."""
    r2 = dbm.R2I
    r1 = args.r1_over_r2 * r2
    F = 13
    g = make(args, f_max=F, c_f=2048)
    g.capture()
    g.reset(); g.configure(seed=args.seed); g.set_admit(1)
    seeded = [0]

    def admit_control(g):
        if seeded[0] == 0 and int((g.t_state.numpy() != dbm.FREE).sum()) >= F:
            g.set_admit(0)
            seeded[0] = 1

    grow_until(g, args.nodes, max_frames=args.steps, r2=r2, on_frame=admit_control)
    nd = g.nodes()
    cand = g.candidates()
    print(f"[reference] snapshot: {len(nd['ids'])} nodes in {int((g.t_state.numpy() != dbm.FREE).sum())} trees, "
          f"{len(cand['ids'])} live candidates, maxr {np.linalg.norm(nd['pos'], axis=1).max() / r2:.2f} R2")
    # warm-CG residual and Born error at this snapshot
    rel, _, _ = g.cg_residual()
    q4 = g.q.numpy().copy()
    born = (g.birth_frame.numpy() == g.frame()) & ((g.flags.numpy() & dbm.ALIVE) != 0)
    phi_born_state = cand["phi"].copy()
    g.solve(200)
    qc = g.q.numpy()
    charged = (g.flags.numpy() & dbm.CHARGED) != 0
    phi_conv = g.potential(cand["pos"])
    e_carry = np.abs(q4[born] - qc[born]) / np.abs(qc[born])
    print(f"  warm CG ({g.n_cg} it) relative residual this frame: {rel:.2e} -> < 1e-3: {verdict(rel < 1e-3)}  "
          f"({int(born.sum())} nodes appended this frame)")
    print(f"  q after {g.n_cg} warm it vs converged (200 it): rms {harness.rmse(q4[charged], qc[charged]) / np.abs(qc[charged]).max():.4f} "
          f"of max|q|; appended nodes mean rel err {e_carry.mean():.3f} -> <= 5%: {verdict(e_carry.mean() <= 0.05)}")
    print(f"  candidate phi from the in-frame Born state vs converged: Spearman {harness.spearman(phi_born_state, phi_conv):.4f} "
          f"RMSE {harness.rmse(phi_born_state, phi_conv):.4f} (mean |phi| {np.abs(phi_conv).mean():.4f})")
    # cross-tree perturbation of the Born state: phi at candidates of OTHER trees than those that appended
    trees_born = set(np.unique(g.tree.numpy()[born]).tolist())
    other = np.array([p not in trees_born for p in cand["pool"]])
    if other.any():
        d = np.abs(phi_born_state[other] - phi_conv[other]) / np.maximum(np.abs(phi_conv[other]), 1e-6)
        print(f"  cross-tree perturbation (candidates of trees without appends): median {np.median(d) * 100:.2f}% p90 {np.percentile(d, 90) * 100:.2f}%")
    # pure Born (0 it) on a fresh copy of the same run
    g0 = make(args, f_max=F, c_f=2048, n_cg=0)
    g0.reset(); g0.configure(seed=args.seed); g0.set_admit(1)
    seeded[0] = 0
    grow_until(g0, args.nodes, max_frames=args.steps, r2=r2, on_frame=admit_control)
    b0 = (g0.birth_frame.numpy() == g0.frame()) & ((g0.flags.numpy() & dbm.ALIVE) != 0)
    qb = g0.q.numpy().copy()
    g0.solve(200)
    qc0 = g0.q.numpy()
    e0 = np.abs(qb[b0] - qc0[b0]) / np.abs(qc0[b0])
    print(f"  pure Born q = -{args.born_gain:g} a phi vs re-solved on {int(b0.sum())} in-frame nodes: mean rel err {e0.mean():.3f} "
          f"median {np.median(e0):.3f} (|q_born|/|q| median {np.median(np.abs(qb[b0]) / np.abs(qc0[b0])):.2f}) -> <= 5%: {verdict(e0.mean() <= 0.05)}")
    # cross-tree perturbation: error in a candidate's phi from the Born charges of OTHER trees vs its own tree
    c0 = g0.candidates()
    alive0 = (g0.flags.numpy() & dbm.ALIVE) != 0
    pos0, tree0 = g0.pos.numpy(), g0.tree.numpy()
    dq = qb - qc0
    phi_c0 = g0.potential(c0["pos"])
    own = np.zeros(len(c0["ids"])); other = np.zeros(len(c0["ids"]))
    born_ids = np.nonzero(b0 & alive0)[0]
    r1_ = g0.param("r1")
    for j in born_ids:
        d = np.linalg.norm(c0["pos"] - pos0[j], axis=1)
        lj = np.linalg.norm(pos0[j])
        gpair = 1.0 / d - (r1_ / lj) / np.linalg.norm(c0["pos"] - pos0[j] * (r1_ ** 2 / lj ** 2), axis=1)
        contrib = dq[j] * gpair
        mine = c0["pool"] == tree0[j]
        own[mine] += contrib[mine]
        other[~mine] += contrib[~mine]
    rel_own = np.abs(own) / np.maximum(np.abs(phi_c0), 1e-4)
    rel_other = np.abs(other) / np.maximum(np.abs(phi_c0), 1e-4)
    print(f"  in-frame Born perturbation of candidate phi: own tree median {np.median(rel_own) * 100:.1f}% p90 {np.percentile(rel_own, 90) * 100:.1f}%; "
          f"cross-tree median {np.median(rel_other) * 100:.2f}% p90 {np.percentile(rel_other, 90) * 100:.2f}% -> cross-tree <= 5%: {verdict(np.percentile(rel_other, 90) <= 0.05)}")
    # warm-CG residual vs iterations at ~this append rate
    res_rows = []
    for ncg in (4, 6, 8, 12):
        gi = make(args, f_max=F, c_f=2048, n_cg=ncg)
        gi.reset(); gi.configure(seed=args.seed); gi.set_admit(1)
        seeded[0] = 0
        res = []
        grow_until(gi, args.nodes, max_frames=args.steps, r2=r2, on_frame=lambda g_: (admit_control(g_), res.append(g_.cg_residual()[0])))
        res_rows.append([ncg, f"{max(res[3:]):.1e}", f"{np.median(res[3:]):.1e}", gi.node_count()])
    print_table(res_rows, ["n_cg", "max rel residual", "median", "nodes"])
    print(f"  residual < 1e-3 at n_cg <= 4: {verdict(float(res_rows[0][1]) < 1e-3)}  at n_cg = 6: {verdict(float(res_rows[1][1]) < 1e-3)}  "
          f"at n_cg = 12: {verdict(float(res_rows[3][1]) < 1e-3)}")
    # A. off-lattice protocol: phi at the live candidates, grid sampled by trilinear interpolation
    rows = []
    a_default = g.param("a")
    for N in sorted(set([96, 128, args.n_ref])):
        ref = reference.Reference(N, r1, r2, device=args.device)
        ref.setup(nodes=nd["pos"], node_radius=a_default if N > 128 else 0.0)
        t0 = time.perf_counter()
        res = ref.solve(1e-8)
        phi_ref = ref.sample(cand["pos"])
        for aoh in (0.15, 0.25, 0.35):
            g.configure(a=aoh * dbm.H)
            g.solve(300)
            phi_e = g.potential(cand["pos"])
            rows.append([N, f"engine f32 CG, images={args.images}", aoh, f"{harness.spearman(phi_e, phi_ref):.4f}",
                         f"{harness.rmse(phi_e, phi_ref):.4f}",
                         f"{harness.rmse((phi_e - phi_e.min()) / (phi_e.max() - phi_e.min()), (phi_ref - phi_ref.min()) / (phi_ref.max() - phi_ref.min())):.4f}"])
        for images, label in ((0, "dense f64, no images (dossier)"), (1, "dense f64, electrode image"),
                              (2, "dense f64, electrode + glass images")):
            qd = reference.dense_charges(nd["pos"], r1, r2, a_default, images, 12)
            phi_d = reference.dense_potential(cand["pos"], nd["pos"], qd, r1, r2, images, 12)
            rows.append([N, label, args.a_over_h, f"{harness.spearman(phi_d, phi_ref):.4f}", f"{harness.rmse(phi_d, phi_ref):.4f}",
                         f"{harness.rmse((phi_d - phi_d.min()) / (phi_d.max() - phi_d.min()), (phi_ref - phi_ref.min()) / (phi_ref.max() - phi_ref.min())):.4f}"])
        print(f"  reference N={N}: {ref.iterations} CG it, rel residual {res:.1e}, {time.perf_counter() - t0:.1f}s, "
              f"cell {ref.dx * 1e3:.2f} mm = {ref.dx / dbm.H:.2f} h; mean phi_ref at candidates {phi_ref.mean():.4f}")
        del ref
    g.configure(a=a_default)
    print("  off-lattice protocol (engine candidates, trilinear grid sample):")
    print_table(rows, ["N_ref", "model", "a/h", "Spearman", "abs RMSE", "norm RMSE"])
    plan_rows = [r for r in rows if r[0] == 128 and r[1].startswith("engine")]
    ok = any(float(r[3]) >= 0.99 and float(r[4]) <= 0.2 for r in plan_rows)
    print(f"  Spearman >= 0.99 and abs RMSE <= 0.2 at the plan's 128^3 reference for some a/h: {verdict(ok)}")
    if args.n_ref != 128:
        best = [r for r in rows if r[0] == args.n_ref and r[1].startswith("engine")]
        ok2 = any(float(r[3]) >= 0.99 and float(r[4]) <= 0.2 for r in best)
        print(f"  (same at N={args.n_ref}: {verdict(ok2)})")
    # B. lattice protocol of the dossier: nodes snapped to cell centres, candidates = the lattice
    # frontier (exact grid values), charges from the float64 dense conductor system
    rows_b = []
    for N in (96, 128):
        ref = reference.Reference(N, r1, r2, device=args.device)
        snapped = ref.snap_to_cells(nd["pos"])
        ref.setup(nodes=snapped)
        ref.solve(1e-8)
        pts, phi_ref = ref.frontier_cells()
        for images, label in ((0, "no images (dossier)"), (1, "electrode image"), (2, "electrode + glass images")):
            for aoh in ((0.15, 0.25, 0.35) if images == 1 else (args.a_over_h,)):
                qd = reference.dense_charges(snapped, r1, r2, aoh * dbm.H, images, 12)
                phi_d = reference.dense_potential(pts, snapped, qd, r1, r2, images, 12)
                rows_b.append([N, len(snapped), len(pts), label, aoh, f"{harness.spearman(phi_d, phi_ref):.4f}",
                               f"{harness.rmse(phi_d, phi_ref):.4f}"])
        del ref
    print("  lattice-frontier protocol (dossier methodology: nodes snapped to cell centres, candidates = free cells "
          "6-adjacent to node cells, exact grid values, dense float64 conductor solve):")
    print_table(rows_b, ["N_ref", "cells", "frontier", "kernel", "a/h", "Spearman", "abs RMSE"])


# --- strike -----------------------------------------------------------------------------------

def strikes_within(g, frames, V, seed):
    g.reset(); g.configure(V=V, seed=seed); g.set_admit(1)
    for _ in range(frames):
        g.run(1)
    return int((g.t_state.numpy() != dbm.FREE).sum())


def threshold_voltage(g, seed, lo=1500.0, hi=4000.0, frames=10, iters=12):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if strikes_within(g, frames, mid, seed) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def mode_strike(args):
    r2 = dbm.R2I
    r1 = args.r1_over_r2 * r2
    g = make(args, f_max=4, c_f=1024)
    g.capture()
    print(f"[strike] E_bd0 = {dbm.E_BD0 / 1e3:.0f} kV/m, h = {dbm.H * 1e3} mm, u0(R1 + h) = "
          f"{(1 / r1 - 1 / (r1 + dbm.H)) / (1 / r1 - 1 / r2):.4f} -> analytic V_th = "
          f"{dbm.E_BD0 * dbm.H / ((1 / r1 - 1 / (r1 + dbm.H)) / (1 / r1 - 1 / r2)):.0f} V")
    rows = []
    for V in (2000.0, 2250.0, 2500.0, 2750.0, 3000.0, 5000.0):
        n = [strikes_within(g, 30, V, s) for s in (1, 2, 3)]
        rows.append([f"{V:.0f}", n])
    print_table(rows, ["V", "trees struck in 30 frames (seeds 1,2,3)"])
    no_2000 = all(strikes_within(g, 30, 2000.0, s) == 0 for s in (1, 2, 3))
    yes_2750 = all(strikes_within(g, 30, 2750.0, s) > 0 for s in (1, 2, 3))
    print(f"  no strikes at 2.0 kV: {verdict(no_2000)}   strikes at 2.75 kV: {verdict(yes_2750)}")
    vth_cold = threshold_voltage(g, 1)
    # hot channel: T/T0 = 1.1 along a radial path from the electrode
    g.set_hot_channels([((0, 0, r1), (0, 0, r1 + 12 * dbm.H), 3 * dbm.H, 1.1 * dbm.T0)])
    vth_hot = threshold_voltage(g, 1)
    g.set_hot_channels([])
    ratio = vth_hot / vth_cold
    print(f"  V_th cold {vth_cold:.0f} V, hot channel (T/T0 = 1.1) {vth_hot:.0f} V, ratio {ratio:.3f} -> ~0.92: "
          f"{verdict(0.89 <= ratio <= 0.94)}")
    # hot channel steering: where do the first roots strike?
    g.set_hot_channels([((0, 0, r1), (0, 0, r1 + 12 * dbm.H), 3 * dbm.H, 1.1 * dbm.T0)])
    ang = []
    for s in range(1, 9):
        g.reset(); g.configure(V=vth_cold * 0.97, seed=s); g.set_admit(1)
        for _ in range(10):
            g.run(1)
        nd = g.nodes()
        roots = nd["pos"][(nd["flags"] & dbm.ROOT) != 0]
        if len(roots):
            ang.append(float(harness.angles_to_direction(roots[:1], (0, 0, 1))[0]))
    g.set_hot_channels([])
    print(f"  at 0.97 V_th(cold) with the hot channel: {len(ang)}/8 seeds struck, root angle to the hot channel "
          f"{[round(a, 2) for a in ang]} rad")


# --- touch ------------------------------------------------------------------------------------

def mode_touch(args):
    r2 = dbm.R2I
    F = args.trees if args.trees > 1 else 6
    finger = np.array([0.0, 0.0, 1.0])
    g = make(args, f_max=F, c_f=2048)
    g.capture()
    rows = []
    best = None
    for qf in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4):
        g.set_fingers([(finger, qf)] if qf > 0 else [])
        within, maxr, contacts_total = [], [], []
        for seed in (1, 2, 3):
            g.reset(); g.configure(seed=seed); g.set_admit(1)
            seeded = [0]

            def admit_control(g):
                if seeded[0] == 0 and int((g.t_state.numpy() != dbm.FREE).sum()) >= F:
                    g.set_admit(0)
                    seeded[0] = 1

            grow_until(g, args.nodes, max_frames=args.steps, r2=r2, on_frame=admit_control)
            nd = g.nodes()
            feet = nd["pos"][(nd["flags"] & dbm.FOOT) != 0]
            r = np.linalg.norm(nd["pos"], axis=1)
            maxr.append(r.max() / r2)
            if len(feet):
                a = harness.angles_to_direction(feet, finger)
                within.append(float((a < 0.5).mean()))
                contacts_total.append(len(feet))
            else:
                within.append(0.0)
                contacts_total.append(0)
        rows.append([qf, f"{np.mean(within):.2f}", contacts_total, f"{np.mean(maxr):.3f}"])
        if best is None and np.mean(within) >= 0.3 and np.mean(maxr) >= 0.94:
            best = qf
    print(f"[touch] finger at +z, d_f = (R2 + h_f) n_f, {F} trees to {args.nodes} nodes, 3 seeds; "
          f"isotropic expectation of contacts within 0.5 rad = {(1 - math.cos(0.5)) / 2:.3f}")
    print_table(rows, ["q_f", "contacts within 0.5 rad", "contacts (seeds)", "maxr/R2"])
    print(f"  >= 30% of contacts within 0.5 rad with maxr unchanged: {verdict(best is not None)}"
          + (f" at q_f = {best}" if best is not None else ""))
    # dynamic protocol: finger added to an established assembly; frames to visible bending / capture
    seeded = [0]

    def admit_control(g):
        if seeded[0] == 0 and int((g.t_state.numpy() != dbm.FREE).sum()) >= F:
            g.set_admit(0)
            seeded[0] = 1

    dyn = []
    for qf in (0.02, 0.05, 0.1, 0.2):
        bends, captures = [], []
        for seed in (7, 8, 9):
            g.set_fingers([])
            g.reset(); g.configure(seed=seed); g.set_admit(1)
            seeded[0] = 0
            grow_until(g, args.nodes // 2, max_frames=args.steps, r2=r2, on_frame=admit_control)
            base_ids = set(g.nodes()["ids"].tolist())
            feet0 = set(g.nodes()["ids"][(g.nodes()["flags"] & dbm.FOOT) != 0].tolist())
            d_f = (r2 + g.param("hf")) * finger

            def step_cos(g, frame):
                nd = g.nodes()
                new = (nd["birth_frame"] == frame) & (nd["parent"] >= 0)
                if new.sum() == 0:
                    return float("nan")
                idx = {i: k for k, i in enumerate(nd["ids"])}
                par = np.array([idx[p_] for p_ in nd["parent"][new]])
                step = nd["pos"][new] - nd["pos"][par]
                to_f = d_f[None, :] - nd["pos"][par]
                return float(np.mean(np.einsum("ij,ij->i", step, to_f)
                                     / (np.linalg.norm(step, axis=1) * np.linalg.norm(to_f, axis=1))))

            base = []
            for _ in range(5):
                g.run(1)
                base.append(step_cos(g, g.frame()))
            baseline = float(np.nanmean(base))
            base_ids = set(g.nodes()["ids"].tolist())
            feet0 = set(g.nodes()["ids"][(g.nodes()["flags"] & dbm.FOOT) != 0].tolist())
            g.set_fingers([(finger, qf)])
            bend_frame, capture_frame = None, None
            for f in range(1, 61):
                g.run(1)
                nd = g.nodes()
                if bend_frame is None and step_cos(g, g.frame()) - baseline >= 0.2:
                    bend_frame = f
                newfeet = np.array([(i not in feet0) for i in nd["ids"]]) & ((nd["flags"] & dbm.FOOT) != 0)
                if capture_frame is None and newfeet.any():
                    if (harness.angles_to_direction(nd["pos"][newfeet], finger) < 0.5).any():
                        capture_frame = f
                if bend_frame is not None and capture_frame is not None:
                    break
            bends.append(bend_frame); captures.append(capture_frame)
        dyn.append([qf, bends, captures, verdict(all(b is not None and b <= 5 for b in bends)),
                    verdict(all(c is not None and 12 <= c <= 30 for c in captures))])
    print("  dynamic protocol (finger added to a half-grown assembly, seeds 7,8,9): first frame at which the mean cosine "
          "between growth steps and the finger direction rises >= 0.2 above the pre-finger baseline; frame of the first "
          "new contact within 0.5 rad")
    print_table(dyn, ["q_f", "bend frame", "capture frame", "bend in 2-5 frames", "capture in 0.2-0.5 s"])


# --- bench ------------------------------------------------------------------------------------

def mode_bench(args):
    """Script B. Every timing sample restores the same device snapshot first (the graphs keep
    committing nodes), so each row is measured at its stated node count; growth during the
    timed replays is printed. The step graph holds two steps (parity of the staged extrema),
    so it replays as a live steady state; us/step is half of it."""
    print(f"[bench] Script B: captured graphs replayed from a restored snapshot, median of 5 samples; "
          f"S_MAX={args.s_max} n_cg={args.n_cg} mid_cg={args.mid_cg} images={args.images}; GPU shared with other agents")
    rows = []
    for n_nodes in (1000, 4000, 5000, 20000):
        for c_max in (8192, 32768, 65536):
            f_max = 32
            n_max = 8192 if n_nodes <= 5000 else 24576
            g = dbm.Dbm(device=args.device, n_max=n_max, f_max=f_max, c_f=c_max // f_max, images=args.images,
                        n_cg=args.n_cg, mid_cg=tuple(int(v) for v in args.mid_cg.split(",")), s_max=args.s_max)
            g.configure(eta=args.eta, V=args.voltage)
            g.capture()
            g.set_admit(1)
            frames = 0
            while g.node_count() < n_nodes and frames < 3000:
                g.run(1)
                frames += 1
            snap = g.snapshot()
            n0, c0 = g.node_count(), len(g.candidates()["ids"])
            free0 = int(g.cnt.numpy()[dbm.CNT_FREE_TOP])
            with wp.ScopedDevice(args.device):
                with wp.ScopedCapture() as cap:
                    g._growth_step(0)
                    g._growth_step(1)
                step_graph = cap.graph
                with wp.ScopedCapture() as cap2:
                    g._rebase(0)
                rebase_graph = cap2.graph
                with wp.ScopedCapture() as cap3:
                    g._cg()
                cg_graph = cap3.graph

            def timed(graph, reps, samples=5):
                vals = []
                for _ in range(samples):
                    g.restore(snap)
                    wp.capture_launch(graph)
                    wp.synchronize()
                    g.restore(snap)
                    with wp.ScopedTimer("bench", print=False, synchronize=True) as timer:
                        for _ in range(reps):
                            wp.capture_launch(graph)
                    vals.append(timer.elapsed / reps)
                return statistics.median(vals), g.node_count()

            t_step, n_step = timed(step_graph, 10)
            t_rebase, _ = timed(rebase_graph, 20)
            t_cg, _ = timed(cg_graph, 10)
            t_frame, n_frame = timed(g.graph, 2)
            rows.append([n0, c0, c_max, free0, f"{0.5 * t_step * 1e3:.1f}", n_step, f"{t_rebase * 1e3:.1f}",
                         f"{t_cg * 1e3:.1f}", f"{t_frame:.3f}", n_frame])
            del snap, step_graph, rebase_graph, cg_graph, g
    print_table(rows, ["nodes", "live cands", "C_MAX", "free ids", "us/step (2 kernels)", "nodes after 20 steps",
                       "us re-base", "us CG", "ms/frame graph", "nodes after 2 frames"])
    frame_ok = all(float(r[8]) <= 2.0 for r in rows if r[0] <= 5500)
    step_ok = all(float(r[4]) <= 40.0 for r in rows if 3500 <= r[0] <= 4500)
    print(f"  whole growth stage (frame graph, S_MAX={args.s_max}) <= 2.0 ms at n <= 5k: {verdict(frame_ok)}   "
          f"growth step <= 40 us at n = 4k: {verdict(step_ok)}")


# --- replay -----------------------------------------------------------------------------------

def snapshot(g):
    return dict(pos=g.pos.numpy().copy(), q=g.q.numpy().copy(), flags=g.flags.numpy().copy(),
                cnt=g.cnt.numpy().copy(), t_state=g.t_state.numpy().copy(), cand_phi=g.cand_phi.numpy().copy(),
                cand_alive=g.cand_alive.numpy().copy(), s_arc=g.s_arc.numpy().copy(), cg=g.cg.numpy().copy())


def mode_replay(args):
    g = make(args, f_max=13, c_f=2048)
    g.configure(enable_reroute=1)
    g.capture()
    snaps = []
    t0 = time.perf_counter()
    for rep in range(2):
        g.reset()
        g.run(600)
        snaps.append(snapshot(g))
    same = all(np.array_equal(snaps[0][k].view(np.uint8) if snaps[0][k].dtype.kind == "f" else snaps[0][k],
                              snaps[1][k].view(np.uint8) if snaps[1][k].dtype.kind == "f" else snaps[1][k])
               for k in snaps[0])
    diffs = [k for k in snaps[0] if not np.array_equal(snaps[0][k].view(np.uint8) if snaps[0][k].dtype.kind == "f" else snaps[0][k],
                                                       snaps[1][k].view(np.uint8) if snaps[1][k].dtype.kind == "f" else snaps[1][k])]
    print(f"[replay] captured once, 2 x 600 frames in {time.perf_counter() - t0:.1f}s, nodes {int((snaps[0]['flags'] & dbm.ALIVE).astype(bool).sum())}, "
          f"frame counter {snaps[0]['cnt'][0]}: bit-identical {verdict(same)}" + (f" (differs: {diffs})" if diffs else ""))
    # captured vs uncaptured must agree too
    g.reset()
    for _ in range(50):
        g.step()
    s_un = snapshot(g)
    g.reset()
    g.run(50)
    s_gr = snapshot(g)
    same2 = all(np.array_equal(s_un[k], s_gr[k]) for k in s_un)
    print(f"  captured vs uncaptured 50 frames bit-identical: {verdict(same2)}")
    # a captured reset() must equal a host reset() and lead to the same run (no host temporaries)
    with wp.ScopedDevice(args.device):
        with wp.ScopedCapture() as cap:
            g.reset()
        reset_graph = cap.graph
    g.run(20)
    wp.capture_launch(reset_graph)
    s_cap = snapshot(g)
    g.run(20)
    g.reset()
    s_host = snapshot(g)
    same3 = all(np.array_equal(s_cap[k], s_host[k]) for k in s_cap) and \
        s_cap["cnt"].tolist() == [0, g.n_max, 0]
    wp.capture_launch(reset_graph)
    g.run(50)
    s_again = snapshot(g)
    same4 = all(np.array_equal(s_again[k], s_gr[k]) for k in s_again)
    print(f"  captured reset == host reset (cnt {s_cap['cnt'].tolist()}): {verdict(same3)}   "
          f"50 frames after a captured reset bit-identical: {verdict(same4)}")


# --- nan-scan ---------------------------------------------------------------------------------

def mode_nan_scan(args):
    g = make(args, f_max=32, c_f=2048)
    g.configure(enable_reroute=1)
    g.capture()
    bad_frames, worst = [], dict(nodes=0, cands=0, decaying=0)
    t0 = time.perf_counter()
    frames = args.frames
    for f in range(0, frames, 500):
        g.run(min(500, frames - f))
        bad = harness.find_nonfinite_members(g)
        cnt = g.cnt.numpy()
        nodes = g.node_count()
        worst["nodes"] = max(worst["nodes"], nodes)
        worst["cands"] = max(worst["cands"], len(g.candidates()["ids"]))
        topo = harness.tree_topology_violations(g.nodes(), dbm.H)
        worst["decaying"] = max(worst["decaying"], int(((g.flags.numpy() & dbm.DECAYING) != 0).sum()))
        if bad or cnt[dbm.CNT_FREE_TOP] < 0 or nodes > g.n_max or cnt[dbm.CNT_FREE_TOP] + nodes != g.n_max \
                or topo["total"]:
            bad_frames.append((g.frame(), bad, int(cnt[dbm.CNT_FREE_TOP]), nodes, topo))
    st = g.trees()["state"]
    print(f"[nan-scan] {g.frame()} frames in {time.perf_counter() - t0:.1f}s with re-routes enabled: nodes now {g.node_count()} "
          f"(max {worst['nodes']}), candidates max {worst['cands']}, decaying max {worst['decaying']}, "
          f"tree states {np.bincount(st, minlength=7)}, free list consistent, topology invariant "
          f"(parent alive / same tree / not decaying / within 1.5 h) and NaN/Inf-free: {verdict(not bad_frames)}"
          + (f" {bad_frames[:3]}" if bad_frames else ""))


# --- check ------------------------------------------------------------------------------------

def host_pair_potential(x, xj, r1, a, images):
    """Host twin of dbm.pair_potential (inverse multiquadric, Kelvin image)."""
    d = np.linalg.norm(x - xj, axis=1)
    g = 1.0 / np.sqrt(d * d + a * a)
    if images:
        lj = np.linalg.norm(xj)
        di = np.linalg.norm(x - xj * (r1 * r1 / (lj * lj)), axis=1)
        g = g - (r1 / lj) / np.sqrt(di * di + a * a)
    return g


def mode_check(args):
    """Bookkeeping invariants of the engine (the adversarial-review probes)."""
    r2 = dbm.R2I
    # 1. a fresh root keeps K - 1 live candidates after its first growth step; the ring advances
    g = make(args, f_max=1, c_f=1024, s_max=1)
    g.reset(); g.set_admit(1)
    for _ in range(50):
        g.run(1)
        if g.node_count():
            break
    nd = g.nodes()
    root = int(nd["ids"][(nd["flags"] & dbm.ROOT) != 0][0])
    K = g.k
    spawned = int((g.cand_parent.numpy()[:K] == root).sum())
    alive1 = int((g.candidates()["parent"] == root).sum())
    geometric = int((np.linalg.norm(g.cand_pos.numpy()[:K], axis=1) >= g.param("r1") + g.param("a") - 1e-7).sum())
    tail1 = int(g.t_tail.numpy()[0])
    g.set_admit(0)
    g.run(1)
    nd2 = g.nodes()
    alive2 = int((g.candidates()["parent"] == root).sum())
    tail2 = int(g.t_tail.numpy()[0])
    children = int((nd2["parent"] == root).sum())
    child_slots = int((g.cand_parent.numpy()[K:2 * K] == nd2["ids"][nd2["parent"] == root][0]).sum()) if children == 1 else -1
    ok1 = spawned == K and alive1 == geometric and tail1 == K and children == 1 and alive2 == alive1 - 1 \
        and tail2 == 2 * K and child_slots == K
    print(f"[check] root ring: K={K} spawned {spawned} in slots [0,K), alive {alive1} (geometric {geometric}), t_tail {tail1}; "
          f"after the second step: root children {children}, root candidates alive {alive2}, child's candidates in [K,2K) "
          f"{child_slots}, t_tail {tail2} -> {verdict(ok1)}")
    # 2. incremental candidate potentials stay exact while trees idle (free list exhausted):
    #    n_cg = 0 keeps q = Born, so cand_phi must equal the host potential of the current charges
    #    minus the last step's commits (t_new) for candidates not spawned in that step
    g = make(args, f_max=4, c_f=512, n_max=256, n_cg=0)
    g.reset(); g.set_admit(1)
    r1, a_node, images = g.param("r1"), g.param("a"), g.param("images")
    worst, rows = 0.0, []
    for _ in range(12):
        g.run(1)
        c = g.candidates()
        if len(c["ids"]) == 0:
            continue
        expected = g.potential(c["pos"])
        last_step = g.frame() * g.s_max + g.s_max - 1
        stamp = g.cand_stamp.numpy()[c["ids"]]
        t_new = g.t_new.numpy()[:g.f_max]
        pos, q = g.pos.numpy().astype(np.float64), g.q.numpy().astype(np.float64)
        contrib = np.zeros(len(expected))
        for j in t_new[t_new >= 0]:
            contrib += q[j] * host_pair_potential(c["pos"].astype(np.float64), pos[j], r1, a_node, images)
        expected[stamp != last_step] -= contrib[stamp != last_step]
        err = float(np.abs(c["phi"] - expected).max())
        worst = max(worst, err)
        rows.append([g.frame(), g.node_count(), int(g.cnt.numpy()[dbm.CNT_FREE_TOP]), len(c["ids"]),
                     int((t_new >= 0).sum()), f"{err:.1e}"])
    print_table(rows, ["frame", "nodes", "free ids", "live cands", "trees committed last step", "max |dphi|"])
    print(f"  incremental (Kim Eq.11) candidate potential exact through idle steps (max |dphi| {worst:.1e} <= 1e-4): "
          f"{verdict(worst <= 1e-4)}")
    # 3. re-route: topology invariant and timer arming (no re-route right after an attachment)
    F = 13
    g = make(args, f_max=F, c_f=2048)
    g.configure(enable_reroute=1)
    g.capture()
    g.reset(); g.set_admit(1)
    prev = g.t_state.numpy().copy()
    prev_rg = g.t_regrow.numpy().copy()
    attach = {}
    gaps, viol, worst_topo = [], [], dict(total=0)
    decaying_max = 0
    for f in range(1, args.frames + 1 if args.frames < 1000 else 601):
        g.run(1)
        st = g.t_state.numpy()
        rg = g.t_regrow.numpy()
        for t in range(F):
            if st[t] == dbm.ATTACHED and prev[t] != dbm.ATTACHED:
                attach[t] = f
            if st[t] == dbm.ATTACHED and rg[t] != 0 and prev_rg[t] == 0 and t in attach:
                gaps.append(f - attach[t])        # re-route armed in place (the tree stays ATTACHED)
                del attach[t]
        prev = st.copy()
        prev_rg = rg.copy()
        if f % 25 == 0:
            topo = harness.tree_topology_violations(g.nodes(), dbm.H)
            decaying_max = max(decaying_max, int(((g.flags.numpy() & dbm.DECAYING) != 0).sum()))
            if topo["total"]:
                viol.append((f, topo))
            worst_topo = max(worst_topo, topo, key=lambda d: d["total"])
    gaps = np.array(gaps)
    ok3 = not viol
    ok4 = len(gaps) > 0 and gaps.min() >= 1 and np.median(gaps) >= 30   # Poisson timer: a 1-frame gap is legitimate
    print(f"  re-route run ({F} trees, {g.frame()} frames, re-routes enabled): nodes {g.node_count()}, "
          f"trees {int((g.t_state.numpy() != dbm.FREE).sum())}, decaying max {decaying_max}, topology violations "
          f"{worst_topo} -> {verdict(ok3)}")
    print(f"  frames from attachment to the first re-route ({len(gaps)} events, Poisson mean 120 expected): "
          f"min {gaps.min() if len(gaps) else None} median {np.median(gaps) if len(gaps) else None} "
          f"max {gaps.max() if len(gaps) else None} -> no re-route in the attachment frame: {verdict(ok4)}")


# --- main -------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plasma globe growth lab (headless, no DRM/GL)")
    p.add_argument("mode", choices=["grow", "pools", "halo", "reference", "strike", "touch", "bench", "replay", "nan-scan", "check"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--steps", type=int, default=2000, help="maximum frames per run")
    p.add_argument("--nodes", type=int, default=2000, help="target node count per run")
    p.add_argument("--frames", type=int, default=10000, help="frames for nan-scan")
    p.add_argument("--trees", type=int, default=1)
    p.add_argument("--eta", type=float, default=dbm.ETA)
    p.add_argument("--gamma", type=float, default=dbm.GAMMA)
    p.add_argument("--voltage", type=float, default=dbm.V_DEFAULT)
    p.add_argument("--a-over-h", type=float, default=dbm.A_OVER_H)
    p.add_argument("--r1-over-r2", type=float, default=dbm.R1 / dbm.R2I)
    p.add_argument("--global-norm", type=int, default=0)
    p.add_argument("--grow-after-attach", type=int, default=1)
    p.add_argument("--images", type=int, default=1, help="Kelvin images of the node charges in the electrode")
    p.add_argument("--n-cg", type=int, default=6)
    p.add_argument("--s-max", type=int, default=dbm.S_MAX, help="unrolled growth steps per frame (16 or 8)")
    p.add_argument("--n-ref", type=int, default=128)
    p.add_argument("--born-gain", type=float, default=dbm.BORN_GAIN)
    p.add_argument("--mid-cg", default="0,0", help="every,iterations: mid-frame CG sweeps + re-base")
    p.add_argument("--save", default=None, help="save the grown nodes to this .npz (grow mode)")
    args = p.parse_args()
    {"grow": mode_grow, "pools": mode_pools, "halo": mode_halo, "reference": mode_reference,
     "strike": mode_strike, "touch": mode_touch, "bench": mode_bench, "replay": mode_replay,
     "nan-scan": mode_nan_scan, "check": mode_check}[args.mode](args)


if __name__ == "__main__":
    main()
