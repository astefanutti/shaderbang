# Copyright (C) 2025 Antonin Stefanutti <antonin.stefanutti@gmail.com>
# SPDX-License-Identifier: MIT

"""
Noble-gas emission colours
==========================

Linear-sRGB colours of the neutral (I) and singly-ionised (II) line emission of Ne, Ar, Kr and
Xe for the plasma-globe renderer (plan 4.9): NIST "strong lines" x CIE 1931 2-degree colour
matching functions -> XYZ -> linear sRGB (Rec. 709 primaries, D65), gamut-mapped and normalised
to unit luminance. Per segment the renderer then uses ``mix(neutral, ion, x_ion) * class * I_k``.

Data
----
``data/nist_lines.json`` holds the visible (380-780 nm, air wavelengths) rows of the NIST
Handbook of Basic Atomic Spectroscopic Data "Strong Lines" tables (Sansonetti, Martin & Young),
https://physics.nist.gov/PhysRefData/Handbook/Tables/<element>table2.htm, with provenance
(URL, fetch date, columns, reference codes). ``fetch_nist()`` regenerates it. ``BAKED`` carries
the resulting colours so the module works without the JSON (``bake()`` regenerates it).

Caveats
-------
- NIST relative intensities are a per-reference, roughly photographic / visual scale: they are
  comparable within one (species, stage, reference) list, NOT between Ne I (SS04) and Ne II
  (P71), nor between elements. Each species/stage colour is therefore a *chromaticity* only
  (normalised to luminance Y = 1); the neutral:ion balance is the renderer's ``x_ion`` knob and
  the overall gain is exposure. Kr II mixes two references (DHM33, HP70a) and Xe II two (H39 +
  one HP87 line); their scales are assumed comparable (unverified).
- The compiled "strong lines" lists mix source conditions. Measured consequence: with every
  Ne I line the colour is a pale salmon (xy 0.400 / 0.341, purity 0.3) because the 453-489 nm
  Ne I lines are listed at 100-150 against 200 for 585.2 nm, whereas a neon glow is orange-red.
  The NIST *persistent* lines (flag P; for Ne I the sixteen 2p5 3s - 2p5 3p lines 540-725 nm
  that make the neon look) give xy 0.588 / 0.410, dominant 593 nm. ``selection="persistent"``
  restricts a species to its persistent lines (falls back to all lines when a stage has none:
  Ne II, Xe I); the default is the raw list, presets pick per component (Ne I persistent, the
  rest all lines - Ar I / Kr I keep their glow-tube lavender only with the blue lines in).
- Lines are intensity-weighted deltas (no line shape, no self-absorption, no Doppler width).
- CMFs are the Wyman, Sloan & Shirley 2013 multi-lobe piecewise-Gaussian fit (JCGT 2(2), eq. 4,
  table 1; max squared error 2e-4 against the 1 nm tables) - no data file.
- Every line spectrum is outside the sRGB gamut. Negative channels are removed by desaturating
  toward the equal-luminance white (luminance preserved, hue kept), never by clipping channels
  (which shifts both hue and luminance).

Presets
-------
``ne_xe`` (the PPPL-4485 Ne + ~2 % Xe fill) mixes species as follows: neutral = Ne I persistent
lines (the orange-red 585-725 nm cluster; Xe I is weak, <= 60 on its scale, and only 2 % of the
gas);
ionised = luminance blend of 75 % Xe II + 25 % Ne II. Justification: Ne + Xe is a Penning
mixture - Ne metastables (16.6 eV) ionise Xe (12.13 eV) at ~5e-11 cm^3/s, i.e. in ~40 ns at
2 % Xe and 740 Torr (n_Xe ~ 4.8e17 cm^-3), ~100x faster than the ~3.5 us three-body loss of
Ne*, so nearly every Ne metastable becomes a Xe+ and the ionised-stage light (Xe II 484-605 nm,
blue-green/white) dominates far beyond the 2 % mole fraction, while Ne II (21.6 eV to ionise,
blue 440 nm lines) only appears in the hottest core. The 75 % share is a CHOSEN knob (the
xenon fraction is unmeasured, dossier open question); PPPL's "filaments are blue, red at the
glass" is reproduced with x_ion high on the shaft and low at the foot. ``ne``, ``ar``, ``kr``
and ``xe`` are single-element presets (neutral = X I, ionised = X II; Ne I persistent lines).

Run ``python examples/plasma/spectra.py`` to print the colour table and write the swatch grid
``/tmp/plasma_spectra.png``; ``--fetch`` re-downloads the NIST tables; ``--bake`` prints the
``BAKED`` source.
"""

import argparse
import datetime
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).resolve().parent / "data" / "nist_lines.json"
NIST_URL = "https://physics.nist.gov/PhysRefData/Handbook/Tables/{element}table2.htm"
NIST_SOURCE = ("NIST Handbook of Basic Atomic Spectroscopic Data, 'Strong Lines' tables "
               "(J. E. Sansonetti, W. C. Martin, S. L. Young), https://physics.nist.gov/Handbook")
ELEMENTS = {"Ne": "neon", "Ar": "argon", "Kr": "krypton", "Xe": "xenon"}
STAGES = {1: "I", 2: "II"}
VISIBLE_NM = (380.0, 780.0)

# Wyman, Sloan & Shirley 2013, table 1: (alpha, beta [nm], gamma [1/nm, left], delta [1/nm, right]).
CMF_LOBES = (
    ((0.362, 442.0, 0.0624, 0.0374), (1.056, 599.8, 0.0264, 0.0323), (-0.065, 501.1, 0.0490, 0.0382)),
    ((0.821, 568.8, 0.0213, 0.0247), (0.286, 530.9, 0.0613, 0.0322)),
    ((1.217, 437.0, 0.0845, 0.0278), (0.681, 459.0, 0.0385, 0.0725)),
)

# XYZ (D65) -> linear sRGB, Rec. 709 primaries (IEC 61966-2-1).
XYZ_TO_RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]])
LUMA = np.array([0.2126, 0.7152, 0.0722])
D65_xy = np.array([0.31271, 0.32902])

# Unit-luminance, gamut-mapped linear sRGB per "<element> <stage>[ persistent]", regenerated by
# ``bake()`` from data/nist_lines.json (fetched 2026-09-13). Used when the JSON is missing.
BAKED = {
    "Ne I": [1.8901, 0.7694, 0.6630],
    "Ne I persistent": [2.8842, 0.5409, 0.0000],
    "Ne II": [1.8849, 0.0000, 8.3002],
    "Ne II persistent": [1.8849, 0.0000, 8.3002],
    "Ar I": [2.7879, 0.1116, 4.5355],
    "Ar I persistent": [2.4974, 0.6558, 0.0000],
    "Ar II": [0.1308, 0.0562, 12.9085],
    "Ar II persistent": [0.0000, 0.7754, 6.1690],
    "Kr I": [1.6879, 0.7127, 1.8205],
    "Kr I persistent": [1.4781, 0.9588, 0.0000],
    "Kr II": [0.0000, 0.6932, 6.9841],
    "Kr II persistent": [0.0000, 0.5169, 8.7297],
    "Xe I": [1.8465, 0.7515, 0.9687],
    "Xe I persistent": [1.8465, 0.7515, 0.9687],
    "Xe II": [0.9279, 1.0782, 0.4381],
    "Xe II persistent": [1.1884, 1.0419, 0.0307],
}

# (element, stage, luminance share, line selection) per component.
PRESETS = {
    "ne_xe": {"neutral": (("Ne", 1, 1.0, "persistent"),),
              "ion": (("Xe", 2, 0.75, "all"), ("Ne", 2, 0.25, "all"))},
    "ne": {"neutral": (("Ne", 1, 1.0, "persistent"),), "ion": (("Ne", 2, 1.0, "all"),)},
    "ar": {"neutral": (("Ar", 1, 1.0, "all"),), "ion": (("Ar", 2, 1.0, "all"),)},
    "kr": {"neutral": (("Kr", 1, 1.0, "all"),), "ion": (("Kr", 2, 1.0, "all"),)},
    "xe": {"neutral": (("Xe", 1, 1.0, "all"),), "ion": (("Xe", 2, 1.0, "all"),)},
}
SELECTIONS = ("all", "persistent")


def cmf(wavelength_nm):
    """CIE 1931 2-degree x̄, ȳ, z̄ at ``wavelength_nm`` (array), shape (3, N)."""
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    out = np.zeros((3,) + wl.shape)
    for c, lobes in enumerate(CMF_LOBES):
        for alpha, beta, gamma, delta in lobes:
            t = (wl - beta) * np.where(wl < beta, gamma, delta)
            out[c] += alpha * np.exp(-0.5 * t * t)
    return out


def parse_nist_table(text, element):
    """Rows [wavelength_nm, rel_intensity, element, stage, flags, reference] of one NIST
    'Strong Lines' HTML page, restricted to ``VISIBLE_NM`` (air wavelengths above 200 nm)."""
    def strip(cell):
        return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()

    rows = []
    for inten, wl, spectrum, ref in re.findall(
            r"<tr><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td><td>(.*?)</td>", text, flags=re.S):
        m = re.match(r"(\d+)\s*(.*)", strip(inten))
        if not m:
            continue
        try:
            wavelength_nm = float(strip(wl)) / 10.0
        except ValueError:
            continue
        symbol, roman = strip(spectrum).split()
        if symbol != element or roman not in STAGES.values():
            continue
        if not VISIBLE_NM[0] <= wavelength_nm <= VISIBLE_NM[1]:
            continue
        stage = 1 if roman == "I" else 2
        rows.append([round(wavelength_nm, 4), int(m.group(1)), element, stage,
                     m.group(2).replace(" ", ""), strip(ref)])
    return rows


def fetch_nist(path=DATA_PATH, timeout=60.0):
    """Download the four NIST tables, parse them and write ``path``. Returns the data dict."""
    lines = []
    urls = {}
    for element, name in ELEMENTS.items():
        url = NIST_URL.format(element=name)
        urls[element] = url
        request = urllib.request.Request(url, headers={"User-Agent": "shaderbang-plasma/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("latin-1")
        rows = parse_nist_table(text, element)
        if not rows:
            raise RuntimeError(f"no visible lines parsed from {url}")
        lines.extend(rows)
    data = {
        "provenance": {
            "source": NIST_SOURCE,
            "urls": urls,
            "fetched": datetime.date.today().isoformat(),
            "columns": ["wavelength_nm", "rel_intensity", "element", "stage", "flags", "reference"],
            "wavelength": "air wavelength in nm (NIST lists Angstrom; divided by 10), "
                          f"restricted to {VISIBLE_NM[0]:.0f}-{VISIBLE_NM[1]:.0f} nm",
            "intensity": "NIST relative intensity; comparable only within one (element, stage, "
                         "reference) list, not across stages, elements or references",
            "flags": "P persistent line; NIST character codes c complex, d double, h hazy, "
                     "l shaded to longer wavelengths, s sharp, w wide",
            "references": sorted({(r[2], r[3], r[5]) for r in lines}),
        },
        "lines": lines,
    }
    body = json.dumps(data["provenance"], indent=2)
    rows = ",\n".join("    " + json.dumps(r) for r in lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{\n  "provenance": ' + body.replace("\n", "\n  ") + ',\n  "lines": [\n'
                    + rows + "\n  ]\n}\n")
    return data


_cache = {}


def load_lines(path=DATA_PATH):
    """The parsed JSON at ``path`` (cached per path), or None when the file is missing."""
    path = Path(path).resolve()
    if path not in _cache and path.exists():
        _cache[path] = json.loads(path.read_text())
    return _cache.get(path)


def lines(element, stage, selection="all", data=None):
    """(N, 2) array of [wavelength_nm, rel_intensity] for one element and ionisation stage;
    ``selection`` 'all' or 'persistent' (NIST flag P, all lines when the stage has none)."""
    data = data or load_lines()
    if data is None:
        raise FileNotFoundError(f"{DATA_PATH} missing - run `python examples/plasma/spectra.py --fetch` "
                                "(species_rgb() and preset() fall back to BAKED without it)")
    rows = [r for r in data["lines"] if r[2] == element and r[3] == stage]
    if selection == "persistent":
        rows = [r for r in rows if "P" in r[4]] or rows
    elif selection != "all":
        raise ValueError(f"selection must be one of {SELECTIONS}")
    return np.array([(r[0], r[1]) for r in rows], dtype=np.float64).reshape(-1, 2)


def spectrum_to_xyz(wavelength_nm, intensity):
    """XYZ of a line spectrum (intensity-weighted deltas)."""
    return cmf(wavelength_nm) @ np.asarray(intensity, dtype=np.float64)


def xyz_to_rgb(xyz):
    """XYZ -> linear sRGB (may be negative)."""
    return XYZ_TO_RGB @ np.asarray(xyz, dtype=np.float64)


def luminance(rgb):
    return float(LUMA @ np.asarray(rgb, dtype=np.float64))


def gamut_map(rgb):
    """Move ``rgb`` toward the white of the same luminance until every channel is >= 0.

    rgb' = rgb + t (Y - rgb); the luminance Y is unchanged, the hue angle is kept and only the
    saturation drops, unlike per-channel clipping."""
    rgb = np.asarray(rgb, dtype=np.float64)
    y = luminance(rgb)
    if y <= 0.0:
        return np.zeros(3)
    t = 0.0
    for c in range(3):
        if rgb[c] < 0.0:
            t = max(t, -rgb[c] / (y - rgb[c]))
    return rgb + min(t, 1.0) * (y - rgb)


def normalise(rgb):
    """Scale ``rgb`` to unit luminance."""
    rgb = np.asarray(rgb, dtype=np.float64)
    y = luminance(rgb)
    return rgb / y if y > 0.0 else np.zeros(3)


def dominant_wavelength(xy):
    """Dominant wavelength (nm) of chromaticity ``xy`` w.r.t. D65; negative = complementary
    wavelength of a purple (the ray from white hits the purple line)."""
    wl = np.arange(VISIBLE_NM[0], VISIBLE_NM[1] + 1.0)
    locus = cmf(wl)
    locus = (locus[:2] / locus.sum(axis=0)).T
    d = np.asarray(xy, dtype=np.float64) - D65_xy
    for sign in (1.0, -1.0):
        direction = sign * d
        a = locus[:-1]
        b = locus[1:] - a
        # solve D65 + s * direction = a + u * b
        det = direction[0] * (-b[:, 1]) - direction[1] * (-b[:, 0])
        rhs = a - D65_xy
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (rhs[:, 0] * (-b[:, 1]) - rhs[:, 1] * (-b[:, 0])) / det
            u = (direction[0] * rhs[:, 1] - direction[1] * rhs[:, 0]) / det
        hit = np.where((s > 0.0) & (u >= 0.0) & (u <= 1.0) & np.isfinite(s))[0]
        if hit.size:
            i = hit[np.argmin(s[hit])]
            return sign * float(wl[i] + u[i])
    return float("nan")


def species_key(element, stage, selection="all"):
    return f"{element} {STAGES[stage]}" + ("" if selection == "all" else f" {selection}")


def species_info(element, stage, selection="all", data=None):
    """Everything about one species: lines, XYZ, xyY, raw / mapped / unit-luminance rgb."""
    spectrum = lines(element, stage, selection, data)
    xyz = spectrum_to_xyz(spectrum[:, 0], spectrum[:, 1])
    total = xyz.sum()
    xy = xyz[:2] / total if total > 0.0 else np.zeros(2)
    rgb_raw = normalise(xyz_to_rgb(xyz))
    rgb = normalise(gamut_map(rgb_raw))
    order = np.argsort(-spectrum[:, 1])[:4]
    return {
        "key": species_key(element, stage, selection),
        "lines": spectrum,
        "dominant_lines": spectrum[order],
        "xyz": xyz,
        "xy": xy,
        "dominant_nm": dominant_wavelength(xy),
        "rgb_raw": rgb_raw,
        "rgb": rgb,
        "srgb8": srgb8(rgb),
    }


def species_rgb(species: str, stage: int, selection: str = "all") -> np.ndarray:
    """Unit-luminance linear-sRGB colour of ``species`` ('Ne', 'Ar', 'Kr', 'Xe') at ionisation
    ``stage`` (1 neutral, 2 singly ionised) using 'all' or 'persistent' NIST lines, from the
    JSON when present, else ``BAKED``."""
    if load_lines() is not None:
        return species_info(species, stage, selection)["rgb"]
    return np.array(BAKED[species_key(species, stage, selection)], dtype=np.float64)


def mix(neutral, ionised, x_ion):
    """Emission colour at ionised-line fraction ``x_ion`` in [0, 1] (a scalar -> rgb (3,), an
    array of N fractions -> (N, 3)); both inputs at unit luminance -> the result is at unit
    luminance (the blend is linear in radiance)."""
    x = np.clip(np.asarray(x_ion, dtype=np.float64), 0.0, 1.0)[..., None]
    return (1.0 - x) * np.asarray(neutral, dtype=np.float64) + x * np.asarray(ionised, dtype=np.float64)


def blend(components):
    """Luminance-weighted blend of ((element, stage, share, selection), ...), unit luminance."""
    rgb = sum(share * species_rgb(element, stage, selection)
              for element, stage, share, selection in components)
    return normalise(rgb)


def preset(name) -> dict:
    """{'name', 'neutral_rgb', 'ion_rgb'} for a gas preset ('ne_xe', 'ne', 'ar', 'kr', 'xe')."""
    p = PRESETS[name]
    return {"name": name, "neutral_rgb": blend(p["neutral"]), "ion_rgb": blend(p["ion"])}


def srgb8(rgb):
    """Display swatch of a (unit-luminance) linear colour: scaled to max channel 1, sRGB OETF,
    8 bits. Only for looking at - the renderer uses the linear values."""
    c = np.asarray(rgb, dtype=np.float64)
    c = np.clip(c / max(c.max(), 1e-12), 0.0, 1.0)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return tuple(int(round(v * 255.0)) for v in c)


def bake():
    """Python source of ``BAKED`` computed from the JSON."""
    rows = []
    for element in ELEMENTS:
        for stage in STAGES:
            for selection in SELECTIONS:
                info = species_info(element, stage, selection)
                rgb = info["rgb"]
                rows.append(f'    "{info["key"]}": [{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}],')
    return "BAKED = {\n" + "\n".join(rows) + "\n}"


def print_table(stream=sys.stdout):
    fmt = "{:<17} {:>5} {:>8} {:>7} {:>7}  {:<24} {:<16} {}"
    print(fmt.format("spec.", "lines", "dom. nm", "x", "y", "linear sRGB (Y = 1)", "sRGB8 swatch",
                     "strongest lines nm (rel. int.)"), file=stream)
    for element in ELEMENTS:
        for stage in STAGES:
            for selection in SELECTIONS:
                info = species_info(element, stage, selection)
                r = info["rgb"]
                dom = " ".join(f"{w:.1f}({int(i)})" for w, i in info["dominant_lines"])
                print(fmt.format(info["key"], len(info["lines"]), f"{info['dominant_nm']:.1f}",
                                 f"{info['xy'][0]:.4f}", f"{info['xy'][1]:.4f}",
                                 f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})", str(info["srgb8"]), dom),
                      file=stream)
    print("(dominant nm < 0: purple, complementary wavelength; 'persistent' rows equal 'all' when"
          " a stage has no P lines)", file=stream)
    print(file=stream)
    print("gamut mapping (raw unit-luminance linear sRGB -> desaturated toward equal-luminance white):",
          file=stream)
    for element in ELEMENTS:
        for stage in STAGES:
            info = species_info(element, stage)
            raw = info["rgb_raw"]
            print(f"  {info['key']:<6} raw ({raw[0]:+.3f}, {raw[1]:+.3f}, {raw[2]:+.3f})", file=stream)
    print(file=stream)
    print("presets (unit luminance):", file=stream)
    for name in PRESETS:
        p = preset(name)
        n, i = p["neutral_rgb"], p["ion_rgb"]
        print(f"  {name:<6} neutral ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}) {srgb8(n)}"
              f"   ion ({i[0]:.3f}, {i[1]:.3f}, {i[2]:.3f}) {srgb8(i)}", file=stream)
    drift = max(np.abs(species_info(e, s, sel)["rgb"] - BAKED[species_key(e, s, sel)]).max()
                for e in ELEMENTS for s in STAGES for sel in SELECTIONS)
    print(f"\nBAKED vs JSON max |delta| = {drift:.2e}"
          + ("" if drift < 1e-3 else "  <-- run --bake and update BAKED"), file=stream)


def swatch_png(path, tile=96, steps=9):
    """Grid of species swatches (rows 1-2: all / persistent lines) and preset x_ion ramps."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    keys = [(e, s) for e in ELEMENTS for s in STAGES]
    cols = max(len(keys), steps + 1)
    margin, label = 8, 18
    pitch = tile + 2 * label
    rows = len(SELECTIONS) + len(PRESETS)
    image = Image.new("RGB", (cols * tile + 2 * margin, rows * pitch + margin), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    ink = (230, 230, 230)
    for row, selection in enumerate(SELECTIONS):
        y0 = margin + row * pitch
        draw.text((margin, y0), f"species, {selection} NIST lines (unit luminance, swatch scaled to max channel 1)",
                  fill=ink, font=font)
        for c, (e, s) in enumerate(keys):
            info = species_info(e, s, selection)
            x0 = margin + c * tile
            draw.rectangle([x0, y0 + label, x0 + tile - 2, y0 + label + tile - 2], fill=info["srgb8"])
            draw.text((x0 + 4, y0 + label + tile), f"{e} {STAGES[s]} {info['dominant_nm']:.0f} nm", fill=ink, font=font)
    for row, name in enumerate(PRESETS, start=len(SELECTIONS)):
        p = preset(name)
        y0 = margin + row * pitch
        draw.text((margin, y0), f"preset {name}: mix(neutral, ion, x_ion), x_ion 0 -> 1", fill=ink, font=font)
        for c in range(steps + 1):
            rgb = mix(p["neutral_rgb"], p["ion_rgb"], c / steps)
            x0 = margin + c * tile
            draw.rectangle([x0, y0 + label, x0 + tile - 2, y0 + label + tile - 2], fill=srgb8(rgb))
            draw.text((x0 + 4, y0 + label + tile), f"{c / steps:.2f}", fill=ink, font=font)
    image.save(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="NIST line spectra -> linear sRGB emission colours")
    parser.add_argument("--fetch", action="store_true", help="re-download the NIST tables into data/nist_lines.json")
    parser.add_argument("--bake", action="store_true", help="print the BAKED dict source")
    parser.add_argument("--swatch", metavar="PNG", default="/tmp/plasma_spectra.png", help="swatch grid output")
    args = parser.parse_args(argv)
    if args.fetch:
        data = fetch_nist()
        print(f"wrote {DATA_PATH} ({len(data['lines'])} lines)")
    if load_lines() is None:
        print(f"{DATA_PATH} missing - run `{sys.argv[0]} --fetch` to download the NIST tables "
              "(species_rgb() and preset() still work from BAKED)", file=sys.stderr)
        return 1
    if args.bake:
        print(bake())
        return
    print_table()
    print(f"\nswatches: {swatch_png(args.swatch)}")


if __name__ == "__main__":
    sys.exit(main())
