"""HNL tables in data/hnl: the width of N -> l pi pi0 counted twice, and the u dbar charged-current matching that follows.

Each of the conjugate rows 2Pi<l> and 2Pi<l>bar carries twice the width of one charge state. At m_N = m_tau with electron
mixing the tables give (2Pie)/(Pie) = 4.49, while for the same kinematics tau decays give
BR(tau -> pi pi0 nu) / BR(tau -> pi nu) = 25.49 / 10.82 = 2.36, and arXiv:1805.08567 finds N -> l pi pi0 equal to
N -> l rho, whose width is fixed by that tau ratio. The one-pion rows (Pi<l>, Pi<l>bar) and the neutral current (2Piv)
are right.

The script halves the 2Pi rows and redoes the matching of the u dbar charged current as the source defines it: the
inclusive width Gamma(N -> l u dbar) + conjugate of 1805.08567, with m_u = m_d = m_pi and the three-loop Delta_QCD with
alpha_s(m_N) at three loops from alpha_s(m_Z) = 0.118, against the exclusive Pi + 2Pi rows. This inclusive width
reproduces the source's Jets-ud + Pi rows to 0.35% above its matching mass, and with the old 2Pi rows the source's own
matching node. With the corrected rows the inclusive width wins at a lower mass; from there to the old matching mass the
2Pi rows are replaced by Jets-ud rows equal to the inclusive width minus the Pi rows, as above the old matching. Total
widths and all branching ratios follow; numbers that do not change keep their text.

python tools/fix_hnl_two_pion.py [data/hnl]
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

GF, VUD, MPI, MZ = 1.1663788e-5, 0.974, 0.13957, 91.1876
LEPTONS = {"e": (0.000511, 2), "mu": (0.105658, 3), "tau": (1.77686, 4)}  # mass, column of the branching ratios


class Text(float):
    """A number that remembers its text in the source file"""
    def __new__(cls, text):
        value = super().__new__(cls, text)
        value.text = text
        return value


def number(value):
    if isinstance(value, Text):
        return value.text
    if isinstance(value, int):
        return str(value)
    if value == 0.:
        return "0.0"
    if 0.1 <= abs(value) < 10.:
        return repr(value)
    digits = 1
    while float(f"{value:.{digits}e}") != value:  # shortest round trip, as the source
        digits += 1
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    mantissa = mantissa.rstrip("0")
    mantissa = mantissa + "0" if mantissa.endswith(".") else mantissa
    return f"{mantissa}e{int(exponent)}"


def dump(item):
    if isinstance(item, list):
        return "[" + ",".join(dump(x) for x in item) + "]"
    if isinstance(item, str):
        return json.dumps(item).replace("/", "\\/")  # the source escapes slashes
    return number(item)


def lam(a, b, c):
    return a * a + b * b + c * c - 2. * a * b - 2. * a * c - 2. * b * c


def phase_space(xu, xd, xl):
    """I(x_u, x_d, x_l) of 1805.08567"""
    lo, hi = (xd + xl)**2, (1. - xu)**2
    if lo >= hi:
        return 0.
    x = np.linspace(lo, hi, 20001)[1:]
    f = 12. / x * (x - xl**2 - xd**2) * (1. + xu**2 - x) * np.sqrt(np.clip(lam(x, xl**2, xd**2), 0., None) * np.clip(lam(1., x, xu**2), 0., None))
    return float(np.trapezoid(f, x))


def alpha_s(q):
    """three loops from alpha_s(m_Z) = 0.118, flavour thresholds at 4.8 and 1.5 GeV"""
    def beta(a, nf):
        b0 = (33. - 2. * nf) / (12. * math.pi)
        b1 = (153. - 19. * nf) / (24. * math.pi**2)
        b2 = (2857. - 5033. / 9. * nf + 325. / 27. * nf**2) / (128. * math.pi**3)
        return -(b0 * a**2 + b1 * a**3 + b2 * a**4)
    a, t, steps = 0.118, math.log(MZ**2), 2000
    for lo, nf in ((4.8, 5), (1.5, 4), (q, 3)):
        target = math.log(max(lo, q)**2)
        h = (target - t) / steps
        for _ in range(steps):
            k1 = beta(a, nf); k2 = beta(a + h / 2 * k1, nf); k3 = beta(a + h / 2 * k2, nf); k4 = beta(a + h * k3, nf)
            a += h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
            if a > 5.:  # Landau pole; not reached at the masses used here
                return math.inf
        t = target
        if lo <= q:
            break
    return a


def inclusive(m, m_l):
    """Gamma(N -> l u dbar) + conjugate [GeV], |U|^2 = 1"""
    a = alpha_s(m) / math.pi
    return 2. * 3. * VUD**2 * GF**2 * m**5 / (192. * math.pi**3) * phase_space(MPI / m, MPI / m, m_l / m) * (1. + a + 5.2 * a**2 + 26.4 * a**3)


def main(directory):
    path_rows, path_widths = directory / "eventcalc_branching_ratios.json", directory / "eventcalc_total_widths.dat"
    source = path_rows.read_text()
    rows = json.loads(source, parse_float=Text)
    if dump(rows) != source:
        sys.exit("cannot reproduce the text of the table; nothing changed")
    width_text = path_widths.read_text()
    width_lines = width_text.splitlines()
    widths = [[Text(x) for x in line.split("\t")] for line in width_lines]
    masses = np.array([float(r[0]) for r in widths])
    index = {r[0]: r for r in rows}
    for lepton, (m_l, column) in LEPTONS.items():
        total = np.array([float(r[column - 1]) for r in widths])
        branching = {r[0]: np.array([float(p[1]) for p in r[column]]) for r in rows}
        assert all(np.allclose([p[0] for p in r[column]], masses) for r in rows)
        partial = {name: br * total for name, br in branching.items()}
        names = {k: (f"{k}{lepton}", f"{k}{lepton}bar") for k in ("Pi", "2Pi", "Jets-ud")}
        pi, pipi, jets = (partial[a] + partial[b] for a, b in (names["Pi"], names["2Pi"], names["Jets-ud"]))
        old = int(np.argmax(jets > 0.))
        calibration = (jets[old] + pi[old]) / inclusive(masses[old], m_l)
        candidates = [i for i in range(old) if pipi[i] > 0.]
        # the source's rule with its own rows: the inclusive width first wins at the old matching node
        assert calibration * inclusive(masses[old - 1], m_l) <= pi[old - 1] + pipi[old - 1], lepton
        incl, new = {}, old
        while new - 1 >= candidates[0]:  # walk down while the inclusive width beats the corrected exclusive one
            value = calibration * inclusive(masses[new - 1], m_l)
            if not value > pi[new - 1] + pipi[new - 1] / 2.:
                break
            incl[new - 1] = value
            new -= 1
        changed = range(candidates[0], old)
        for i in changed:
            if i >= new:
                partial[names["Jets-ud"][0]][i] = partial[names["Jets-ud"][1]][i] = (incl[i] - pi[i]) / 2.
                partial[names["2Pi"][0]][i] = partial[names["2Pi"][1]][i] = 0.
            else:
                partial[names["2Pi"][0]][i] /= 2.
                partial[names["2Pi"][1]][i] /= 2.
        invisible = total * (1. - sum(branching.values()))
        new_total = invisible + sum(partial.values())
        for name, row in index.items():
            for i in changed:
                row[column][i][1] = float(partial[name][i] / new_total[i])
        for i in changed:
            widths[i][column - 1] = float(new_total[i])
        ratio = new_total[list(changed)] / total[list(changed)]
        print(f"{lepton}: u dbar matching {masses[old - 1]:.4g}-{masses[old]:.4g} GeV -> {masses[new - 1]:.4g}-{masses[new]:.4g} GeV;"
              f" total width x {ratio.min():.3f} to {ratio.max():.3f} for {masses[changed[0]]:.4g} <= m_N < {masses[old]:.4g} GeV")
    path_rows.write_text(dump(rows))
    # the width table: Python's repr with the exponent as in the source (e-9, not e-09)
    path_widths.write_text("\n".join("\t".join(x.text if isinstance(x, Text) else re.sub(r"e([-+])0(\d)", r"e\1\2", repr(float(x)))
                                                for x in r) for r in widths)
                           + ("\n" if width_text.endswith("\n") else ""))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/hnl"))
