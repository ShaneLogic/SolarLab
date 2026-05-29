"""Convert physics symbols in a markdown doc to real <sub>/<sup> typography.

Protects code fences/spans, links, and HTML; converts only prose physics
symbols (V_oc → V<sub>oc</sub>, cm^-3 → cm<sup>-3</sup>, etc.). Sheet-name
identifiers (Nt_C_PVK, Nd_ETL) have a different token shape and are left alone.
Idempotent. Usage: python md_physics_typography.py <file.md> [--check]
"""
import re, sys

PROTECT = re.compile(r'(```.*?```|`[^`]*`|\]\([^)]*\)|</?[A-Za-z][^>]*>)', re.DOTALL)

# longer keys first so V_bi_eff beats V_bi
SUBS = [
    ("V_bi_eff", "V<sub>bi,eff</sub>"), ("V_oc", "V<sub>oc</sub>"),
    ("V_bi", "V<sub>bi</sub>"), ("V_max", "V<sub>max</sub>"),
    ("V_app", "V<sub>app</sub>"), ("V_bend", "V<sub>bend</sub>"),
    ("V_total", "V<sub>total</sub>"), ("V_T", "V<sub>T</sub>"),
    ("J_sc", "J<sub>sc</sub>"), ("J_0", "J<sub>0</sub>"),
    ("J_n", "J<sub>n</sub>"), ("J_p", "J<sub>p</sub>"),
    ("N_t", "N<sub>t</sub>"), ("N_T", "N<sub>t</sub>"),
    ("N_D", "N<sub>D</sub>"), ("N_A", "N<sub>A</sub>"),
    ("N_C", "N<sub>C</sub>"), ("N_V", "N<sub>V</sub>"),
    ("E_t", "E<sub>t</sub>"), ("E_g", "E<sub>g</sub>"),
    ("E_C", "E<sub>C</sub>"), ("E_V", "E<sub>V</sub>"),
    ("E_Fn", "E<sub>Fn</sub>"), ("E_F", "E<sub>F</sub>"),
    ("B_rad", "B<sub>rad</sub>"), ("C_n", "C<sub>n</sub>"),
    ("C_p", "C<sub>p</sub>"), ("v_th", "v<sub>th</sub>"),
    ("Phi_b", "Φ<sub>b</sub>"), ("P_esc", "P<sub>esc</sub>"),
]
# build one regex per symbol with boundaries that avoid mid-token (code-like) hits
SUB_PATS = [(re.compile(r'(?<![A-Za-z0-9_])' + re.escape(k) + r'(?![A-Za-z0-9_])'), v)
            for k, v in SUBS]
SUP = re.compile(r'\b(cm|m|s|µm|nm)\^(-?\d+)')


def _sup(m):
    return f"{m.group(1)}<sup>{m.group(2).replace('-', '−')}</sup>"


def convert_prose(s):
    for pat, rep in SUB_PATS:
        s = pat.sub(rep, s)
    s = SUP.sub(_sup, s)
    return s


def convert(text):
    out = []
    for i, seg in enumerate(PROTECT.split(text)):
        # PROTECT.split with one capture group -> odd indices are protected
        out.append(seg if i % 2 else convert_prose(seg))
    return "".join(out)


def main():
    path = sys.argv[1]
    check = "--check" in sys.argv
    src = open(path).read()
    new = convert(src)
    if check:
        print("unchanged" if new == src else "WOULD CHANGE")
        return
    open(path, "w").write(new)
    print(f"normalized {path}")


if __name__ == "__main__":
    main()
