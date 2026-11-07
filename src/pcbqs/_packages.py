"""Private numerical snapshot; independently maintained. See PROVENANCE.json."""
import re
SIZES = {'01005': (0.4, 0.2, 0.13), '0201': (0.6, 0.3, 0.26), '0402': (1.0, 0.5, 0.35), '0603': (1.6, 0.8, 0.45),
         '0805': (2.0, 1.25, 0.5), '1206': (3.2, 1.6, 0.55), '1210': (3.2, 2.5, 0.55), '1806': (4.5, 1.6, 0.6),
         '1812': (4.5, 3.2, 0.6), '2010': (5.0, 2.5, 0.6), '2512': (6.3, 3.2, 0.6)}

METRIC = {'0402': '01005', '0603': '0201', '1005': '0402', '1608': '0603', '2012': '0805', '3216': '1206',
          '3225': '1210', '4516': '1806', '4532': '1812', '5025': '2010', '6332': '2512'}


def package(lib):
    """The imperial chip size of a footprint name: 'Resistor_SMD:R_0402_1005Metric' -> '0402', 'C_1608Metric' ->
    '0603', 'L_0603' -> '0603'; None when the name has none."""
    name = str(lib or '').split(':')[-1]
    m = re.search(r'(?<!\d)(\d{4,5})_(\d{4})Metric', name)
    if m and m.group(1) in SIZES:
        return m.group(1)
    m = re.search(r'(?<!\d)(\d{4})Metric', name)
    if m and m.group(1) in METRIC:
        return METRIC[m.group(1)]
    m = re.search(r'(?<![\d])(01005|0201|0402|0603|0805|1206|1210|1806|1812|2010|2512)(?![\d])', name)
    return m.group(1) if m else None

