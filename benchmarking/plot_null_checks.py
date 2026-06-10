# Generate benchmark null-check plots. Add a small spec below for each probe.
# Run: python benchmarking/plot_null_checks.py              # all plots
#      python benchmarking/plot_null_checks.py cptac_pda_os # one plot

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
NULL_ROOT = ROOT / "null_plots"
REF = {
    "break_his": {
        "DINOv2-small": 0.46474252393773324,
        "DINOv2-giant": 0.6272390983730772,
        "GigaPath": 0.6243343994129004,
        "GenBio": 0.7172705650786622,
        "H-optimus-0": 0.7504974072495875,
    },
    "bracs": {
        "DINOv2-small": 0.5098926950408877,
        "DINOv2-giant": 0.5617357361355029,
        "GigaPath": 0.5671780598671213,
        "GenBio": 0.5997352732922685,
        "H-optimus-0": 0.5619480145399081,
    },
    "cptac_pda_os": {
        "DINOv2-small": 0.45418655434119133,
        "DINOv2-giant": 0.4922900741928302,
        "GigaPath": 0.5267446163085876,
        "GenBio": 0.4911377432613642,
        "H-optimus-0": 0.48601083132651213,
    },
    "leopard_bcr": {
        "DINOv2-small": 0.6205757202995551,
        "DINOv2-giant": 0.6774454222633349,
        "GigaPath": 0.6597327017291083,
        "GenBio": 0.7016340341869399,
        "H-optimus-0": 0.6636391552335322,
    },
    "consep": {
        "DINOv2-small": 0.20158898624373067,
        "DINOv2-giant": 0.2173301037446952,
        "GigaPath": 0.2374348148157729,
        "GenBio": 0.23090162054187516,
        "H-optimus-0": 0.22184144950207366,
    },
    "mhist": {
        "DINOv2-small": 0.7716593971109481,
        "DINOv2-giant": 0.800263188323488,
        "GigaPath": 0.7786124844891756,
        "GenBio": 0.7937309166367107,
        "H-optimus-0": 0.7934803257545178,
    },
    "monusac": {
        "DINOv2-small": 0.23847640060933575,
        "DINOv2-giant": 0.2654998076055546,
        "GigaPath": 0.3079366629883673,
        "GenBio": 0.33175441401332173,
        "H-optimus-0": 0.3391667788382647,
    },
    "pcam": {
        "DINOv2-small": 0.7939288381638873,
        "DINOv2-giant": 0.7749174702570425,
        "GigaPath": 0.9182702309587295,
        "GenBio": 0.912186785966442,
        "H-optimus-0": 0.9077556968955527,
    },
    "pannuke": {
        "DINOv2-small": 0.35871640554467055,
        "DINOv2-giant": 0.3703049828588618,
        "GigaPath": 0.4181057091822521,
        "GenBio": 0.4130311894509096,
        "H-optimus-0": 0.41212404118540436,
    },
    "pathorob": {
        "DINOv2-small": 0.7542676245548043,
        "DINOv2-giant": 0.7984753622433232,
        "GigaPath": 0.744849760947313,
        "GenBio": 0.941207938223459,
        "H-optimus-0": 0.8926200973478514,
    },
    "surgen": {
        "DINOv2-small": 0.6225068351934023,
        "DINOv2-giant": 0.6173598972106434,
        "GigaPath": 0.6262381784769845,
        "GenBio": 0.6374957046598837,
        "H-optimus-0": 0.6583561173113411,
    },
    "ucla_lung": {
        "DINOv2-small": 0.5826983283123633,
        "DINOv2-giant": 0.5999837556855101,
        "GigaPath": 0.7041024277866383,
        "GenBio": 0.7680459566424478,
        "H-optimus-0": 0.7004031543505227,
    },
}
NULL_CHECKS = {
    "break_his": {
        "title": "BreakHis Tile Classification: Mean F1 Null Check",
        "xlabel": "Mean of linear, KNN, and 16-shot macro F1",
        "output": NULL_ROOT / "break_his_null_distributions.png",
        "xlim": (0.30, 0.78),
        "bins": np.linspace(0.326, 0.353, 10),
        "tick_step": 0.05,
        "label_offsets": [36, 205, 130, 36, 205],
        "label_dx": [-54, 6, 42, 6, 6],
        "null": np.array([
            0.34238004878020883, 0.33790163269061385, 0.3280545316982834, 0.3362673956881947,
            0.34466748258570856, 0.33838927143531244, 0.3352858847860733, 0.3479248018682,
            0.3459556111869877, 0.3404701037529411, 0.3409570867921387, 0.34251798520606824,
            0.33937591731565897, 0.3464688331161147, 0.3503147003633873, 0.3463966879354272,
            0.3486021371859734, 0.34271880653872416, 0.33886846770990925, 0.3398386159466205,
        ]),
    },
    "bracs": {
        "title": "BRACS Tile Classification: Mean F1 Null Check",
        "xlabel": "Mean of linear, KNN, and 16-shot macro F1",
        "output": NULL_ROOT / "bracs_null_distributions.png",
        "xlim": (0.25, 0.62),
        "bins": np.linspace(0.272, 0.303, 11),
        "tick_step": 0.05,
        "label_offsets": [36, 205, 130, 36, 330],
        "label_dx": [-90, 6, 55, 16, -85],
        "null": np.array([
            0.2946858301328159, 0.28434805198139146, 0.283903372503787, 0.28159611165074433,
            0.3003254873640832, 0.2913262223135544, 0.2887118724021346, 0.27396959066921983,
            0.2825158573627888, 0.2875922916814255, 0.29360830004701094, 0.2851982618086952,
            0.273203023242095, 0.2864291234116277, 0.27672273149827137, 0.29028961043287743,
            0.287606186867787, 0.29838205869242446, 0.27452559407277005, 0.2813147916424203,
        ]),
    },
    "cptac_pda_os": {
        "title": "CPTAC-PDA OS Survival: Z-scored CoxPH Null Check",
        "xlabel": "3-fold validation mean Harrell c-index",
        "output": NULL_ROOT / "cptac_pda_os_null_distributions.png",
        "xlim": (0.44, 0.59),
        "bins": np.linspace(0.512, 0.582, 13),
        "tick_step": 0.02,
        "label_offsets": [36, 205, 130, 430, 300],
        "label_dx": [-66, 6, 6, 6, -50],
        "null": np.array([
            0.5352338281090659, 0.5361268079139841, 0.5814613953363484, 0.5568297828397438,
            0.5391837993545828, 0.5422322159453402, 0.5528777196067322, 0.5594970209178519,
            0.5406866739752894, 0.5328400656709358, 0.5218677521828855, 0.5449116841041870,
            0.5404137270794034, 0.5119515705354735, 0.5275386475149120, 0.5360607149024729,
            0.5377491097328821, 0.5292051799265570, 0.5288560019259159, 0.5533895208990011,
        ]),
    },
    "leopard_bcr": {
        "title": "LEOPARD BCR Survival: Z-scored CoxPH Null Check",
        "xlabel": "3-fold mean Harrell c-index",
        "output": NULL_ROOT / "leopard_bcr_null_distributions.png",
        "xlim": (0.60, 0.72),
        "bins": np.linspace(0.606, 0.660, 13),
        "tick_step": 0.02,
        "label_offsets": [36, 205, 130, 36, 280],
        "label_dx": [6, 6, -80, -20, 6],
        "null": np.array([
            0.6235557273283445, 0.6254057147067919, 0.6241510393274239, 0.6549725208403251,
            0.6061514312109039, 0.6333465847881637, 0.6321833472750234, 0.6255823576703392,
            0.6354031802551884, 0.6303184235423534, 0.6211300368521022, 0.6592838264594308,
            0.6378653487931778, 0.6339075307148748, 0.6373141121033155, 0.6333735811073374,
            0.6449291075264363, 0.6451085143039216, 0.6288095290947752, 0.6324277408235312,
        ]),
    },
    "consep": {
        "title": "CoNSeP Segmentation: Jaccard Null Check",
        "xlabel": "Mean validation macro Jaccard",
        "output": NULL_ROOT / "consep_null_distributions.png",
        "xlim": (0.19, 0.245),
        "bins": np.linspace(0.2265, 0.2295, 8),
        "ytick_step": 5,
        "label_offsets": [36, 205, 36, 130, 310],
        "label_dx": [-20, 6, 24, -34, 20],
        "null": np.array([
            0.22845494776596253, 0.22845494776596253, 0.22845494776596253, 0.22845494776596253,
            0.22845494776596253, 0.22845494776596253, 0.22845494776596253, 0.22845494776596253,
            0.22845494776596253, 0.22845494776596253, 0.22845494776596253, 0.22845494776596253,
            0.22845494776596253, 0.22845494776596253, 0.22845494776596253, 0.22845494776596253,
            0.22845494776596253, 0.22845494776596253, 0.22845494776596253, 0.22845494776596253,
        ]),
    },
    "mhist": {
        "title": "MHIST Tile Classification: Mean F1 Null Check",
        "xlabel": "Mean of linear, KNN, and 16-shot macro F1",
        "output": NULL_ROOT / "mhist_null_distributions.png",
        "xlim": (0.55, 0.83),
        "bins": np.linspace(0.56, 0.596, 11),
        "tick_step": 0.05,
        "label_offsets": [36, 205, 130, 36, 230],
        "label_dx": [80, -70, -55, 35, 20],
        "null": np.array([
            0.5748383699469759, 0.589209544256139, 0.5813660855565836, 0.5694003463227014,
            0.5732583198862998, 0.5715899845850573, 0.5898627670850272, 0.5758680780049786,
            0.562815502663418, 0.5937941595899332, 0.5699535524819179, 0.5800484414711572,
            0.5732934881999932, 0.5794174698241984, 0.5727261528038964, 0.5675980174778167,
            0.5779960546154754, 0.5766616309334563, 0.5840929852137693, 0.5775954172364434,
        ]),
    },
    "monusac": {
        "title": "MoNuSAC Segmentation: Jaccard Null Check",
        "xlabel": "Mean validation macro Jaccard",
        "output": NULL_ROOT / "monusac_null_distributions.png",
        "xlim": (0.22, 0.35),
        "bins": np.linspace(0.23, 0.275, 12),
        "tick_step": 0.02,
        "label_offsets": [36, 205, 36, 130, 230],
        "label_dx": [20, -38, 6, 6, 6],
        "null": np.array([
            0.25102350793514716, 0.25833596416090404, 0.23905209281842463, 0.260993687273262,
            0.2607769214129168, 0.23308295125197467, 0.25727547657993566, 0.26985282840736585,
            0.2324165507948149, 0.25546670497193874, 0.27015478637743096, 0.2538759943069189,
            0.243475438432299, 0.2541826777253977, 0.2724065436311398, 0.23432666077591424,
            0.26097396240568177, 0.24325177410434193, 0.26774260596742055, 0.26771988221387805,
        ]),
    },
    "pcam": {
        "title": "PCam Tile Classification: Mean F1 Null Check",
        "xlabel": "Mean of linear, KNN, and 16-shot macro F1",
        "output": NULL_ROOT / "pcam_null_distributions.png",
        "xlim": (0.70, 0.94),
        "bins": np.linspace(0.714, 0.746, 11),
        "tick_step": 0.05,
        "label_offsets": [36, 205, 36, 130, 230],
        "label_dx": [-60, 10, 55, 8, -55],
        "null": np.array([
            0.7242270869246005, 0.7168695697632129, 0.737802890448534, 0.738961728042879,
            0.7326534145909713, 0.7336904776423202, 0.743247963116693, 0.7209104972971813,
            0.7424814346707765, 0.7303622615700313, 0.72123238743288, 0.7320790722655509,
            0.7368173303872845, 0.717154320408827, 0.7249649913592613, 0.7366812917726601,
            0.7233437466821928, 0.729793379250185, 0.7223620104961522, 0.7393390077602118,
        ]),
    },
    "pannuke": {
        "title": "PanNuke Segmentation: Jaccard Null Check",
        "xlabel": "Validation macro Jaccard",
        "output": NULL_ROOT / "pannuke_null_distributions.png",
        "xlim": (0.28, 0.43),
        "bins": np.linspace(0.29, 0.326, 11),
        "tick_step": 0.02,
        "label_offsets": [36, 205, 36, 130, 230],
        "label_dx": [20, -30, 55, 5, -55],
        "null": np.array([
            0.30985192642214393, 0.2963690809246797, 0.3160960239721788, 0.2999642821344385,
            0.30092463581153994, 0.32137484815062717, 0.32465826590668484, 0.3129078170343522,
            0.2994557489944995, 0.3079902140842895, 0.3045075484753387, 0.30733865869192184,
            0.3156873973824911, 0.3124423301217579, 0.31345358507850973, 0.3076360288665467,
            0.29067295241252944, 0.32309956849045557, 0.32158607171838255, 0.2949835156890499,
        ]),
    },
    "pathorob": {
        "title": "PathoROB Robustness: Index Null Check",
        "xlabel": "Mean robustness index",
        "output": NULL_ROOT / "pathorob_null_distributions.png",
        "xlim": (0.15, 0.96),
        "bins": np.linspace(0.186, 0.201, 10),
        "tick_step": 0.10,
        "label_offsets": [36, 205, 130, 36, 230],
        "label_dx": [20, 50, -55, 6, 6],
        "null": np.array([
            0.19219812005393494, 0.1948872719366621, 0.18785053401737395, 0.18983673071386542,
            0.19915963520362834, 0.195029795020064, 0.19238789029821102, 0.193421922677872,
            0.1907174481793095, 0.19329121299160773, 0.19799302988455592, 0.1932379783144365,
            0.19344436036868726, 0.19599908497527307, 0.19664755635505984, 0.19443009815304405,
            0.19139190873940415, 0.1950853837597158, 0.19651853355083954, 0.19241083423206182,
        ]),
    },
    "surgen": {
        "title": "SurGen KRAS Mutation: AUROC Null Check",
        "xlabel": "3-fold validation AUROC",
        "output": NULL_ROOT / "surgen_null_distributions.png",
        "xlim": (0.55, 0.67),
        "bins": np.linspace(0.557, 0.581, 11),
        "tick_step": 0.02,
        "label_offsets": [36, 205, 130, 36, 230],
        "label_dx": [-45, 0, 45, 8, 6],
        "null": np.array([
            0.5739321410963202, 0.578810153437019, 0.570611656432552, 0.567944810482124,
            0.5612440799007964, 0.5690205130503638, 0.5637988735003661, 0.5778427681412756,
            0.5641014148476835, 0.5679037246201425, 0.5667570555630257, 0.576415968207013,
            0.5587191669281222, 0.5747725337277576, 0.5715939820417432, 0.5610050348856318,
            0.5734503159876295, 0.570144771637309, 0.577349737797499, 0.5659614838719317,
        ]),
    },
    "ucla_lung": {
        "title": "UCLA Lung Progression: Logistic Probe Null Check",
        "xlabel": "3-fold validation AUROC",
        "output": NULL_ROOT / "ucla_lung_null_distributions.png",
        "xlim": (0.56, 0.79),
        "bins": np.linspace(0.684, 0.702, 10),
        "label_offsets": [205, 36, 90, 36, 310],
        "null": np.array([
            0.6953009628448225, 0.6905162738496071, 0.6890764368834544, 0.6936543800578888,
            0.6906713332151928, 0.6967924862661704, 0.6890764368834544, 0.6858866442199775,
            0.6953009628448225, 0.6967407998109753, 0.6920077972709552, 0.6875332270069112,
            0.6890764368834544, 0.6906196467599978, 0.6905679603048025, 0.6906713332151928,
            0.6890764368834544, 0.6906713332151928, 0.6937577529682794, 0.699878906019257,
        ]),
    },
}
COLORS = {"null": "#F58518", "curve": "#B85700", "DINOv2-small": "#4C78A8", "DINOv2-giant": "#555555", "GigaPath": "#54A24B", "GenBio": "#B279A2", "H-optimus-0": "#E45756"}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def text_size(draw, text, f):
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0], box[3] - box[1]


def rotated(img, xy, text, f, color):
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox((0, 0), text, font=f, spacing=4)
    layer = Image.new("RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(layer).multiline_text((6, 6), text, font=f, fill=color, spacing=4)
    img.alpha_composite(layer.rotate(90, expand=True), xy)


def dashed(draw, x, y0, y1, color):
    for y in range(int(y0), int(y1), 18):
        draw.line([(x, y), (x, min(y + 10, y1))], fill=color, width=4)


def kde_counts(values, xs, bin_width):
    # Silverman bandwidth, scaled to histogram counts so the curve overlays bars.
    bw = max(float(bin_width) * 0.75, 1e-4, 1.06 * values.std(ddof=1) * len(values) ** -0.2)
    density = np.exp(-0.5 * ((xs[:, None] - values[None, :]) / bw) ** 2).sum(1) / (len(values) * bw * np.sqrt(2 * np.pi))
    return density * len(values) * bin_width


def draw_null_check(key):
    spec, refs = NULL_CHECKS[key], REF[key]
    values, bins, xlim = spec["null"], spec["bins"], spec["xlim"]
    counts = np.histogram(values, bins=bins)[0]
    W, H, L, R, T, B = 1600, 920, 160, 80, 160, 150
    plot_w, plot_h = W - L - R, H - T - B
    ymax = int(counts.max() + 1)
    img = Image.new("RGBA", (W, H), "white")
    draw = ImageDraw.Draw(img)
    fg, grid = "#252525", "#dedede"
    xp = lambda x: L + int((x - xlim[0]) / (xlim[1] - xlim[0]) * plot_w)
    yp = lambda y: T + plot_h - int(y / ymax * plot_h)

    draw.text((L, 38), spec["title"], font=font(38, True), fill=fg)
    draw.rectangle([L, 109, L + 32, 131], fill=COLORS["null"])
    draw.text((L + 44, 104), "DINOv2-small, randomized weights", font=font(23), fill=fg)
    for y in range(0, ymax + 1, spec.get("ytick_step", 1)):
        yy = yp(y)
        draw.line([(L, yy), (W - R, yy)], fill=grid, width=1)
        draw.text((L - 48, yy - 14), str(y), font=font(22), fill=fg)
    draw.line([(L, T), (L, T + plot_h), (W - R, T + plot_h)], fill=fg, width=3)
    tick_step = spec.get("tick_step", 0.02)
    for x in np.arange(np.ceil(xlim[0] / tick_step) * tick_step, xlim[1] + 0.001, tick_step):
        xx = xp(float(x))
        draw.line([(xx, T + plot_h), (xx, T + plot_h + 9)], fill=fg, width=3)
        draw.text((xx - 30, T + plot_h + 18), f"{x:.2f}", font=font(22), fill=fg)
    xw, _ = text_size(draw, spec["xlabel"], font(25))
    draw.text((L + (plot_w - xw) // 2, H - 62), spec["xlabel"], font=font(25), fill=fg)
    rotated(img, (34, T + plot_h // 2 + 70), "Number of runs", font(25), fg)

    for i, c in enumerate(counts):
        x0, x1 = xp(bins[i]), xp(bins[i + 1])
        pad = min(max(1, int((x1 - x0) * 0.10)), max(0, (x1 - x0 - 1) // 2))
        draw.rectangle([x0 + pad, yp(c), x1 - pad, yp(0)], fill=COLORS["null"])
    xs = np.linspace(max(xlim[0], values.min() - 0.02), min(xlim[1], values.max() + 0.02), 240)
    pts = [(xp(float(x)), yp(float(y))) for x, y in zip(xs, kde_counts(values, xs, float(np.diff(bins).mean())))]
    draw.line(pts, fill=COLORS["curve"], width=5, joint="curve")

    offsets = spec.get("label_offsets", [205, 36, 130, 36, 230])
    dx = spec.get("label_dx", [6, 6, 6, 6, 6])
    for i, name in enumerate(("DINOv2-giant", "DINOv2-small", "GigaPath", "GenBio", "H-optimus-0")):
        x = xp(refs[name])
        dashed(draw, x, T, T + plot_h, COLORS[name])
        rotated(img, (x + dx[i], T + offsets[i]), f"{name}\n{refs[name]:.3f}", font(24, True), COLORS[name])
    img.convert("RGB").save(spec["output"])
    print(spec["output"])


def main():
    NULL_ROOT.mkdir(exist_ok=True)
    keys = sys.argv[1:] or sorted(NULL_CHECKS)
    for key in keys:
        draw_null_check(key)


if __name__ == "__main__":
    main()
