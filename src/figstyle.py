# -*- coding: utf-8 -*-
"""Shared matplotlib style for the figures.

Provides the rcParams (Arial with sans math, 9.5 pt base type, outward
ticks), a fixed entity registry (ENTITY) that pairs each color with a line
style and marker so figures stay readable in grayscale, panel headings,
legend and note placement clear of the data, an audit for type below the
size floor and for overlapping text, and a save routine that writes PNG and
PDF at a fixed printed width. FIG_W and HALF_W are printed widths in inches.
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito derived
BLACK = '#000000'
ORANGE = '#FFB300'
SKY = '#00A2FF'
GREEN = '#00963E'
BLUE = '#2962FF'
VERM = '#FF4E11'
PURPLE = '#E8318A'
GRAY = '#9E9E9E'

CYCLE = [BLACK, VERM, SKY, GREEN, PURPLE, BLUE, ORANGE, GRAY]

# printed widths (in); the single-column text width is 6.48 in
FIG_W = 6.5
HALF_W = 3.15

# type sizes (pt); audit() flags printed type below MIN_FONT
FS_BASE = 9.5
FS_PANEL = 10.5
FS_TITLE = 10.0
FS_LABEL = 9.5
FS_TICK = 8.5
FS_LEGEND = 8.5
FS_ANNOT = 7.5
FS_SMALL = 6.8
MIN_FONT = 6.5

MATH_MIN_FS = MIN_FONT / 0.7       # least size whose subscripts clear MIN_FONT

ENTITY = {
    # what produced a number
    'model': dict(color=BLACK, ls='-', marker='o',
                  label='coupled cable-deck model',
                  label_math='coupled cable-deck model'),
    'law': dict(color=GREEN, ls='--', marker='s',
                label='closed form',
                label_math=r'closed form, $\sqrt{d^2+s^2}-|d|$'),
    'incumbent': dict(color=VERM, ls=(0, (1.4, 1.3)), marker='D',
                      label='isolated-cable formula',
                      label_math='isolated-cable formula'),
    'control': dict(color=GRAY, ls=(0, (5, 2)), marker='x',
                    label='no coupling, control',
                    label_math='no coupling, control'),

    # stay mode order n; marker and dash keep the orders distinct in grayscale
    'n1': dict(color=BLACK, ls='-', marker='o', label='stay mode 1',
               label_math=r'$n = 1$'),
    'n2': dict(color=VERM, ls=(0, (4, 2)), marker='s', label='stay mode 2',
               label_math=r'$n = 2$'),
    'n3': dict(color=SKY, ls=(0, (1.4, 1.3)), marker='^', label='stay mode 3',
               label_math=r'$n = 3$'),
    'n4': dict(color=GREEN, ls=(0, (6, 2, 1.4, 2)), marker='D',
               label='stay mode 4', label_math=r'$n = 4$'),
    'n5': dict(color=PURPLE, ls=(0, (2.6, 1.6)), marker='v',
               label='stay mode 5', label_math=r'$n = 5$'),

    # tolerance bands on the screening criterion
    'tol2': dict(color=BLUE, ls='-', marker='none', label='tolerance 2 %',
                 label_math=r'tol $= \SI{2}{\percent}$'.replace('\\SI{2}{\\percent}', '2\\,\\%')),
    'tol5': dict(color=ORANGE, ls=(0, (4, 2)), marker='none',
                 label='tolerance 5 %', label_math=r'tol $= 5\,\%$'),
    'tol10': dict(color=PURPLE, ls=(0, (1.4, 1.3)), marker='none',
                  label='tolerance 10 %', label_math=r'tol $= 10\,\%$'),
}

ORDERS = ('n1', 'n2', 'n3', 'n4', 'n5')
TOLS = ('tol2', 'tol5', 'tol10')

MODELS = ('model', 'law', 'incumbent', 'control')


def entity_label(key, fontsize=FS_LEGEND, math=None):
    """Label for `key`; mathtext only where it prints at or above the floor."""
    e = ENTITY[key]
    if math is None:
        math = fontsize >= MATH_MIN_FS - 1e-9
    return e['label_math'] if math else e['label']


def style(key, fontsize=FS_LEGEND, math=None, label=True, **over):
    """Plot kwargs for a registered entity: color, dash, marker, label."""
    e = ENTITY[key]
    kw = dict(color=e['color'], ls=e['ls'], marker=e['marker'])
    if label:
        kw['label'] = entity_label(key, fontsize=fontsize, math=math)
    kw.update(over)
    return kw


def color(key):
    return ENTITY[key]['color']


def handle(key, fontsize=FS_LEGEND, math=None, **over):
    from matplotlib.lines import Line2D
    return Line2D([], [], **style(key, fontsize=fontsize, math=math, **over))


RC = {
    'font.size': FS_BASE,
    'font.family': 'sans-serif',
    # Arial first: matplotlib cannot extract the bold face from the macOS
    # Helvetica .ttc, so bold text would render at regular weight.
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'text.usetex': False,
    'axes.unicode_minus': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,

    'axes.titlesize': FS_TITLE, 'axes.titleweight': 'normal',
    'axes.titlepad': 5.0,
    'axes.labelsize': FS_LABEL, 'axes.labelweight': 'normal',
    'axes.linewidth': 1.0, 'axes.edgecolor': 'black',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'axes.axisbelow': True,
    'axes.xmargin': 0.05, 'axes.ymargin': 0.05,
    'axes.formatter.use_mathtext': True,
    'axes.prop_cycle': plt.cycler(color=CYCLE),

    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
    'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.7, 'ytick.minor.size': 1.7,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
    'xtick.color': 'black', 'ytick.color': 'black',
    'xtick.top': False, 'ytick.right': False,

    'legend.frameon': False, 'legend.fontsize': FS_LEGEND,
    'legend.title_fontsize': FS_LEGEND,
    'legend.handlelength': 2.2, 'legend.handleheight': 0.7,
    'legend.handletextpad': 0.55,
    'legend.columnspacing': 1.4, 'legend.labelspacing': 0.32,
    'legend.borderpad': 0.3, 'legend.borderaxespad': 0.0,
    'legend.markerscale': 1.0, 'legend.numpoints': 1, 'legend.scatterpoints': 1,
    'legend.framealpha': 0.92, 'legend.facecolor': 'white',
    'legend.edgecolor': '0.8', 'legend.fancybox': False,

    'lines.linewidth': 2.0, 'lines.markersize': 5.0,
    'patch.linewidth': 0.8, 'hatch.linewidth': 0.6,
    'errorbar.capsize': 2.0,
    'grid.color': '0.88', 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'grid.alpha': 1.0,

    'figure.figsize': (FIG_W, 2.65),
    'figure.facecolor': 'white', 'figure.edgecolor': 'none',
    'figure.dpi': 200, 'figure.constrained_layout.use': False,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'savefig.facecolor': 'white', 'savefig.edgecolor': 'none',
    'savefig.transparent': False,
}


def apply(**overrides):
    plt.rcParams.update(RC)
    if overrides:
        plt.rcParams.update(overrides)


def panel(ax, letter, title='', dy=1.03):
    """Panel heading outside the axes: letter and title both bold."""
    ax.text(-0.005, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight='bold', va='bottom', ha='left')
    if title:
        ax.annotate(title, xy=(-0.005, dy), xycoords='axes fraction',
                    xytext=(16, 1.2), textcoords='offset points',
                    fontsize=FS_LABEL, fontweight='bold',
                    va='bottom', ha='left', color='0.15')


def clean(ax, grid=False):
    """House axis treatment: no top/right spine, optional light y grid."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis='y', color='0.9', lw=0.6)
        ax.set_axisbelow(True)
    return ax


def headroom(ax, top=0.14, bottom=0.0):
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - bottom * span, hi + top * span)


def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def _mathtext_floor(text, size):
    """Smallest size mathtext renders inside this string.

    Matplotlib draws a script level at 0.7 of its parent and a
    scriptscript level at 0.5.
    """
    import re as _re
    smallest = size
    for seg in _re.findall(r'\$[^$]*\$', text):
        body = seg[1:-1]
        depth = 0
        for m in _re.finditer(r'[_^]', body):
            j = m.end()
            nested = body[j:j + 1] == '{' and _re.search(r'[_^]', body[j:j + 40])
            depth = max(depth, 2 if nested else 1)
        if depth:
            smallest = min(smallest, size * (0.5 if depth >= 2 else 0.7))
    return smallest


def audit(fig, min_fontsize=MIN_FONT, verbose=True, scale=1.0):
    """List type below the size floor and text that overlaps other text."""
    probs = []
    try:
        fig.canvas.draw()
        r = _renderer(fig)
    except Exception:
        return probs
    for i, ax in enumerate(fig.axes):
        boxes = []
        items = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title]
        items += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        lg = ax.get_legend()
        if lg is not None:
            items += list(lg.get_texts())
        for t in items:
            s = (t.get_text() or '').strip()
            if not s or not t.get_visible():
                continue
            eff = _mathtext_floor(s, t.get_fontsize()) * scale
            if eff < min_fontsize - 1e-6:
                probs.append(f'ax{i}: {eff:.2f} pt printed < {min_fontsize} pt '
                             f'on "{s[:28]}"')
            try:
                bb = t.get_window_extent(renderer=r)
            except Exception:
                continue
            if bb.width > 0 and bb.height > 0:
                boxes.append((s, bb))
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                ba, bb_ = boxes[a][1], boxes[b][1]
                if ba.overlaps(bb_):
                    ov = (min(ba.x1, bb_.x1) - max(ba.x0, bb_.x0)) * \
                         (min(ba.y1, bb_.y1) - max(ba.y0, bb_.y0))
                    if ov > 0.12 * min(ba.width * ba.height,
                                       bb_.width * bb_.height):
                        probs.append(f'ax{i}: "{boxes[a][0][:20]}" overlaps '
                                     f'"{boxes[b][0][:20]}"')
    if verbose and probs:
        print('  [audit] ' + '\n  [audit] '.join(probs))
    return probs


def bbox_artists(fig, margin=0.5):
    """Artists the tight bbox must contain.

    Axes.get_tightbbox collapses the width of an x label to one pixel, so
    a long label on an outer panel is cropped unless it is listed here.
    """
    extra = []
    try:
        extra += list(fig.get_default_bbox_extra_artists())
        for ax in fig.axes:
            extra += list(ax.get_default_bbox_extra_artists())
            for axis, lim, k in ((ax.xaxis, ax.get_xlim(), 0),
                                 (ax.yaxis, ax.get_ylim(), 1)):
                if not axis.get_visible():
                    continue
                if axis.label.get_text():
                    extra.append(axis.label)
                lo, hi = min(lim), max(lim)
                pad = 1e-9 + 1e-6 * abs(hi - lo)
                for t in axis.get_ticklabels():
                    try:
                        v = t.get_position()[k]
                    except Exception:
                        v = None
                    if v is None or (lo - pad) <= v <= (hi + pad):
                        extra.append(t)
                extra.append(axis.get_offset_text())
        extra = [a for a in extra if a.get_visible() and a.get_in_layout()]
        fig.canvas.draw()
        r = _renderer(fig)
        m = margin * fig.dpi
        x0, y0, x1, y1 = -m, -m, fig.bbox.width + m, fig.bbox.height + m
        keep = []
        for a in extra:
            try:
                bb = a.get_tightbbox(r)
            except Exception:
                bb = None
            if bb is None or (bb.x0 >= x0 and bb.x1 <= x1
                              and bb.y0 >= y0 and bb.y1 <= y1):
                keep.append(a)
    except Exception:
        return None
    return keep or None


# ---------------------------------------------------------------- geometry
# Helpers that place legends and notes in the emptiest region of the axes.


def _to_axes_frac(ax, pts):
    """Data coordinates to axes fraction, dropping non-finite rows."""
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.empty((0, 2))
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.empty((0, 2))
    f = (ax.transData + ax.transAxes.inverted()).transform(pts)
    return f[np.isfinite(f).all(axis=1)]


def occupancy(ax):
    """Every plotted vertex of `ax`, expressed in axes fraction."""
    chunks = []
    for ln in ax.lines:
        try:
            d = ln.get_xydata()
        except Exception:
            continue
        if d is not None and len(d):
            chunks.append(np.asarray(d, float))
    for col in ax.collections:
        try:
            o = np.asarray(col.get_offsets(), float)
        except Exception:
            continue
        if o.ndim == 2 and len(o):
            chunks.append(o)
    for pa in ax.patches:
        try:
            chunks.append(np.asarray(pa.get_bbox().corners(), float))
        except Exception:
            continue
    if not chunks:
        return np.empty((0, 2))
    f = _to_axes_frac(ax, np.vstack(chunks))
    if f.shape[0] == 0:
        return f
    keep = ((f[:, 0] > -0.02) & (f[:, 0] < 1.02) &
            (f[:, 1] > -0.02) & (f[:, 1] < 1.02))
    return f[keep]


_ANCHOR = {'left': 0.0, 'right': 1.0, 'lower': 0.0, 'upper': 1.0,
           'center': 0.5}

LEGEND_LOCS = ['upper right', 'upper left', 'lower right', 'lower left',
               'center right', 'center left', 'upper center', 'lower center']


def loc_box(loc, size=(0.38, 0.32), pad=0.015):
    """(x0, y0, w, h) in axes fraction of the region a legend `loc` fills."""
    w, h = size
    v, _, u = loc.partition(' ')
    ya, xa = _ANCHOR.get(v, 0.5), _ANCHOR.get(u or 'center', 0.5)
    x0 = pad if xa == 0.0 else (1.0 - pad - w if xa == 1.0 else 0.5 - 0.5 * w)
    y0 = pad if ya == 0.0 else (1.0 - pad - h if ya == 1.0 else 0.5 - 0.5 * h)
    return max(x0, 0.0), max(y0, 0.0), w, h


def legend_loc(ax, size=(0.38, 0.32), pad=0.015, candidates=None):
    """Loc string for the emptiest region of `ax` (ties break by order)."""
    try:
        pts = occupancy(ax)
    except Exception:
        return 'best'
    order = list(candidates or LEGEND_LOCS)
    if pts.shape[0] == 0:
        return order[0]
    best, best_n = order[0], None
    for loc in order:
        x0, y0, w, h = loc_box(loc, size=size, pad=pad)
        inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x0 + w) &
                  (pts[:, 1] >= y0) & (pts[:, 1] <= y0 + h))
        n = int(inside.sum())
        if best_n is None or n < best_n:
            best, best_n = loc, n
        if best_n == 0:
            break
    return best


def place_legend(ax, *args, size=(0.38, 0.32), **kw):
    """ax.legend() dropped into the emptiest region instead of over data."""
    if 'loc' not in kw and 'bbox_to_anchor' not in kw:
        kw['loc'] = legend_loc(ax, size=size)
    kw.setdefault('fontsize', FS_ANNOT)
    kw.setdefault('labelspacing', 0.22)
    kw.setdefault('handletextpad', 0.45)
    return ax.legend(*args, **kw)


def keep_inside(artist, ax=None, pad=0.012):
    """Nudge a Text so its rendered box stays inside the axes."""
    ax = ax if ax is not None else getattr(artist, 'axes', None)
    if ax is None:
        return artist
    try:
        r = _renderer(ax.figure)
        bb = artist.get_window_extent(renderer=r)
        inv = ax.transAxes.inverted()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    except Exception:
        return artist
    dx = dy = 0.0
    if x1 - x0 < 1.0 - 2 * pad:
        if x0 < pad:
            dx = pad - x0
        elif x1 > 1.0 - pad:
            dx = (1.0 - pad) - x1
    if y1 - y0 < 1.0 - 2 * pad:
        if y0 < pad:
            dy = pad - y0
        elif y1 > 1.0 - pad:
            dy = (1.0 - pad) - y1
    if dx == 0.0 and dy == 0.0:
        return artist
    try:
        p0 = ax.transAxes.transform((0.0, 0.0))
        p1 = ax.transAxes.transform((dx, dy))
        tr = artist.get_transform()
        here = tr.transform(artist.get_position())
        artist.set_position(tr.inverted().transform(here + (p1 - p0)))
    except Exception:
        pass
    return artist


def note(ax, x, y, text, pad=0.012, **kw):
    """Annotation in axes fraction, guaranteed to stay inside the axes."""
    kw.setdefault('fontsize', FS_ANNOT)
    kw.setdefault('clip_on', False)
    kw.setdefault('transform', ax.transAxes)
    t = ax.text(x, y, text, **kw)
    return keep_inside(t, ax, pad=pad)


def save(fig, path_png, check=True, normalise_width=True, target_w=None):
    """Write PNG (600 dpi) and a matching vector PDF.

    With `normalise_width` the tight crop is padded symmetrically out to
    `target_w` inches (default FIG_W), so every figure has the same printed
    width and type size. Content is never cropped.
    """
    if check:
        audit(fig)
    kw = {}
    extra = bbox_artists(fig)
    if extra and plt.rcParams['savefig.bbox'] == 'tight':
        kw['bbox_extra_artists'] = extra
    if normalise_width and plt.rcParams['savefig.bbox'] == 'tight':
        from matplotlib.transforms import Bbox
        tw = FIG_W if target_w is None else target_w
        try:
            bb = fig.get_tightbbox(_renderer(fig), bbox_extra_artists=extra)
            bb = bb.padded(plt.rcParams.get('savefig.pad_inches', 0.02))
            if bb.width < tw:
                dx = 0.5 * (tw - bb.width)
                bb = Bbox.from_extents(bb.x0 - dx, bb.y0, bb.x1 + dx, bb.y1)
            kw = {'bbox_inches': bb}
        except Exception:
            pass
    fig.savefig(path_png, **kw)
    fig.savefig(str(path_png).rsplit('.', 1)[0] + '.pdf', **kw)
    print(f'  wrote {path_png}')
