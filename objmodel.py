"""Объектное моделирование фаций дельты на кусте 34/10.

В отличие от этапа 2, где коллектор распределялся статистически, здесь тела
рисуются как геологические объекты с формой и размерами: извилистые русла,
линзы устьевых баров, веера песков разливов. Только так русло видно как русло.

Привязка к скважинам — отбором: предлагается тело, считается, стало ли
совпадение со скважинными данными лучше; ухудшающие тела отклоняются.
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(2026)

# --- фации
BG_SWAMP, BAY, SPLAY, ABAND, CHAN, BAR = 1, 2, 3, 4, 5, 6
NAMES = {BG_SWAMP: 'Дб — дельтовое болото', BAY: 'Дз — дельтовый залив',
         SPLAY: 'Пр — пески разливов', ABAND: 'Зк — заиливающийся канал',
         CHAN: 'Рк — распределительный канал', BAR: 'Уб — устьевой бар'}
SAND = {CHAN, BAR}

# целевые доли — с керновой разметки этапа 1
TARGET = {BG_SWAMP: .18, BAY: .23, SPLAY: .06, ABAND: .04, CHAN: .23, BAR: .26}

# --- сетка
CELL, DZ = 160.0, 3.0
TOP, THICK = 1383.0, 150.0
NZ = int(THICK / DZ)
AZIMUTH = np.deg2rad(20.)          # главное направление сноса (условно)


def grid():
    wd = pd.read_csv('well_cells.csv')
    wp = wd.groupby('WELL')[['X', 'Y']].mean()
    pad = 2600.
    gx = np.arange(wp.X.min() - pad, wp.X.max() + pad, CELL)
    gy = np.arange(wp.Y.min() - pad, wp.Y.max() + pad, CELL)
    return gx, gy, wd, wp


def well_arrays(wd, wp, gx, gy):
    """Скважинные данные в индексах сетки: (i, j, k, песок?)"""
    nzs = int(round(150 / 2))                      # исходная разбивка этапа 2 — 2 м
    rec = []
    for w, d in wd.groupby('WELL'):
        i = int(np.clip(round((wp.loc[w, 'X'] - gx[0]) / CELL), 0, len(gx) - 1))
        j = int(np.clip(round((wp.loc[w, 'Y'] - gy[0]) / CELL), 0, len(gy) - 1))
        for _, r in d.iterrows():
            z = TOP + (r.K + .5) * 2.0
            k = int(np.clip((z - TOP) / DZ, 0, NZ - 1))
            rec.append((i, j, k, int(r.SAND)))
    a = pd.DataFrame(rec, columns=['i', 'j', 'k', 'sand'])
    return a.groupby(['i', 'j', 'k']).sand.mean().round().astype(int).reset_index()


# ---------------------------------------------------------------- тела
def channel(nx, ny, sinuosity=1.0):
    """Осевая линия извилистого русла через всю площадь."""
    L = int(np.hypot(nx, ny) * 1.6)
    a = AZIMUTH + rng.normal(0, .35)
    # гладкий случайный поворот — сумма нескольких синусоид
    t = np.arange(L)
    wob = sum(rng.normal(0, 1) * np.sin(2 * np.pi * t / p + rng.uniform(0, 6.28))
              for p in (L / 1.5, L / 3.5, L / 7.))
    ang = a + sinuosity * .35 * wob / 3
    # старт за пределами площади с наветренной стороны
    x0 = rng.uniform(-.3, 1.3) * nx - np.sin(a) * nx * .8
    y0 = rng.uniform(-.3, 1.3) * ny - np.cos(a) * ny * .8
    xs = x0 + np.cumsum(np.sin(ang))
    ys = y0 + np.cumsum(np.cos(ang))
    return xs, ys


def paint_channel(vol, code, xs, ys, kbase, width_cells, thick_cells, protect):
    nx, ny, nz = vol.shape
    touched = []
    half = width_cells / 2
    r = int(np.ceil(half)) + 1
    for x, y in zip(xs, ys):
        i0, i1 = int(x) - r, int(x) + r + 1
        j0, j1 = int(y) - r, int(y) + r + 1
        if i1 <= 0 or j1 <= 0 or i0 >= nx or j0 >= ny:
            continue
        i0, j0 = max(i0, 0), max(j0, 0); i1, j1 = min(i1, nx), min(j1, ny)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
        d = np.hypot(ii - x, jj - y) / half
        m = d <= 1
        if not m.any():
            continue
        # дно вогнутое: в осевой части мощнее
        tk = np.round(thick_cells * np.sqrt(np.clip(1 - d ** 2, 0, 1))).astype(int)
        for a_, b_, t_ in zip(ii[m], jj[m], tk[m]):
            if t_ <= 0:
                continue
            k0 = max(kbase - t_ + 1, 0); k1 = min(kbase + 1, nz)
            touched.append((a_, b_, k0, k1))
    return touched


def paint_lens(vol, xs_c, ys_c, kbase, rx, ry, thick_cells, ang):
    nx, ny, nz = vol.shape
    r = int(np.ceil(max(rx, ry))) + 1
    i0, i1 = max(int(xs_c) - r, 0), min(int(xs_c) + r + 1, nx)
    j0, j1 = max(int(ys_c) - r, 0), min(int(ys_c) + r + 1, ny)
    if i1 <= i0 or j1 <= j0:
        return []
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    dx, dy = ii - xs_c, jj - ys_c
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    m = d <= 1
    tk = np.round(thick_cells * np.sqrt(np.clip(1 - d ** 2, 0, 1))).astype(int)
    out = []
    for a_, b_, t_ in zip(ii[m], jj[m], tk[m]):
        if t_ <= 0:
            continue
        k0 = max(kbase, 0); k1 = min(kbase + t_, nz)
        out.append((a_, b_, k0, k1))
    return out


# ---------------------------------------------------------------- основное
def mismatch(vol, wells):
    v = vol[wells.i.values, wells.j.values, wells.k.values]
    pred = np.isin(v, list(SAND)).astype(int)
    return int(np.abs(pred - wells.sand.values).sum())


def build(vpc):
    gx, gy, wd, wp = grid()
    nx, ny = len(gx), len(gy)
    wells = well_arrays(wd, wp, gx, gy)
    vol = np.full((nx, ny, NZ), BAY, dtype=np.uint8)

    # фон: болото там, где по кривой пропорций мало коллектора
    kz = np.arange(NZ)
    vp = np.interp(kz, np.linspace(0, NZ - 1, len(vpc)), vpc)
    swamp_layers = kz[vp < np.quantile(vp, .35)]
    vol[:, :, swamp_layers] = BG_SWAMP

    # вероятность уровня для песчаного тела — по кривой пропорций
    pk = np.clip(vp, .02, None); pk = pk / pk.sum()

    total = nx * ny * NZ
    need = {c: int(TARGET[c] * total) for c in (CHAN, BAR, ABAND, SPLAY)}
    have = {c: 0 for c in need}
    mm = mismatch(vol, wells)
    tries = {c: 0 for c in need}
    order = [CHAN, BAR, CHAN, BAR, ABAND, SPLAY]
    oi = 0
    MAXTRY = 30000
    while any(have[c] < need[c] for c in need) and sum(tries.values()) < MAXTRY:
        # «отжиг»: сначала принимаем тела свободно, к концу — только улучшающие,
        # иначе крупные объекты почти всегда отклоняются и доли не набираются
        tol = max(0.0, 3.0 * (1 - sum(tries.values()) / (0.5 * MAXTRY)))
        code = order[oi % len(order)]; oi += 1
        if have[code] >= need[code]:
            continue
        tries[code] += 1
        # привязка к скважинам: с большой вероятностью тело протаскивается
        # через ячейку скважины, где керн/ГК говорят «песок», а в модели его нет
        tgt = None
        if code in (CHAN, BAR) and rng.random() < .8:
            cur = vol[wells.i.values, wells.j.values, wells.k.values]
            miss = wells[(wells.sand.values == 1) & (~np.isin(cur, list(SAND)))]
            if len(miss):
                r_ = miss.iloc[rng.integers(len(miss))]
                tgt = (int(r_.i), int(r_.j), int(r_.k))
        kb = tgt[2] if tgt else int(rng.choice(NZ, p=pk))
        if code in (CHAN, ABAND):
            xs, ys = channel(nx, ny, sinuosity=1.0 if code == CHAN else .7)
            if tgt:
                d_ = np.hypot(xs - tgt[0], ys - tgt[1])
                q = int(np.argmin(d_))
                xs = xs + (tgt[0] - xs[q]); ys = ys + (tgt[1] - ys[q])
            w = rng.uniform(400, 950) / CELL
            t = max(1, int(round(rng.uniform(4, 9) / DZ)))
            cells = paint_channel(vol, code, xs, ys, kb, w, t, None)
        elif code == BAR:
            xs, ys = channel(nx, ny)
            f = rng.uniform(.35, .8)
            idx = int(len(xs) * f)
            cx, cy = xs[idx], ys[idx]
            if tgt:
                cx, cy = tgt[0] + rng.normal(0, 2), tgt[1] + rng.normal(0, 2)
            elif not (0 <= cx < nx and 0 <= cy < ny):
                cx, cy = rng.uniform(0, nx), rng.uniform(0, ny)
            cells = paint_lens(vol, cx, cy, kb,
                               rng.uniform(1200, 2600) / CELL,
                               rng.uniform(700, 1600) / CELL,
                               max(1, int(round(rng.uniform(5, 12) / DZ))),
                               AZIMUTH + rng.normal(0, .5))
        else:  # SPLAY
            cells = paint_lens(vol, rng.uniform(0, nx), rng.uniform(0, ny), kb,
                               rng.uniform(400, 900) / CELL,
                               rng.uniform(300, 700) / CELL,
                               max(1, int(round(rng.uniform(2, 4) / DZ))),
                               rng.uniform(0, 6.28))
        if not cells:
            continue
        # пробная запись
        backup = []
        n_new = 0
        for a_, b_, k0, k1 in cells:
            backup.append((a_, b_, k0, k1, vol[a_, b_, k0:k1].copy()))
            n_new += (k1 - k0)
            vol[a_, b_, k0:k1] = code
        mm2 = mismatch(vol, wells)
        if mm2 <= mm + tol:
            mm = mm2
            # считаем фактические доли: тела перекрывают друг друга,
            # поэтому «сколько нарисовали» и «сколько осталось видно» — разное
            cnt = np.bincount(vol.ravel(), minlength=7)
            for c in have:
                have[c] = int(cnt[c])
        else:
            for a_, b_, k0, k1, old in backup:
                vol[a_, b_, k0:k1] = old
    return vol, gx, gy, wells, wp, mm


if __name__ == '__main__':
    m = np.load('model_real.npz')
    vpc = m['vpc']
    vol, gx, gy, wells, wp, mm = build(vpc)
    total = vol.size
    print(f'сетка {vol.shape[0]} x {vol.shape[1]} x {vol.shape[2]}, '
          f'ячейка {CELL:.0f} x {CELL:.0f} x {DZ:.0f} м')
    print(f'площадь {(gx[-1]-gx[0])/1000:.1f} x {(gy[-1]-gy[0])/1000:.1f} км, '
          f'интервал {TOP:.0f}-{TOP+THICK:.0f} м\n')
    print(f'{"фация":<32}{"доля в модели":>15}{"цель":>8}')
    for c in sorted(NAMES):
        print(f'  {NAMES[c]:<30}{(vol == c).mean():>14.0%}{TARGET[c]:>8.0%}')
    sand = np.isin(vol, list(SAND)).mean()
    print(f'\nдоля коллектора (Рк+Уб): {sand:.0%}  (по скважинам {wells.sand.mean():.0%})')
    print(f'несовпадений со скважинами: {mm} из {len(wells)} '
          f'({1 - mm / len(wells):.0%} совпало)')
    np.savez_compressed('objmodel.npz', vol=vol, gx=gx, gy=gy, TOP=TOP, DZ=DZ, CELL=CELL)
    wells.to_csv('obj_wells.csv', index=False)
    wp.to_csv('obj_wellpos.csv')
    print('\nсохранено objmodel.npz')
