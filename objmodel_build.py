"""Объектная модель фаций: тела ставятся в геологическом порядке.

Порядок важен: бары и разливы ложатся в фоновые отложения, заиливающиеся
каналы — следом, а активные русла врезаются во всё, что было раньше.
Поэтому ранние тела ставятся с запасом — часть их срежут русла.
"""
import numpy as np, pandas as pd
from objmodel import (BG_SWAMP, BAY, SPLAY, ABAND, CHAN, BAR, NAMES, SAND,
                      TARGET, CELL, DZ, TOP, THICK, NZ, AZIMUTH,
                      grid, well_arrays, channel, paint_channel, paint_lens,
                      mismatch)

rng = np.random.default_rng(11)
import objmodel
objmodel.rng = rng


def place(vol, wells, code, need_frac, allow_cut, p_target=.6, maxtry=6000):
    nx, ny, nz = vol.shape
    total = vol.size
    tries = 0
    while (vol == code).sum() < need_frac * total and tries < maxtry:
        tries += 1
        tgt = None
        if rng.random() < p_target:
            cur = vol[wells.i.values, wells.j.values, wells.k.values]
            if code in SAND:
                miss = wells[(wells.sand.values == 1) & (~np.isin(cur, list(SAND)))]
            else:
                miss = wells[(wells.sand.values == 0) & (np.isin(cur, list(SAND)))]
            if len(miss):
                r_ = miss.iloc[rng.integers(len(miss))]
                tgt = (int(r_.i), int(r_.j), int(r_.k))
        kb = tgt[2] if tgt else int(rng.integers(nz))
        if code in (CHAN, ABAND):
            xs, ys = channel(nx, ny, 1.0 if code == CHAN else .6)
            if tgt:
                q = int(np.argmin(np.hypot(xs - tgt[0], ys - tgt[1])))
                xs = xs + (tgt[0] - xs[q]); ys = ys + (tgt[1] - ys[q])
            cells = paint_channel(vol, code, xs, ys, kb,
                                  rng.uniform(350, 900) / CELL,
                                  max(1, int(round(rng.uniform(4, 9) / DZ))), None)
        elif code == BAR:
            cx = tgt[0] + rng.normal(0, 3) if tgt else rng.uniform(0, nx)
            cy = tgt[1] + rng.normal(0, 3) if tgt else rng.uniform(0, ny)
            cells = paint_lens(vol, cx, cy, kb,
                               rng.uniform(1200, 2600) / CELL,
                               rng.uniform(700, 1600) / CELL,
                               max(1, int(round(rng.uniform(5, 12) / DZ))),
                               AZIMUTH + rng.normal(0, .5))
        else:
            cells = paint_lens(vol, rng.uniform(0, nx), rng.uniform(0, ny), kb,
                               rng.uniform(400, 900) / CELL,
                               rng.uniform(300, 700) / CELL,
                               max(1, int(round(rng.uniform(2, 4) / DZ))),
                               rng.uniform(0, 6.28))
        if not cells:
            continue
        mm0 = mismatch(vol, wells)
        backup = []
        for a_, b_, k0, k1 in cells:
            seg = vol[a_, b_, k0:k1]
            m = np.isin(seg, allow_cut)
            if not m.any():
                continue
            backup.append((a_, b_, k0, k1, seg.copy()))
            seg[m] = code
        if not backup:
            continue
        if mismatch(vol, wells) > mm0:            # тело ухудшило привязку — снимаем
            for a_, b_, k0, k1, old in backup:
                vol[a_, b_, k0:k1] = old
    return tries


def build():
    gx, gy, wd, wp = grid()
    nx, ny = len(gx), len(gy)
    wells = well_arrays(wd, wp, gx, gy)
    vpc = np.load('model_real.npz')['vpc']
    vp = np.interp(np.arange(NZ), np.linspace(0, NZ - 1, len(vpc)), vpc)

    vol = np.full((nx, ny, NZ), BAY, dtype=np.uint8)
    swamp = np.arange(NZ)[vp < np.quantile(vp, .35)]
    vol[:, :, swamp] = BG_SWAMP
    BGS = [BG_SWAMP, BAY]

    print('ставим тела:')
    for code, frac, cut, pt in [
            (BAR,   TARGET[BAR] * 1.45,   BGS, .60),
            (SPLAY, TARGET[SPLAY] * 1.25, BGS, .15),
            (ABAND, TARGET[ABAND] * 1.30, BGS, .15),
            (CHAN,  TARGET[CHAN],         BGS + [SPLAY, ABAND, BAR], .60)]:
        t = place(vol, wells, code, frac, cut, pt)
        print(f'  {NAMES[code]:<32} попыток {t:5d}   доля {(vol == code).mean():.0%}')
    return vol, gx, gy, wells, wp


if __name__ == '__main__':
    vol, gx, gy, wells, wp = build()
    print(f'\nсетка {vol.shape[0]} x {vol.shape[1]} x {vol.shape[2]}, '
          f'ячейка {CELL:.0f} x {CELL:.0f} x {DZ:.0f} м, '
          f'площадь {(gx[-1]-gx[0])/1000:.1f} x {(gy[-1]-gy[0])/1000:.1f} км\n')
    print(f'{"фация":<34}{"в модели":>10}{"цель":>8}')
    for c in sorted(NAMES):
        print(f'  {NAMES[c]:<32}{(vol == c).mean():>9.0%}{TARGET[c]:>8.0%}')
    mm = mismatch(vol, wells)
    print(f'\nдоля коллектора: {np.isin(vol, list(SAND)).mean():.0%} '
          f'(по скважинам {wells.sand.mean():.0%})')
    print(f'совпадение со скважинами: {1 - mm/len(wells):.0%} ({len(wells)-mm} из {len(wells)})')
    np.savez_compressed('objmodel.npz', vol=vol, gx=gx, gy=gy,
                        TOP=TOP, DZ=DZ, CELL=CELL)
    wells.to_csv('obj_wells.csv', index=False); wp.to_csv('obj_wellpos.csv')
    print('сохранено objmodel.npz')
