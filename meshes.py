"""Поверхности по каждой фации для трёхмерного просмотра.

Индикатор каждой фации слегка сглаживается — это убирает ступенчатость ячеек
и заметно уменьшает число треугольников, тела становятся похожи на тела.
Координаты пакуются в целые числа и base64, иначе файл весит десятки мегабайт.
"""
import numpy as np, pandas as pd, json, base64
from skimage import measure
from scipy.ndimage import gaussian_filter
from objmodel import NAMES, SAND

SIGMA, STEP = 1.2, 2

d = np.load('objmodel.npz')
vol, gx, gy = d['vol'], d['gx'], d['gy']
TOP, DZ, CELL = float(d['TOP']), float(d['DZ']), float(d['CELL'])
nx, ny, nz = vol.shape
wp = pd.read_csv('obj_wellpos.csv', index_col=0)
wells = pd.read_csv('obj_wells.csv')

COL = {1: '#8B6F47', 2: '#C39B6A', 3: '#D3E8A0', 4: '#A8E05F',
       5: '#FFA500', 6: '#FFF9A8'}
SCALE = 16.0            # вершины храним в 1/16 ячейки


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


X0, Y0 = float(gx[0]), float(gy[0])
out = {'facies': [], 'wells': [], 'meta': {}}

for code in sorted(NAMES):
    f = gaussian_filter((vol == code).astype(np.float32), SIGMA)
    f = np.pad(f, 1, mode='constant')
    try:
        verts, faces, _, _ = measure.marching_cubes(f, .5, step_size=STEP)
    except (ValueError, RuntimeError):
        continue
    verts -= 1.0
    assert len(verts) < 65535, f'{NAMES[code]}: слишком много вершин'
    vi = np.round(verts * SCALE).astype(np.int16)
    out['facies'].append({
        'code': int(code), 'name': NAMES[code], 'color': COL[code],
        'share': round(float((vol == code).mean()), 4),
        'sand': bool(code in SAND),
        'nv': int(len(verts)), 'nf': int(len(faces)),
        'v': b64(vi), 'f': b64(faces.astype(np.uint16)),
    })
    print(f'{NAMES[code]:<34} вершин {len(verts):6d}  треугольников {len(faces):6d}')

for w, r in wp.iterrows():
    i = int(np.clip(round((r.X - X0) / CELL), 0, nx - 1))
    j = int(np.clip(round((r.Y - Y0) / CELL), 0, ny - 1))
    sub = wells[(wells.i == i) & (wells.j == j)].sort_values('k')
    col = vol[i, j, :]
    out['wells'].append({
        'name': str(w).replace('34_', ''), 'x': float(r.X), 'y': float(r.Y),
        'zs': (TOP + (sub.k.values + .5) * DZ).round(1).tolist(),
        'sand': sub.sand.astype(int).tolist(),
        'model': col.astype(int).tolist(),
    })

# объём после того же сглаживания — по нему строятся «крышки» на срезе,
# чтобы при обрезке по глубине тела выглядели телами, а не пустыми оболочками
codes = sorted(NAMES)
stack = np.stack([gaussian_filter((vol == c).astype(np.float32), SIGMA) for c in codes])
volS = np.array(codes, dtype=np.uint8)[stack.argmax(0)]
out['volS'] = b64(volS.astype(np.uint8))

out['meta'] = dict(nx=nx, ny=ny, nz=nz, cell=CELL, dz=DZ, top=TOP,
                   x0=X0, y0=Y0, x1=float(gx[-1]), y1=float(gy[-1]),
                   z1=TOP + nz * DZ, scale=SCALE,
                   sand_share=round(float(np.isin(vol, list(SAND)).mean()), 4),
                   well_match=None)

s = json.dumps(out, separators=(',', ':'), ensure_ascii=False)
open('facies3d.json', 'w').write(s)
print(f'\nfacies3d.json: {len(s)/1e6:.2f} МБ, '
      f'треугольников всего {sum(f["nf"] for f in out["facies"]):,}')
