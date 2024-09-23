from shaderbang.input import TouchSlot

def homothety_and_rotation(slots: list[TouchSlot], center: tuple[float, float]) -> (float, float, float, float):
    a2 = b2 = 0.0
    ad = bc = ac = bd = 0.0
    cx, cy = center
    for slot in slots:
        a = slot.prevX - cx
        b = slot.prevY - cy
        c = slot.touchX - cx
        d = slot.touchY - cy
        a2 += a * a
        b2 += b * b
        ad += a * d
        bc += b * c
        ac += a * c
        bd += b * d

    det = a2 + b2
    if abs(det) < 1e-8:
        return 1.0, 0.0, 0.0, 0.0

    s = (ac + bd) / det
    r = (-bc + ad) / det
    tx = cy * r - cx * s + cx
    ty = -cx * r - cy * s + cy

    return s, r, tx, ty
