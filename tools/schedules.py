
def parse_linear(spec, total_steps):
    assert spec.startswith('linear:')
    body = spec[len('linear:'):]
    lhs, rhs = body.split('@')
    a, b = lhs.split('->')
    a = float(a); b = float(b)
    t = float(rhs)
    def f(step):
        frac = step / max(1, total_steps)
        if frac >= t: return int(b)
        return int(a + (b - a) * (frac / t))
    return f

def parse_piecewise(spec, total_steps):
    assert spec.startswith('step:')
    pieces = spec[len('step:'):].split(';')
    segs = []
    for seg in pieces:
        rng, at = seg.split('@')
        a, b = rng.split('->')
        segs.append((int(a), int(b), float(at)))
    def f(step):
        frac = step / max(1, total_steps)
        cur = segs[0][0]
        for (a,b,t) in segs:
            if frac <= t: return int(b)
            cur = b
        return int(segs[-1][1])
    return f
