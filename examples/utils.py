def get_state_dict(x):
    R = x[:9].reshape((3,3))
    p = x[9:12]
    w = x[12:15]
    v = x[15:]
    return {
        'R': R,
        'p': p,
        'w': w,
        'v': v
    }