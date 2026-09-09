import math


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name):
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean
