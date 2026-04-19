import math

def euclidian_distance(p1, p2):
    tot = 0.0
    for x1, x2 in zip(p1, p2):
        tot += (x1 - x2) ** 2
    return math.sqrt(tot)

def manhattan_distance(p1, p2):
    tot = 0.0
    for x1, x2 zip(p1, p2):
        tot += abs(x1 - x2)
    return tot