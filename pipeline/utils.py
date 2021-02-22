#! /usr/bin/env python3

import math

def get_all_pairs(labels):
    # binomial choice, poorly implemented -> O(N^2)
    pairs = set()
    for l1 in labels:
        for l2 in labels:
            if l1 == l2:
                continue
            pairs.add( (min(l1,l2), max(l1,l2)) )

    assert len(pairs) == math.comb(len(labels), 2)

    # sort tuples by sneakily turning them into 2 digit numbers
    pairs = sorted(list(pairs), key=lambda x: (10 * x[0]) + x[1])

    return pairs

if __name__ == '__main__':
    pairs = get_all_pairs([0,1,2,3,4,5,6,7,8,9])
    for pair in pairs:
        print(pair)
