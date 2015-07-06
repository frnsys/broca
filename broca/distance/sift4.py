"""
Sift4
-----

As described in:

    <http://siderite.blogspot.com/2014/11/super-fast-and-accurate-string-distance.html>

Sift4 is an approximation of Levenshtein distance,
with O(n) complexity (whereas Levensthein is O(n*m)).
"""


def sift4(s1, s2, max_offset=5):
    """
    This is an implementation of general Sift4.
    """
    t1, t2 = list(s1), list(s2)
    l1, l2 = len(t1), len(t2)

    if not s1:
        return l2

    if not s2:
        return l1

    # Cursors for each string
    c1, c2 = 0, 0

    # Largest common subsequence
    lcss = 0

    # Local common substring
    local_cs = 0

    # Number of transpositions ('ab' vs 'ba')
    trans = 0

    # Offset pair array, for computing the transpositions
    offsets = []

    while c1 < l1 and c2 < l2:
        if t1[c1] == t2[c2]:
            local_cs += 1

            # Check if current match is a transposition
            is_trans = False
            i = 0
            while i < len(offsets):
                ofs = offsets[i]
                if c1 <= ofs['c1'] or c2 <= ofs['c2']:
                    is_trans = abs(c2-c1) >= abs(ofs['c2'] - ofs['c1'])
                    if is_trans:
                        trans += 1
                    elif not ofs['trans']:
                        ofs['trans'] = True
                        trans += 1
                    break
                elif c1 > ofs['c2'] and c2 > ofs['c1']:
                    del offsets[i]
                else:
                    i += 1
            offsets.append({
                'c1': c1,
                'c2': c2,
                'trans': is_trans
            })

        else:
            lcss += local_cs
            local_cs = 0
            if c1 != c2:
                c1 = c2 = min(c1, c2)

            for i in range(max_offset):
                if c1 + i >= l1 and c2 + i >= l2:
                    break
                elif c1 + i < l1 and s1[c1+i] == s2[c2]:
                    c1 += i - 1
                    c2 -= 1
                    break

                elif c2 + i < l2 and s1[c1] == s2[c2 + i]:
                    c2 += i - 1
                    c1 -= 1
                    break

        c1 += 1
        c2 += 1

        if c1 >= l1 or c2 >= l2:
            lcss += local_cs
            local_cs = 0
            c1 = c2 = min(c1, c2)

    lcss += local_cs
    return round(max(l1, l2) - lcss + trans)
