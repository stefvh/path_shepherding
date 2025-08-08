import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def find_n_seq(ll, n):
    row_ids = []
    for i, r in enumerate(ll):
        window = list(ll[i:i+n])
        rg = list(range(r, r+n))
        if len(window) < n:
            break
        if window == rg:
            row_ids.append(r)
    return row_ids
