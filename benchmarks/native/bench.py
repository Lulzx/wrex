"""Hand-written sequential baselines for the Appendix B loops in plain Python.
Same input and output format as bench.c."""
import time

REPS = 20


def best(f):
    b, out = None, None
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        out = f()
        dt = time.perf_counter_ns() - t0
        b = dt if b is None else min(b, dt)
    return b, out


def prefix_sum(xs):
    out, s = [], 0
    for x in xs:
        s += x
        out.append(s)
    return out


def balances(txs):
    out, b = [], 0
    for rate, dep in txs:
        b = b * rate + dep
        out.append(b)
    return out


def stock_levels(deltas, cap):
    out, s = [], 0
    for d in deltas:
        s = min(max(s + d, 0), cap)
        out.append(s)
    return out


def word_count(text):
    in_word, c = False, 0
    for ch in text:
        alpha = 65 <= ch <= 90 or 97 <= ch <= 122
        if alpha and not in_word:
            c += 1
        in_word = alpha
    return c


def viterbi(xs):
    a = b = 0
    for x in xs:
        a, b = max(a + x, b - x), max(a - x, b + x)
    return a, b


def main():
    vals = list(map(int, open("data.txt").read().split()))
    n = vals[0]
    xs, text = vals[1:1 + n], bytes(vals[1 + n:1 + 2 * n])
    txs = [(1, x) for x in xs]
    for name, f in (("prefix_sum", lambda: prefix_sum(xs)),
                    ("linear_recurrence", lambda: balances(txs)),
                    ("saturating_counters", lambda: stock_levels(xs, 50))):
        ns, o = best(f)
        print(name, ns, f"{sum(o)}:{o[-1]}")
    ns, c = best(lambda: word_count(text))
    print("tokenizer", ns, c)
    ns, (a, b) = best(lambda: viterbi(xs))
    print("viterbi", ns, f"{a}:{b}")


if __name__ == "__main__":
    main()
