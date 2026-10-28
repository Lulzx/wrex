// Hand-written sequential baselines for the Appendix B loops. Same input and
// output format as bench.c.
use std::hint::black_box;
use std::time::Instant;

const REPS: usize = 200;

fn best<T>(mut f: impl FnMut() -> T) -> (u128, T) {
    let mut best = u128::MAX;
    let mut out = None;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let r = black_box(f());
        best = best.min(t0.elapsed().as_nanos());
        out = Some(r);
    }
    (best, out.unwrap())
}

fn report_scan(name: &str, (ns, out): (u128, Vec<i64>)) {
    println!("{} {} {}:{}", name, ns, out.iter().sum::<i64>(), out[out.len() - 1]);
}

fn main() {
    let data = std::fs::read_to_string("data.txt").unwrap();
    let mut it = data.split_ascii_whitespace().map(|s| s.parse::<i64>().unwrap());
    let n = it.next().unwrap() as usize;
    let xs: Vec<i64> = it.by_ref().take(n).collect();
    let text: Vec<u8> = it.take(n).map(|c| c as u8).collect();
    let txs: Vec<(i64, i64)> = xs.iter().map(|&x| (1, x)).collect();

    report_scan("prefix_sum", best(|| {
        let mut s = 0;
        xs.iter().map(|&x| { s += x; s }).collect::<Vec<i64>>()
    }));
    report_scan("linear_recurrence", best(|| {
        let mut b = 0;
        txs.iter().map(|&(rate, dep)| { b = b * rate + dep; b }).collect::<Vec<i64>>()
    }));
    report_scan("saturating_counters", best(|| {
        let mut s = 0;
        xs.iter().map(|&d| { s = (s + d).clamp(0, 50); s }).collect::<Vec<i64>>()
    }));
    let (ns, c) = best(|| {
        let (mut in_word, mut c) = (false, 0i64);
        for &ch in &text {
            let alpha = ch.is_ascii_alphabetic();
            if alpha && !in_word { c += 1; }
            in_word = alpha;
        }
        c
    });
    println!("tokenizer {} {}", ns, c);
    let (ns, (a, b)) = best(|| {
        xs.iter().fold((0i64, 0i64), |(a, b), &x| ((a + x).max(b - x), (a - x).max(b + x)))
    });
    println!("viterbi {} {}:{}", ns, a, b);
}
