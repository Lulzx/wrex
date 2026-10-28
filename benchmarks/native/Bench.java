// Hand-written sequential baselines for the Appendix B loops. Same input and
// output format as bench.c. Each loop runs REPS times, so the JIT has compiled
// it well before the fastest run.
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Supplier;

public class Bench {
    static final int REPS = 200;
    static long ns;

    static <T> T best(Supplier<T> f) {
        long b = Long.MAX_VALUE;
        T out = null;
        for (int r = 0; r < REPS; r++) {
            long t0 = System.nanoTime();
            out = f.get();
            b = Math.min(b, System.nanoTime() - t0);
        }
        ns = b;
        return out;
    }

    static String sumLast(long[] o) {
        long s = 0;
        for (long v : o) s += v;
        return s + ":" + o[o.length - 1];
    }

    public static void main(String[] args) throws Exception {
        String[] f = Files.readString(Path.of("data.txt")).trim().split("\\s+");
        int n = Integer.parseInt(f[0]);
        long[] xs = new long[n];
        byte[] text = new byte[n];
        long[] rate = new long[n];
        for (int i = 0; i < n; i++) {
            xs[i] = Long.parseLong(f[1 + i]);
            text[i] = (byte) Integer.parseInt(f[1 + n + i]);
            rate[i] = 1;
        }

        long[] o = best(() -> {
            long[] out = new long[n];
            long s = 0;
            for (int i = 0; i < n; i++) { s += xs[i]; out[i] = s; }
            return out;
        });
        System.out.println("prefix_sum " + ns + " " + sumLast(o));

        o = best(() -> {
            long[] out = new long[n];
            long b = 0;
            for (int i = 0; i < n; i++) { b = b * rate[i] + xs[i]; out[i] = b; }
            return out;
        });
        System.out.println("linear_recurrence " + ns + " " + sumLast(o));

        o = best(() -> {
            long[] out = new long[n];
            long s = 0;
            for (int i = 0; i < n; i++) { s = Math.clamp(s + xs[i], 0, 50); out[i] = s; }
            return out;
        });
        System.out.println("saturating_counters " + ns + " " + sumLast(o));

        long c = best(() -> {
            boolean inWord = false;
            long count = 0;
            for (byte ch : text) {
                boolean alpha = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
                if (alpha && !inWord) count++;
                inWord = alpha;
            }
            return count;
        });
        System.out.println("tokenizer " + ns + " " + c);

        long[] ab = best(() -> {
            long a = 0, b = 0;
            for (long x : xs) {
                long na = Math.max(a + x, b - x), nb = Math.max(a - x, b + x);
                a = na; b = nb;
            }
            return new long[] {a, b};
        });
        System.out.println("viterbi " + ns + " " + ab[0] + ":" + ab[1]);
    }
}
