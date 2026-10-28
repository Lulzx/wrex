// Hand-written sequential baselines for the Appendix B loops. Same input and
// output format as bench.c.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const reps = 200

func best(f func()) int64 {
	b := int64(-1)
	for r := 0; r < reps; r++ {
		t0 := time.Now()
		f()
		if dt := time.Since(t0).Nanoseconds(); b < 0 || dt < b {
			b = dt
		}
	}
	return b
}

func sumLast(o []int64) string {
	var s int64
	for _, v := range o {
		s += v
	}
	return fmt.Sprintf("%d:%d", s, o[len(o)-1])
}

func main() {
	raw, err := os.ReadFile("data.txt")
	if err != nil {
		panic(err)
	}
	f := strings.Fields(string(raw))
	n, _ := strconv.Atoi(f[0])
	xs := make([]int64, n)
	text := make([]byte, n)
	for i := 0; i < n; i++ {
		v, _ := strconv.ParseInt(f[1+i], 10, 64)
		xs[i] = v
		c, _ := strconv.Atoi(f[1+n+i])
		text[i] = byte(c)
	}
	rate := make([]int64, n)
	for i := range rate {
		rate[i] = 1
	}

	var out []int64
	ns := best(func() {
		out = make([]int64, n)
		var s int64
		for i, x := range xs {
			s += x
			out[i] = s
		}
	})
	fmt.Println("prefix_sum", ns, sumLast(out))

	ns = best(func() {
		out = make([]int64, n)
		var b int64
		for i, d := range xs {
			b = b*rate[i] + d
			out[i] = b
		}
	})
	fmt.Println("linear_recurrence", ns, sumLast(out))

	ns = best(func() {
		out = make([]int64, n)
		var s int64
		for i, d := range xs {
			s = min(max(s+d, 0), 50)
			out[i] = s
		}
	})
	fmt.Println("saturating_counters", ns, sumLast(out))

	var count int64
	ns = best(func() {
		inWord := false
		count = 0
		for _, ch := range text {
			alpha := (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')
			if alpha && !inWord {
				count++
			}
			inWord = alpha
		}
	})
	fmt.Println("tokenizer", ns, count)

	var a, b int64
	ns = best(func() {
		a, b = 0, 0
		for _, x := range xs {
			a, b = max(a+x, b-x), max(a-x, b+x)
		}
	})
	fmt.Printf("viterbi %d %d:%d\n", ns, a, b)
}
