# ch13-radix-onepass-multiradix-single

Implement one *stable* multiradix pass: extract r-bit digit (at `shift`), build per-block histograms, do a host-side global exclusive scan, then scatter stably to `out`.

- Keys: `uint32_t`, ascending order expected after composing passes externally
- Stability: required (equal digits preserve input order)
- Bits per pass: r ∈ {1,2,4}, shift provided
- No dependencies; fully self-contained

## Build & Run

```
make
./test_reference
./test_student
```

## Pass/Fail
- Student must match CPU one-pass output exactly for all test sizes, bit-widths, and shifts
- Inputs must remain unchanged (immutability check)
- Guard canaries must remain intact (no OOB)