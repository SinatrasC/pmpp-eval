# Chapter 13: Radix Sort - Multi-radix Single Turn

**Task**: Implement a multi-radix LSD radix sort that processes multiple bits per pass. This optimization reduces the number of passes required compared to 1-bit radix sort, improving performance through better memory bandwidth utilization.

## Algorithm Overview

**Multi-radix LSD Radix Sort**:
1. Choose radix size (e.g., 2-bit = 4 buckets, 4-bit = 16 buckets)
2. For each radix-sized chunk (fewer passes than 1-bit):
   - Count elements for each bucket (2^radix buckets)
   - Compute prefix sums to determine output positions
   - Scatter elements based on radix value to appropriate positions
   - Copy result back to input array for next iteration

**Stability**: Must be stable - equal elements maintain relative order from original array.

## Key Learning Objectives

- Understand radix optimization through wider bit processing
- Implement parallel multi-bucket counting/histogramming
- Apply parallel prefix sum for multi-bucket position calculation
- Practice efficient memory access patterns with larger bucket counts
- Learn performance trade-offs between passes and bucket complexity

## Implementation Details

- **Input**: Unsorted array of 32-bit unsigned integers
- **Output**: Sorted array (ascending order)
- **Radix Size**: 2-bit (4 buckets) or 4-bit (16 buckets) - your choice
- **Passes**: 32/radix_bits passes total (e.g., 16 passes for 2-bit, 8 passes for 4-bit)
- **Memory**: Use auxiliary buffer for scatter operations
- **Optimization**: Use shared memory for efficient bucket counting/scanning

## Test Coverage

The test harness validates:
- Correctness against CPU reference sort
- Stability preservation for equal elements
- Multiple input patterns (random, ascending, descending, duplicates)
- Various array sizes (small to large)
- Edge cases (empty arrays, single elements)
- Memory boundary checking with guard canaries

Success requires all test cases to pass with bit-perfect output matching the stable CPU reference implementation.