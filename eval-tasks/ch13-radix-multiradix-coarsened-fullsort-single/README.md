# Chapter 13: Radix Sort - Multi-radix Coarsened Single Turn

**Task**: Implement a multi-radix LSD radix sort with thread coarsening optimization. This advanced version combines multi-radix efficiency with thread coarsening to improve memory bandwidth utilization and reduce overhead from small work-per-thread scenarios.

## Algorithm Overview

**Multi-radix Coarsened LSD Radix Sort**:
1. Choose radix size (e.g., 2-bit = 4 buckets, 4-bit = 16 buckets)
2. Apply thread coarsening - each thread processes multiple elements per pass
3. For each radix-sized chunk (fewer passes than 1-bit):
   - Count elements for each bucket using coarsened threads
   - Compute prefix sums to determine output positions
   - Scatter elements based on radix value to appropriate positions
   - Copy result back to input array for next iteration

**Stability**: Must be stable - equal elements maintain relative order from original array.

## Key Learning Objectives

- Understand thread coarsening optimization techniques
- Implement efficient work distribution with coarsened threads
- Apply memory coalescing with coarsened access patterns
- Practice load balancing with variable work per thread
- Learn advanced CUDA optimization combining multiple techniques

## Implementation Details

- **Input**: Unsorted array of 32-bit unsigned integers
- **Output**: Sorted array (ascending order)
- **Radix Size**: 2-bit (4 buckets) or 4-bit (16 buckets) - your choice
- **Coarsening Factor**: Each thread processes N elements (e.g., 4-8 elements per thread)
- **Passes**: 32/radix_bits passes total
- **Memory**: Use auxiliary buffer for scatter operations
- **Optimization**: Use shared memory + coarsened memory access patterns

## Test Coverage

The test harness validates:
- Correctness against CPU reference sort
- Stability preservation for equal elements
- Multiple input patterns (random, ascending, descending, duplicates)
- Various array sizes (small to large)
- Edge cases (empty arrays, single elements)
- Memory boundary checking with guard canaries

Success requires all test cases to pass with bit-perfect output matching the stable CPU reference implementation.