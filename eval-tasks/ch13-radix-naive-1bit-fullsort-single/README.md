# Chapter 13: Radix Sort - Naive 1-bit Single Turn

**Task**: Implement a naive 1-bit radix sort using LSD (Least Significant Digit) approach. This foundational radix sort processes one bit at a time, building up the complete sort through repeated partitioning.

## Algorithm Overview

**LSD Radix Sort (1-bit)**:
1. For each bit position (0 to 31 for 32-bit integers):
   - Count elements with bit=0 and bit=1
   - Compute prefix sums to determine output positions
   - Scatter elements based on bit value to appropriate positions
   - Copy result back to input array for next iteration

**Stability**: Must be stable - equal elements maintain relative order from original array.

## Key Learning Objectives

- Understand radix sort fundamentals and bit-level processing
- Implement parallel counting/histogramming
- Apply parallel prefix sum (scan) for position calculation
- Practice stable scattering with proper indexing
- Learn multi-pass sorting algorithm coordination

## Implementation Details

- **Input**: Unsorted array of 32-bit unsigned integers
- **Output**: Sorted array (ascending order)
- **Approach**: Process 1 bit per pass, 32 passes total
- **Memory**: Use auxiliary buffer for scatter operations
- **Optimization**: Use shared memory for local counting/scanning

## Test Coverage

The test harness validates:
- Correctness against CPU reference sort
- Stability preservation for equal elements
- Multiple input patterns (random, ascending, descending, duplicates)
- Various array sizes (small to large)
- Edge cases (empty arrays, single elements)
- Memory boundary checking with guard canaries

Success requires all test cases to pass with bit-perfect output matching the stable CPU reference implementation.