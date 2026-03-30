# RaBitQ C++ Implementation

C++ implementation of RaBitQ (Randomized Binary Quantization) for approximate nearest neighbor search, based on the [reference implementation](https://github.com/gaoj0017/RaBitQ).

## Build

```bash
cd src
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Test

```bash
ctest --test-dir build
# or directly:
./build/test_setup
```

## Dependencies

Fetched automatically via CMake FetchContent (no system install needed):

- **Eigen 3.4.0** — Matrix operations (QR decomposition, matrix multiply)
- **Google Test 1.14.0** — Unit testing

## Compiler Flags

- Release: `-Ofast -march=core-avx2`
- Debug: `-O0 -g -march=core-avx2 -fsanitize=address`

## File Structure

```
src/
├── CMakeLists.txt     # Build system
├── defines.hpp        # Common types and constants
├── README.md          # This file
├── tests/
│   └── test_setup.cpp # Build verification tests
└── build/             # Build output (gitignored)
```
