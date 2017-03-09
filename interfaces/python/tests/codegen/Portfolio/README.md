Solve a sequence of Portfolio problems in C for varying risk aversion parameter gamma:
(very hacky now)
TODO: Automate this process.

1) Run
    python portfolio.py
   It generates code directory and portfolio_data.txt.

2) Create code/build directory and copy portfolio_data.txt there.

3) Copy portfolio_sparse.c in code/src directory.

4) Replace "example" and "example.c" in the last line of CMakeLists (in code directory)
   by "portfolio_sparse" and "portfolio_sparse.c"

5) Run cmake command from code/build directory
    cmake ..                        (Unix)
    cmake -G "MinGW Makefiles" ..   (Windows)

6) Run
    make                            (Unix)
    cmake --build .                  (Windows)

7) Run
    out/portfolio_sparse
