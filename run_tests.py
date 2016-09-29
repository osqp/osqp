#!/usr/bin/env python

# Test QP solver against Maros Mezaros Benchmark suite

import scipy.io as spio


def main:
    for file in os.listdir('tests/maros_meszaros'):

        # Do all the tests






# Parsing optional command line arguments
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1:])
