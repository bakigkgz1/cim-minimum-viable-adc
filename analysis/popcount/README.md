# Popcount cost probe

`popcount.cpp` costs the digital popcount that replaces the analog unit column with NeuroSim's own
`AdderTree` model (PR one-bit inputs) at the technology node set in `NeuroSIM/Param.cpp`.

Build from the NeuroSim `NeuroSIM/` directory at commit 9825ef4 (not redistributed here):

    g++ -O2 -std=c++11 -w -I. /path/to/popcount.cpp $(ls *.cpp | grep -v main.cpp) -o popcount && ./popcount

Output used in the manuscript (22 nm, 1 GHz): PR 64 / 128 / 256 -> 0.13 / 0.27 / 0.55 pJ per operation,
167 / 343 / 698 µm², one clock cycle.
