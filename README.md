# SignalTemporalLogic.jl Slides

AA228V/CS238V Stanford University.

Install lexer and style:
```
python -m venv stl
source stl/bin/activate
pip install --upgrade git+https://github.com/sisl/pygments-julia#egg=pygments_julia
pip install --upgrade -e pygments-style-algfordmdark
```

- Make sure [`pythontex`](https://github.com/gpoore/pythontex) is installed.
- Make sure `julia` is installed.

To compile, run:
```
latexmk
```