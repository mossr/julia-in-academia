# Julia in Academia Slides

**JuliaCon 2025**: _Julia in Academia: Textbooks, Stanford Courses, and the Future_.

Robert Moss (_Stanford University_)

See the abstract for the talk [here](https://pretalx.com/juliacon-2025/talk/YT7AVS/) and slides PDF [here](./output/main.pdf).

<p align="center">
  <kbd>
    <a href="./output/main.pdf">
      <img src="./media/algforval-juliacon.png">
    </a>
  </kbd>
</p>

## Resources

- Textbook template: [sisl/tufte_algorithms_book](https://github.com/sisl/tufte_algorithms_book)
- Slides template: [mossr/julia-tufte-beamer](https://github.com/mossr/julia-tufte-beamer)
- Interactive papers (`PlutoPapers.jl`): [mossr/PlutoPapers.jl](https://github.com/mossr/PlutoPapers.jl)
- "How We Wrote a Textbook using Julia" (Tim Wheeler, JuliaCon 2019): https://youtu.be/ofWy5kaZU3g
- PDFs of Julia-based textbooks: https://algorithmsbook.com
- Stanford's _Validation of Safety-Critical Systems_ lecture videos: [youtube-playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOq1LMLI8U7djzDb8--xpaC)
- Stanford AA228V course material:
    - Core library: [sisl/StanfordAA228V.jl](https://github.com/sisl/StanfordAA228V.jl)
    - Pluto notebooks: [sisl/AA228VProjects](https://github.com/sisl/AA228VProjects)
    - Lecture notebooks: [sisl/AA228VLectureNotebooks](https://github.com/sisl/AA228VLectureNotebooks)
    - Gradescope.jl: [sisl/Gradescope.jl](https://github.com/sisl/Gradescope.jl)
    - Course-specific Gradescope.jl implementation: [sisl/AA228VGradescope.jl](https://github.com/sisl/AA228VGradescope.jl)

## Installation

1. Clone repo and `cd` to the folder.
1. Install lexer and style:
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

This is an updated fork of [mossr/julia-tufte-beamer](https://github.com/mossr/julia-tufte-beamer).
