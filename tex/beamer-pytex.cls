\ProvidesClass{beamer-pytex}[2025/01/03 Custom Beamer class with PythonTeX support]

% Capture all class options and pass them to the beamer class
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{beamer}%
}
\ProcessOptions\relax
\LoadClass{beamer}

\RequirePackage{pythontex}
\makepythontexfamily[pyglexer=julia1]{julia}
\makepythontexfamily[pyglexer=julia1]{jl}
\makepythontexfamily@con[jlcon]{julia}
\setpythontexpygopt{style=algfordmdark}

\begin{jlcode}
	include("../../jl/support_code.jl")
	cur_plot = nothing
	function plot_pgfplots(args...; kwags...)
		global cur_plot
		cur_plot = PGFPlots.plot(args...; kwags...)
	end
\end{jlcode}
\RequirePackage{newfile}
\newoutputstream{juliaplot}
\DeclareRobustCommand{\plot}[1]{
	\IfFileExists{#1.tex}{}{
	\immediate\openoutputfile{#1.tex}{juliaplot}
	\immediate\closeoutputstream{juliaplot}
	}
	\jlc{save("#1.tex", cur_plot, include_preamble=false)}\input{#1}}
\DeclareRobustCommand{\plotlater}[1]{
	\IfFileExists{#1.tex}{}{
	\immediate\openoutputfile{#1.tex}{juliaplot}
	\immediate\closeoutputstream{juliaplot}
	}
	\jlc{save("#1.tex", cur_plot, include_preamble=false)}}
\jlc{plot(args...; kwags...) = plot_pgfplots(args...; kwags...)}
