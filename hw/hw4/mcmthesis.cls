%%
%% This is file `mcmthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% mcmthesis.dtx  (with options: `class')
%% !Mode:: "TeX:UTF-8"
%% -----------------------------------
%% 
%% This is a generated file.
%% 
%% Copyright (C)
%%     2010 -- 2015 by latexstudio
%%     2014 -- 2016 by Liam Huang
%% 
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%% 
%% This work has the LPPL maintenance status `maintained'.
%% 
%% The Current Maintainer of this work is Liam Huang.
%% 
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{mcmthesis}
  [2017/01/01 v6.2.1 The Thesis Template Designed For MCM/ICM]
\typeout{The Thesis Template Designed For MCM/ICM}
\def\MCMversion{v6.2.1}
\RequirePackage{xkeyval}
\RequirePackage{etoolbox}
\define@boolkey{MCM}[MCM@opt@]{CTeX}[false]{}
\define@boolkey{MCM}[MCM@opt@]{titlepage}[true]{}
\define@boolkey{MCM}[MCM@opt@]{abstract}[true]{}
\define@boolkey{MCM}[MCM@opt@]{sheet}[true]{}
\define@boolkey{MCM}[MCM@opt@]{titleinsheet}[false]{}
\define@boolkey{MCM}[MCM@opt@]{keywordsinsheet}[false]{}
\define@cmdkeys{MCM}[MCM@opt@]{tcn,problem}
\define@key{MCM}{tcn}[0000]{\gdef\MCM@opt@tcn{#1}}
\define@key{MCM}{problem}[A]{\gdef\MCM@opt@problem{#1}}
\setkeys{MCM}{tcn=0000,problem=B}

\define@key{mcmthesis.cls}{tcn}[0000]{\gdef\MCM@opt@tcn{#1}}
\define@key{mcmthesis.cls}{problem}[A]{\gdef\MCM@opt@problem{#1}}
\define@boolkey{mcmthesis.cls}[MCM@opt@]{titlepage}{}
\define@boolkey{mcmthesis.cls}[MCM@opt@]{abstract}{}
\define@boolkey{mcmthesis.cls}[MCM@opt@]{sheet}{}
\define@boolkey{mcmthesis.cls}[MCM@opt@]{titleinsheet}{}
\define@boolkey{mcmthesis.cls}[MCM@opt@]{keywordsinsheet}{}
\MCM@opt@sheettrue
\MCM@opt@titlepagetrue
\MCM@opt@titleinsheetfalse
\MCM@opt@keywordsinsheetfalse
\MCM@opt@abstracttrue
\newcommand{\mcmsetup}[1]{\setkeys{MCM}{#1}}
\ProcessOptionsX\relax
\LoadClass[a4paper, 12pt]{article}
\newcommand{\team}{Team \#\ {\color{Black}\MCM@opt@tcn} }
%\RequirePackage{fancyhdr, fancybox}
\RequirePackage{fancybox}
\RequirePackage{ifthen}
\RequirePackage{lastpage}
\RequirePackage{listings}
\RequirePackage[toc, page, title, titletoc, header]{appendix}
\RequirePackage{paralist}
\RequirePackage{amsthm, amsfonts}
\RequirePackage{amsmath, bm}
\RequirePackage{amssymb, mathrsfs}
\RequirePackage{latexsym}
\RequirePackage{longtable, multirow, hhline, tabularx, array}
\RequirePackage{flafter}
\RequirePackage{pifont, calc}
\RequirePackage{colortbl, booktabs}
\RequirePackage{geometry}
\RequirePackage[T1]{fontenc}
\RequirePackage[scaled]{berasans}
\RequirePackage{hyperref}
\RequirePackage{ifpdf, ifxetex}
%\RequirePackage[subfigure]{tocloft}

\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{xspace}

% common abbreviations

\providecommand{\eg}{\textit{e.g.}\@\xspace}
\providecommand{\ie}{\textit{i.e.}\@\xspace}
\providecommand{\wrt}{\textit{w.r.t.}\@\xspace}
\providecommand{\etal}{\textit{et al}\@\xspace}


% useful operators

\newcommand{\argmin}{\mathop{\mathrm{argmin}}}
\newcommand{\argmax}{\mathop{\mathrm{argmax}}}


% v? vectors:

\newcommand{\va}{\mathbf{a}}
\newcommand{\vb}{\mathbf{b}}
\newcommand{\vc}{\mathbf{c}}
\newcommand{\vd}{\mathbf{d}}
\newcommand{\ve}{\mathbf{e}}
\newcommand{\vf}{\mathbf{f}}
\newcommand{\vg}{\mathbf{g}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vi}{\mathbf{i}}
\newcommand{\vj}{\mathbf{j}}
\newcommand{\vk}{\mathbf{k}}
\newcommand{\vl}{\mathbf{l}}
\newcommand{\vm}{\mathbf{m}}
\newcommand{\vn}{\mathbf{n}}
\newcommand{\vo}{\mathbf{o}}
\newcommand{\vp}{\mathbf{p}}
\newcommand{\vq}{\mathbf{q}}
\newcommand{\vr}{\mathbf{r}}
\newcommand{\vs}{\mathbf{s}}
\newcommand{\vt}{\mathbf{t}}
\newcommand{\vu}{\mathbf{u}}
\newcommand{\vv}{\mathbf{v}}
\newcommand{\vw}{\mathbf{w}}
\newcommand{\vx}{\mathbf{x}}
\newcommand{\vy}{\mathbf{y}}
\newcommand{\vz}{\mathbf{z}}

\newcommand{\vone}{\mathbf{1}}
\newcommand{\vzero}{\mathbf{0}}

\newcommand{\valpha}   {\boldsymbol{\alpha}}
\newcommand{\vbeta}    {\boldsymbol{\beta}}
\newcommand{\vgamma}   {\boldsymbol{\gamma}}
\newcommand{\vdelta}   {\boldsymbol{\delta}}
\newcommand{\vepsilon} {\boldsymbol{\epsilon}}
\newcommand{\vzeta}    {\boldsymbol{\zeta}}
\newcommand{\veta}     {\boldsymbol{\eta}}
\newcommand{\vtheta}   {\boldsymbol{\theta}}
\newcommand{\viota}    {\boldsymbol{\iota}}
\newcommand{\vkappa}   {\boldsymbol{\kappa}}
\newcommand{\vlambda}  {\boldsymbol{\lambda}}
\newcommand{\vmu}      {\boldsymbol{\mu}}
\newcommand{\vnu}      {\boldsymbol{\nu}}
\newcommand{\vxi}      {\boldsymbol{\xi}}
\newcommand{\vpi}      {\boldsymbol{\pi}}
\newcommand{\vrho}     {\boldsymbol{\rho}}
\newcommand{\vsigma}   {\boldsymbol{\sigma}}
\newcommand{\vtau}     {\boldsymbol{\tau}}
\newcommand{\vupsilon} {\boldsymbol{\upsilon}}
\newcommand{\vphi}     {\boldsymbol{\phi}}
\newcommand{\vchi}     {\boldsymbol{\chi}}
\newcommand{\vpsi}     {\boldsymbol{\psi}}
\newcommand{\vomega}   {\boldsymbol{\omega}}


% m? matrices:

\newcommand{\mA}{\mathbf{A}}
\newcommand{\mB}{\mathbf{B}}
\newcommand{\mC}{\mathbf{C}}
\newcommand{\mD}{\mathbf{D}}
\newcommand{\mE}{\mathbf{E}}
\newcommand{\mF}{\mathbf{F}}
\newcommand{\mG}{\mathbf{G}}
\newcommand{\mH}{\mathbf{H}}
\newcommand{\mI}{\mathbf{I}}
\newcommand{\mJ}{\mathbf{J}}
\newcommand{\mK}{\mathbf{K}}
\newcommand{\mL}{\mathbf{L}}
\newcommand{\mM}{\mathbf{M}}
\newcommand{\mN}{\mathbf{N}}
\newcommand{\mO}{\mathbf{O}}
\newcommand{\mP}{\mathbf{P}}
\newcommand{\mQ}{\mathbf{Q}}
\newcommand{\mR}{\mathbf{R}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mT}{\mathbf{T}}
\newcommand{\mU}{\mathbf{U}}
\newcommand{\mV}{\mathbf{V}}
\newcommand{\mW}{\mathbf{W}}
\newcommand{\mX}{\mathbf{X}}
\newcommand{\mY}{\mathbf{Y}}
\newcommand{\mZ}{\mathbf{Z}}

\newcommand{\mGamma}{\boldsymbol{\Gamma}}
\newcommand{\mDelta}{\boldsymbol{\Delta}}
\newcommand{\mTheta}{\boldsymbol{\Theta}}
\newcommand{\mLambda}{\boldsymbol{\Lambda}}
\newcommand{\mXi}{\boldsymbol{\Xi}}
\newcommand{\mPi}{\boldsymbol{\Pi}}
\newcommand{\mSigma}{\boldsymbol{\Sigma}}
\newcommand{\mUpsilon}{\boldsymbol{\Upsilon}}
\newcommand{\mPhi}{\boldsymbol{\Phi}}
\newcommand{\mPsi}{\boldsymbol{\Psi}}
\newcommand{\mOmega}{\boldsymbol{\Omega}}


% c? calligraphic letters

\newcommand{\cA}{\mathcal{A}}
\newcommand{\cB}{\mathcal{B}}
\newcommand{\cC}{\mathcal{C}}
\newcommand{\cD}{\mathcal{D}}
\newcommand{\cE}{\mathcal{E}}
\newcommand{\cF}{\mathcal{F}}
\newcommand{\cG}{\mathcal{G}}
\newcommand{\cH}{\mathcal{H}}
\newcommand{\cI}{\mathcal{I}}
\newcommand{\cJ}{\mathcal{J}}
\newcommand{\cK}{\mathcal{K}}
\newcommand{\cL}{\mathcal{L}}
\newcommand{\cM}{\mathcal{M}}
\newcommand{\cN}{\mathcal{N}}
\newcommand{\cO}{\mathcal{O}}
\newcommand{\cP}{\mathcal{P}}
\newcommand{\cQ}{\mathcal{Q}}
\newcommand{\cR}{\mathcal{R}}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\cT}{\mathcal{T}}
\newcommand{\cU}{\mathcal{U}}
\newcommand{\cV}{\mathcal{V}}
\newcommand{\cW}{\mathcal{W}}
\newcommand{\cX}{\mathcal{X}}
\newcommand{\cY}{\mathcal{Y}}
\newcommand{\cZ}{\mathcal{Z}}


% blackboard bold

\newcommand{\Abb}{\mathbb{A}}

\newcommand{\Cbb}{\mathbb{C}}
\newcommand{\Dbb}{\mathbb{D}}
\newcommand{\Ebb}{\mathbb{E}}
\newcommand{\Fbb}{\mathbb{F}}
\newcommand{\Gbb}{\mathbb{G}}
\newcommand{\Hbb}{\mathbb{H}}
\newcommand{\Ibb}{\mathbb{I}}
\newcommand{\Jbb}{\mathbb{J}}
\newcommand{\Kbb}{\mathbb{K}}
\newcommand{\Lbb}{\mathbb{L}}
\newcommand{\Mbb}{\mathbb{M}}
\newcommand{\Nbb}{\mathbb{N}}
\newcommand{\Obb}{\mathbb{O}}
\newcommand{\Pbb}{\mathbb{P}}
\newcommand{\Qbb}{\mathbb{Q}}
\newcommand{\Rbb}{\mathbb{R}}
\newcommand{\Sbb}{\mathbb{S}}
\newcommand{\Tbb}{\mathbb{T}}
\newcommand{\Ubb}{\mathbb{U}}
\newcommand{\Vbb}{\mathbb{V}}
\newcommand{\Wbb}{\mathbb{W}}
\newcommand{\Xbb}{\mathbb{X}}
\newcommand{\Ybb}{\mathbb{Y}}
\newcommand{\Zbb}{\mathbb{Z}}

\ifMCM@opt@CTeX
\else
  \RequirePackage{environ}
\fi
\ifpdf
  \RequirePackage{graphicx}
  \RequirePackage{epstopdf}
\else
  \ifxetex
    \RequirePackage{graphicx}
  \else
    \RequirePackage[dvipdfmx]{graphicx}
    \RequirePackage{bmpsize}
  \fi
\fi
\RequirePackage[usenames,dvipsnames,svgnames,table]{xcolor}
\ifpdf
  
\else
  \ifxetex
     \else
    \hypersetup{dvipdfm, hidelinks}
  \fi
\fi
\geometry{a4paper, margin = 0.8in}
%\pagestyle{fancy}
%\fancyhf{}
%\lhead{\small\sffamily CV Prj4 Mnist CNN }
%\rhead{\small\sffamily Page {\color{RoyalBlue}\thepage}\ of \pageref{LastPage}}
%\setlength\parskip{.5\baselineskip}
\renewcommand\tableofcontents{%
    \centerline{\normalfont\Large\bfseries\sffamily\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \vskip 5ex%
    \@starttoc{toc}%
    }
\setcounter{totalnumber}{4}
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\renewcommand{\textfraction}{0.15}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.65}
\renewcommand{\floatpagefraction}{0.60}
\renewcommand{\figurename}{Figure}
\renewcommand{\tablename}{Table}
\graphicspath{{./}{./img/}{./fig/}{./image/}{./figure/}{./picture/}
            {./imgs/}{./figs/}{./images/}{./figures/}{./pictures/}}
\def\maketitle{%
  \let\saved@thepage\thepage
  \let\thepage\relax
  \ifMCM@opt@sheet
  \makesheet
  \fi
  \newpage
  \ifMCM@opt@titlepage
    \MCM@maketitle
  \fi
  \newpage
  \let\thepage\saved@thepage
  \setcounter{page}{1}
  \pagestyle{fancy}
}
\def\abstractname{Summary}
  \newbox\@abstract%
  \setbox\@abstract\hbox{}%
  \long\def\abstract{\bgroup\global\setbox\@abstract\vbox\bgroup\hsize\textwidth\leftskip1.5cm\rightskip1.5cm}%
  \def\endabstract{\egroup\egroup}
  \def\make@abstract{%
    \begin{center}
      \textbf{Airport Security System Optimization}
    \end{center}
    \usebox\@abstract\par
  }
\def\keywordsname{Keywords}
\ifMCM@opt@CTeX
  \newbox\@keywords
  \setbox\@keywords\hbox{}
  \def\keywords{\global\setbox\@keywords\vbox\bgroup\noindent\leftskip0cm}
  \def\endkeywords{\egroup}%
  \def\make@keywords{%
   \par\hskip.4cm\textbf{\keywordsname}: \usebox\@keywords\hfill\par
  }
\else
  \NewEnviron{keywords}{\xdef\@keywords{\expandonce\BODY}}
  \def\make@keywords{%
    \par\hskip.4cm\textbf{\keywordsname}:
    \@keywords\par
  }
\fi
\newcommand{\headset}{2017 \\MCM/ICM \\Summary Sheet}
\newcommand{\problem}[1]{\mcmsetup{problem = #1}}
\def\makesheet{%
  \pagestyle{empty}%
  \null%
  \vspace*{-5pc}%
  \begin{center}
  \begingroup
  \setlength{\parindent}{0pt}
     \begin{minipage}{0.28\linewidth}
      For office use only\\[4pt]
      \makebox[0.15\linewidth][l]{T1}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{T2}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{T3}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{T4}\rule[-2pt]{0.85\linewidth}{0.5pt}
     \end{minipage}%
     \begin{minipage}{0.44\linewidth}
      \centering
      Team Control Number\\[0.7pc]
      {\Huge\textbf{\MCM@opt@tcn}}\\[1.8pc]
      Problem Chosen\\[0.7pc]
      {\Huge\textbf{\MCM@opt@problem}}
     \end{minipage}%
     \begin{minipage}{0.28\linewidth}
      For office use only\\[4pt]
      \makebox[0.15\linewidth][l]{F1}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{F2}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{F3}\rule[-2pt]{0.85\linewidth}{0.5pt}\\[4pt]
      \makebox[0.15\linewidth][l]{F4}\rule[-2pt]{0.85\linewidth}{0.5pt}
     \end{minipage}\par
  \rule{\linewidth}{0.5pt}\par
  \textbf{\headset}%
  \par
  \endgroup
  \vskip 10pt%
  \ifMCM@opt@titleinsheet
    \normalfont \LARGE \@title \par
  \fi
  \end{center}
\ifMCM@opt@keywordsinsheet
  \make@abstract
  \make@keywords
\else
  \make@abstract
\fi}
\newcommand{\MCM@maketitle}{%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em%
  \ifMCM@opt@abstract%
    \make@abstract
    \make@keywords
  \fi%
}
\newtheorem{Theorem}{Theorem}[section]
\newtheorem{Lemma}[Theorem]{Lemma}
\newtheorem{Corollary}[Theorem]{Corollary}
\newtheorem{Proposition}[Theorem]{Proposition}
\newtheorem{Definition}[Theorem]{Definition}
\newtheorem{Example}[Theorem]{Example}
\providecommand{\dif}{\mathop{}\!\mathrm{d}}
\providecommand{\me}{\mathrm{e}}
\providecommand{\mi}{\mathrm{i}}
\definecolor{grey}{rgb}{0.8,0.8,0.8}
\definecolor{darkgreen}{rgb}{0,0.3,0}
\definecolor{darkblue}{rgb}{0,0,0.3}
\def\lstbasicfont{\fontfamily{pcr}\selectfont\footnotesize}

\lstset{%
%    numbers=left,
%    numberstyle=\small,%
    showstringspaces=false,
    showspaces=false,%
    tabsize=4,%
    frame=lines,%
    basicstyle={\footnotesize\lstbasicfont},%
    keywordstyle=\color{darkblue}\bfseries,%
    identifierstyle=,%
    commentstyle=\color{darkgreen},%\itshape,%
    stringstyle=\color{black}%
}

%\lstset{%
%	basicstyle={\lstbasicfont},                 % set font
%	showstringspaces=false,                     % do not emphasize spaces in strings
%	tabsize=4,                                  % number of spaces of a TAB
%	mathescape=false,escapechar=§,              % escape to latex with §...§
%	upquote=true,                               % upright quotes
%	aboveskip={1.5\baselineskip},               % a bit of space above listings
%	columns=fixed                               % nice spacing
%	language=matlabfloz,                % use our version of highlighting
%	keywordstyle=\color[rgb]{0,0,1},          % keywords in blue
%	commentstyle=\color[rgb]{.133,.545,.133},               % comments
%	stringstyle=\color[rgb]{.627,.126,.941}   % strings in purple
%}

\lstloadlanguages{C,C++,Java,Matlab,Mathematica}

\lstdefinelanguage{matlabfloz}{%
	alsoletter={...},%
	morekeywords={%                             % keywords
		break,case,catch,classdef,continue,else,
		elseif,end,for,function,global,if,
		otherwise,parfor,persistent,
		return,spmd,switch,try,while,...},        % Use the matlab "iskeyword" command to get those
	comment=[l]\%,                              % comments
	morecomment=[l]...,                         % comments
	morecomment=[s]{\%\{}{\%\}},                % block comments
	morestring=[m]'                             % strings 
}[keywords,comments,strings]%
\endinput
%% 
%% This work consists of these files mcmthesis.dtx,
%%                                   figures/ and
%%                                   code/,
%% and the derived files             mcmthesis.cls,
%%                                   mcmthesis-demo.tex,
%%                                   README,
%%                                   LICENSE,
%%                                   mcmthesis.pdf and
%%                                   mcmthesis-demo.pdf.
%%
%% End of file `mcmthesis.cls'.
