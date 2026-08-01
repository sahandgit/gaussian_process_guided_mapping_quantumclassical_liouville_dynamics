\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ut-thesis}[2026/07/28 Local SGS-aligned thesis class]

\newif\ifut@onehalfspacing
\DeclareOption{onehalfspacing}{\ut@onehalfspacingtrue}
\DeclareOption{normalmargins}{}
\DeclareOption{equalmargins}{}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[oneside,openany]{book}

\RequirePackage[letterpaper,margin=1in,bindingoffset=0.15in]{geometry}
\RequirePackage{setspace}
\ifut@onehalfspacing\onehalfspacing\fi

\newcommand{\@degree}{Master of Science}
\newcommand{\@departmentprefix}{Graduate Department of}
\newcommand{\@department}{Chemistry}
\newcommand{\@gradyear}{\the\year}
\newcommand{\@copyrighttext}{}

\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\department}[2][]{%
  \gdef\@departmentprefix{#1}%
  \gdef\@department{#2}%
}
\newcommand{\gradyear}[1]{\gdef\@gradyear{#1}}
\newcommand{\copyrighttext}[1]{\gdef\@copyrighttext{#1}}

\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \vspace*{1.0in}
    {\Large\bfseries \@title\par}
    \vspace{1.3in}
    {\large by\par}
    \vspace{0.25in}
    {\Large \@author\par}
    \vfill
    {\large A thesis submitted in conformity with the requirements\\
    for the degree of \@degree\par}
    \vspace{0.25in}
    {\large \@departmentprefix\ \@department\\
    University of Toronto\par}
    \vspace{0.45in}
    {\large \@gradyear\par}
    \vspace{0.35in}
    {\large \@copyrighttext\par}
  \end{titlepage}
  \setcounter{footnote}{0}
}

\newenvironment{abstract}{%
  \chapter*{Abstract}
  \addcontentsline{toc}{chapter}{Abstract}
}{}

\newenvironment{acknowledgements}{%
  \chapter*{Acknowledgements}
  \addcontentsline{toc}{chapter}{Acknowledgements}
}{}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\setlength{\emergencystretch}{3em}
\raggedbottom
\AtBeginDocument{\pagestyle{plain}}
