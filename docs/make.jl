using TimePtycho
using Documenter

DocMeta.setdocmeta!(TimePtycho, :DocTestSetup, :(using TimePtycho); recursive=true)

makedocs(;
    modules=[TimePtycho],
    authors="chrisbrahms <c.brahms@hw.ac.uk> and contributors",
    repo="https://github.com/cbrahms/TimePtycho.jl/blob/{commit}{path}#{line}",
    sitename="TimePtycho.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cbrahms.github.io/TimePtycho.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cbrahms/TimePtycho.jl",
    devbranch="main",
)
