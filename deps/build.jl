using Pkg
using Conda

ENV["PYTHON"] = ""
Pkg.build("PyCall")

# Conda.add(["pytorch", "torchvision"]; channel = "pytorch")
Conda.add("fair-esm")
