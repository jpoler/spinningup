
def print_computation_graph(root, indent=0):
    if not root:
        return
    prefix = "  "*indent
    for fn, idx in root.next_functions:
        print(f"{prefix}fn: {fn}, idx: {idx}")
        if hasattr(fn, "variable"):
            print(f"{prefix}  var: {fn.variable.shape}")
        print_computation_graph(fn, indent=indent + 1)
