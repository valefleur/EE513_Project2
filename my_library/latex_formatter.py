import numpy as np

_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}


def latex_escape(value):
    """Escape the characters LaTeX treats as commands, e.g. the _ in APC_0223_white.dng
    (an unescaped _ stops compilation with 'Missing $ inserted')."""
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(value))


def generate_table(lists_of_data, labels, caption=None, output_path=None,
                   escape=False, longtable=False, VERBOSE=True):
    """
    Generate a LaTeX table from a dictionary of data.

    Args:
        lists_of_data (list): A list of lists, where each inner list represents a column of data.
        labels (list): A list of strings representing the column names.
        caption (str, optional): The caption for the table.
        output_path (str, optional): The path to save the generated LaTeX code.
        escape (bool): escape LaTeX special characters in labels and cells.
        longtable (bool): emit a longtable (needs \\usepackage{longtable}) so a table
            longer than a page breaks across pages; the caption goes inside it.
        VERBOSE (bool): print the columns and every row while building.

    Returns:
        str: A string containing the LaTeX code for the table.
    """
    # Extract column names and rows
    columns = list(labels)
    if VERBOSE:
        print(f"Columns: {columns}")
        print(f"There are [{len(columns)}] columns and [{len(lists_of_data)}] lists of data.")
        print(f"Rows: {len(list(lists_of_data)[0])}.")
    fmt = latex_escape if escape else str

    # Start building the LaTeX table
    env = "longtable" if longtable else "tabular"
    latex_table = f"\\begin{{{env}}}{{" + " | ".join(["l"] * len(columns)) + "}\n"
    if longtable and caption:
        latex_table += f"\\caption{{{caption}}} \\\\\n"
    latex_table += " \\hline\n"

    # Add header row
    latex_table += " & ".join(map(fmt, columns)) + " \\\\\n"
    latex_table += " \\hline\n"
    if longtable:
        latex_table += "\\endhead\n"          # repeat the header on every page

    # Add data rows
    rows = zip(*lists_of_data)
    for row in rows:
        latex_table += " & ".join(map(fmt, row)) + " \\\\\n"
        latex_table += " \\hline\n"
        if VERBOSE:
            print(f"Row: {row}")  # Debugging line to check the content of each row

    # End the table
    latex_table += f"\\end{{{env}}}\n"

    if caption and not longtable:
        latex_table = f"\\begin{{table}}[h]\n\\centering\n{latex_table}\\caption{{{caption}}}\n\\end{{table}}\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_table)
    return latex_table

def latex_number(x, max_denominator=16):
    """Format a number exactly for LaTeX: 4 -> "4", 0.5 -> "\\tfrac{1}{2}",
    -1.5 -> "-\\tfrac{3}{2}". Uses fractions.Fraction, so a value that is not a
    simple fraction (denominator > max_denominator) falls back to 4 significant digits."""
    from fractions import Fraction
    f = Fraction(float(x)).limit_denominator(max_denominator)
    if abs(float(f) - float(x)) > 1e-12:
        return f"{x:.4g}"
    if f.denominator == 1:
        return str(f.numerator)
    sign = "-" if f < 0 else ""
    return f"{sign}\\tfrac{{{abs(f.numerator)}}}{{{f.denominator}}}"


def generate_matrix(array, env="bmatrix"):
    """One 2-D array as an amsmath matrix environment (needs \\usepackage{amsmath}).

    Entries go through latex_number(), so halves print as fractions rather than
    0.5, and columns are centered by the environment.
    """
    rows = [" & ".join(latex_number(v) for v in row) for row in np.asarray(array)]
    return f"\\begin{{{env}}}\n  " + " \\\\\n  ".join(rows) + f"\n\\end{{{env}}}"


def generate_matrix_grid(named_arrays, scale=1, per_row=2, output_path=None):
    """Several labeled matrices laid out in an align* block, ``per_row`` to a line.

    Each entry renders as   label = (1/scale) * [matrix * scale]   so that, for
    example, kernels in eighths print with integer (or half-integer) entries.
    Paste into a document that loads amsmath.

    Args:
        named_arrays (dict): LaTeX label (math mode, e.g. r"K_{\\text{G at R}}") -> 2-D array.
        scale (int): common denominator factored out in front; 1 for none.
        per_row (int): matrices per line of the align* block.
        output_path (str, optional): also write the LaTeX to this file.

    Returns:
        str: the LaTeX code.
    """
    pre = "" if scale == 1 else f"\\frac{{1}}{{{scale}}}"
    cells = [f"{label} &= {pre}{generate_matrix(np.asarray(a) * scale)}"
             for label, a in named_arrays.items()]
    lines = [" &\\quad\n".join(cells[i:i + per_row]) for i in range(0, len(cells), per_row)]
    # The braces keep the taller rows (room for fractions) local to this block.
    latex = "% needs \\usepackage{amsmath}\n{\\renewcommand{\\arraystretch}{1.3}\n" \
        "\\begin{align*}\n" + " \\\\[1ex]\n".join(lines) + "\n\\end{align*}}\n"
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex)
    return latex
