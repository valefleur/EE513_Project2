"""report.py -- human-friendly console output for the homework scripts.

Everything wraps at WIDTH (100) characters, aligns numbers in columns, and keeps labels,
units and uncertainties next to the values they describe. Pure formatting: no measurement
happens here.
"""
import textwrap

WIDTH = 100
PM = "\u00b1"


def say(text, indent=0, hang=0):
    """Print `text` wrapped at WIDTH. `indent` indents every line; `hang` indents continuation lines more."""
    print(textwrap.fill(text, width=WIDTH, initial_indent=" " * indent,
                        subsequent_indent=" " * (indent + hang)))


def pm(value, unc, digits=3, signed=False):
    """Format 'value ± uncertainty' with fixed decimals, e.g. '0.058 ± 0.012'. unc=None -> value only."""
    v = f"{value:{'+' if signed else ''}.{digits}f}"
    return v if unc is None else f"{v} {PM} {unc:.{digits}f}"


def table(headers, rows, indent=4, gap=2, left=(0,)):
    """Print an aligned table. Columns listed in `left` are left-aligned, the others right-aligned.

    headers : list of str (put units in the header, e.g. 'median [px]').
    rows : list of lists; every cell is converted with str(). A row equal to None prints a rule.
    Raises ValueError if the table would be wider than WIDTH.
    """
    cells = [[str(c) for c in r] for r in rows if r is not None]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)] if cells else [len(h) for h in headers]
    total = indent + sum(widths) + gap * (len(widths) - 1)
    if total > WIDTH:
        raise ValueError(f"table is {total} characters wide, limit {WIDTH}")
    fmt = lambda r: " " * indent + (" " * gap).join(
        c.ljust(w) if i in left else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    rule = " " * indent + (" " * gap).join("-" * w for w in widths)
    print(fmt(headers))
    print(rule)
    for r in rows:
        print(rule if r is None else fmt([str(c) for c in r]))


def key_values(rows, indent=4):
    """Print (label, value, unit) rows with labels, values and units each lined up in a column."""
    lw = max(len(r[0]) for r in rows)
    vw = max(len(r[1]) for r in rows)
    for label, value, unit in rows:
        line = " " * indent + f"{label.ljust(lw)}  {value.rjust(vw)}  {unit}".rstrip()
        if len(line) <= WIDTH:
            print(line)
        else:                                   # too long: label (and unit) on one line, value below
            print(" " * indent + f"{label}  {unit}".rstrip())
            say(value, indent=indent + 4, hang=0)


def banner(kind, name, title=None):
    """Heading such as 'CHECK 02a' (kind='CHECK') or 'DELIVERABLE 02', followed by a wrapped title."""
    print(f"\n  {kind} {name}" + ("" if title is None else f" -- {title}"))
    print("  " + "=" * (len(kind) + len(name) + 1 + (0 if title is None else 4 + len(title))))


def grid(label, values, per_row=10, digits=4, indent=4):
    """Print a labelled sequence in rows of `per_row` values, each value tagged with its index."""
    for start in range(0, len(values), per_row):
        chunk = values[start:start + per_row]
        print(" " * indent + f"{label} " + " ".join(f"{i:>7d}" for i in range(start, start + len(chunk))))
        print(" " * indent + " " * (len(label) + 1) + " ".join(f"{v:7.{digits}f}" for v in chunk))
