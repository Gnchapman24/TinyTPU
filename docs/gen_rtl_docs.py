"""Generate MkDocs pages for every SystemVerilog module in ``rtl/``.

The actual parameters and ports come from the source via pyslang (the same slang front-end
that yosys-slang uses for formal), so the tables cannot drift from the RTL. Descriptions come
from comments, following AGENTS.md §7:

* A ``/** ... */`` header comment directly above ``module``. Its first line is the summary,
  and it must contain a line starting with ``Latency:``.
* A trailing ``// ...`` comment on the line of every parameter and port declaration
  (one declaration per line).

Packages (``*_pkg.sv``) are compiled so that modules can import them, but they don't get pages.

Writes ``docs/rtl/index.md`` and ``docs/rtl/<module>.md`` (regenerated from scratch on each
run). Exits with status 1 and lists every problem if any rule is broken, so that
``make docs`` fails.

Usage:
    uv run python docs/gen_rtl_docs.py [--rtl rtl] [--out docs/rtl]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pyslang

SyntaxKind = pyslang.syntax.SyntaxKind


@dataclass
class Item:
    """One documented parameter or port.

    Attributes:
        name: Identifier as declared.
        kind: Port direction (``input``/``output``/``inout``), or ``parameter``.
        type: Declared type text, e.g. ``logic [W-1:0]``. Empty if implicit.
        default: Default value text for parameters; empty for ports.
        description: Trailing ``//`` comment text.
        line: 1-based source line of the identifier.
    """

    name: str
    kind: str
    type: str
    default: str
    description: str
    line: int


@dataclass
class Module:
    """Documentation extracted for one module.

    Attributes:
        name: Module name.
        path: Source file, relative to the repository root.
        line: 1-based line of the ``module`` keyword.
        summary: First line of the header comment.
        details: Remaining header lines, excluding the latency line.
        latency: Text after ``Latency:`` in the header.
        params: Parameters from the ``#( ... )`` list, in order.
        ports: ANSI ports, in order.
    """

    name: str
    path: Path
    line: int
    summary: str = ""
    details: list[str] = field(default_factory=list)
    latency: str = ""
    params: list[Item] = field(default_factory=list)
    ports: list[Item] = field(default_factory=list)


def _norm(text: str) -> str:
    """Collapse whitespace runs to single spaces."""
    return " ".join(text.split())


def _trailing_comment(line_text: str) -> str:
    """Return the text of a ``//`` comment on a source line, or an empty string."""
    idx = line_text.find("//")
    return line_text[idx + 2 :].strip() if idx >= 0 else ""


def _header_comment(lines: list[str], module_line: int) -> list[str] | None:
    """Return the cleaned lines of the ``/** ... */`` block directly above ``module``.

    Args:
        lines: Source file lines.
        module_line: 1-based line of the ``module`` keyword.

    Returns:
        Comment body lines with ``/**``, ``*/`` and leading ``*`` removed, or ``None`` if no
        such block sits directly above the module (only blank lines may separate them).
    """
    i = module_line - 2
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0 or not lines[i].rstrip().endswith("*/"):
        return None
    end = i
    while i >= 0 and "/**" not in lines[i]:
        if "/*" in lines[i]:
            return None  # an ordinary /* */ block, not a doc header
        i -= 1
    if i < 0:
        return None
    body = []
    for raw in lines[i : end + 1]:
        text = raw.strip()
        if text.startswith("/**"):
            text = text[3:]
        if text.endswith("*/"):
            text = text[:-2]
        text = text.strip()
        if text.startswith("*"):
            text = text[1:].strip()
        body.append(text)
    while body and not body[0]:
        body.pop(0)
    while body and not body[-1]:
        body.pop()
    return body


def _check_items(mod: Module, items: list[Item], what: str, errors: list[str]) -> None:
    """Record an error for each item without a description or sharing a line."""
    seen: dict[int, str] = {}
    for it in items:
        where = f"{mod.path}:{it.line}"
        if it.line in seen:
            errors.append(
                f"{where}: {what} '{it.name}' shares a line with '{seen[it.line]}'; "
                "declare one per line so each gets its own // description"
            )
        seen[it.line] = it.name
        if not it.description:
            errors.append(f"{where}: {what} '{it.name}' has no trailing // description")


def extract(path: Path, root: Path, sm, tree, errors: list[str]) -> list[Module]:
    """Extract every module declared in one parsed file.

    Args:
        path: Absolute source path.
        root: Repository root, for relative paths in messages and pages.
        sm: The slang source manager that owns ``tree``.
        tree: The parsed syntax tree for ``path``.
        errors: List that problems are appended to.

    Returns:
        The modules found in the file.
    """
    lines = path.read_text().splitlines()
    rel = path.relative_to(root)
    modules = []
    for member in tree.root.members:
        if member.kind != SyntaxKind.ModuleDeclaration:
            continue
        hdr = member.header
        mod = Module(
            name=hdr.name.valueText,
            path=rel,
            line=sm.getLineNumber(hdr.moduleKeyword.location),
        )

        body = _header_comment(lines, mod.line)
        if body is None:
            errors.append(f"{rel}:{mod.line}: module '{mod.name}' has no /** ... */ header directly above it")
        else:
            if not body or not body[0]:
                errors.append(f"{rel}:{mod.line}: module '{mod.name}' header has no summary line")
            else:
                mod.summary = body[0]
            for text in body[1:]:
                if text.lower().startswith("latency:"):
                    mod.latency = text.split(":", 1)[1].strip()
                else:
                    mod.details.append(text)
            if not mod.latency:
                errors.append(f"{rel}:{mod.line}: module '{mod.name}' header has no 'Latency:' line")

        if hdr.parameters is not None:
            for decl in hdr.parameters.declarations:
                if not isinstance(decl, pyslang.syntax.SyntaxNode):
                    continue  # separator commas
                if decl.kind != SyntaxKind.ParameterDeclaration:
                    errors.append(f"{rel}:{mod.line}: module '{mod.name}': unsupported parameter form '{_norm(str(decl))}'")
                    continue
                for d in decl.declarators:
                    if not isinstance(d, pyslang.syntax.SyntaxNode):
                        continue
                    ln = sm.getLineNumber(d.name.location)
                    default = _norm(str(d.initializer.expr)) if d.initializer else ""
                    mod.params.append(
                        Item(d.name.valueText, "parameter", _norm(str(decl.type)), default,
                             _trailing_comment(lines[ln - 1]), ln)
                    )

        ports = hdr.ports
        if ports is not None:
            if ports.kind != SyntaxKind.AnsiPortList:
                errors.append(f"{rel}:{mod.line}: module '{mod.name}' must use an ANSI port list")
            else:
                for p in ports.ports:
                    if not isinstance(p, pyslang.syntax.SyntaxNode):
                        continue
                    if p.kind != SyntaxKind.ImplicitAnsiPort:
                        errors.append(f"{rel}:{mod.line}: module '{mod.name}': unsupported port form '{_norm(str(p))}'")
                        continue
                    ph = p.header
                    direction = ph.direction.valueText if getattr(ph, "direction", None) else ""
                    dtype = _norm(str(ph.dataType)) if getattr(ph, "dataType", None) is not None else ""
                    ln = sm.getLineNumber(p.declarator.name.location)
                    mod.ports.append(
                        Item(p.declarator.name.valueText, direction, dtype, "",
                             _trailing_comment(lines[ln - 1]), ln)
                    )

        _check_items(mod, mod.params, "parameter", errors)
        _check_items(mod, mod.ports, "port", errors)
        modules.append(mod)
    return modules


def _cell(text: str) -> str:
    """Escape text for a Markdown table cell."""
    return text.replace("|", "\\|") if text else "—"


def render_module(mod: Module) -> str:
    """Render one module page as Markdown."""
    out = [f"# `{mod.name}`", "", mod.summary, ""]
    if mod.details:
        out += ["\n".join(mod.details).strip(), ""]
    out += [f"**Latency:** {mod.latency}", "", f"Source: `{mod.path}` (line {mod.line})", ""]
    out += ["## Parameters", ""]
    if mod.params:
        out += ["| Name | Type | Default | Description |", "|---|---|---|---|"]
        out += [f"| `{p.name}` | {_cell(f'`{p.type}`' if p.type else '')} | "
                f"{_cell(f'`{p.default}`' if p.default else '')} | {_cell(p.description)} |"
                for p in mod.params]
    else:
        out.append("None.")
    out += ["", "## Ports", ""]
    if mod.ports:
        out += ["| Name | Direction | Type | Description |", "|---|---|---|---|"]
        out += [f"| `{p.name}` | {p.kind} | {_cell(f'`{p.type}`' if p.type else '')} | {_cell(p.description)} |"
                for p in mod.ports]
    else:
        out.append("None.")
    out += ["", "---", "", "*Generated from the RTL by `docs/gen_rtl_docs.py`. Do not edit.*", ""]
    return "\n".join(out)


def render_index(modules: list[Module]) -> str:
    """Render the RTL module index page as Markdown."""
    out = ["# RTL modules", ""]
    if not modules:
        out += ["No RTL modules yet.", ""]
    else:
        out += ["| Module | Summary | Latency |", "|---|---|---|"]
        out += [f"| [`{m.name}`]({m.name}.md) | {_cell(m.summary)} | {_cell(m.latency)} |" for m in modules]
        out.append("")
    out += ["*Generated from the RTL by `docs/gen_rtl_docs.py`. Do not edit.*", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Parse ``rtl/``, check documentation rules, and write the pages.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit status: 0 on success, 1 if any rule is broken or the RTL does not compile.
    """
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rtl", type=Path, default=root / "rtl", help="RTL source directory")
    ap.add_argument("--out", type=Path, default=root / "docs" / "rtl", help="output directory")
    args = ap.parse_args(argv)
    rtl_dir, out_dir = args.rtl.resolve(), args.out.resolve()

    # Packages first, so modules that import them compile.
    sources = sorted(rtl_dir.glob("*_pkg.sv")) + sorted(
        p for p in rtl_dir.glob("*.sv") if not p.name.endswith("_pkg.sv")
    ) if rtl_dir.is_dir() else []

    errors: list[str] = []
    modules: list[Module] = []
    if sources:
        sm = pyslang.SourceManager()
        trees = [(p, pyslang.syntax.SyntaxTree.fromFile(str(p), sm)) for p in sources]
        comp = pyslang.ast.Compilation()
        for _, t in trees:
            comp.addSyntaxTree(t)
        diags = [d for d in comp.getAllDiagnostics() if d.isError()]
        if diags:
            engine = pyslang.DiagnosticEngine(sm)
            client = pyslang.TextDiagnosticClient()
            engine.addClient(client)
            for d in diags:
                engine.issue(d)
            print(client.getString(), file=sys.stderr)
            print(f"gen_rtl_docs: RTL does not compile ({len(diags)} error(s))", file=sys.stderr)
            return 1
        for path, tree in trees:
            modules += extract(path, root, sm, tree, errors)

    names = [m.name for m in modules]
    for n in sorted({n for n in names if names.count(n) > 1}):
        errors.append(f"module '{n}' is declared more than once")

    if errors:
        for e in errors:
            print(f"gen_rtl_docs: {e}", file=sys.stderr)
        print(f"gen_rtl_docs: {len(errors)} documentation problem(s); see AGENTS.md §7", file=sys.stderr)
        return 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    modules.sort(key=lambda m: m.name)
    (out_dir / "index.md").write_text(render_index(modules))
    for m in modules:
        (out_dir / f"{m.name}.md").write_text(render_module(m))
    print(f"gen_rtl_docs: wrote {len(modules)} module page(s) to {out_dir.relative_to(root) if out_dir.is_relative_to(root) else out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
