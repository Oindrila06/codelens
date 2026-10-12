"""Parse Python source into function/class/module chunks plus call and import references."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import tree_sitter_python as tsp
from tree_sitter import Language, Node, Parser

MODULE_CHUNK_MAX_LINES = 80


@dataclass
class Chunk:
    name: str
    qualname: str
    kind: str  # function | class | module
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str
    signature: str = ""
    is_test: bool = False


@dataclass
class Ref:
    name: str
    line: int
    kind: str  # call | import
    module: str = ""
    chunk_index: int = 0  # index into ParsedFile.chunks of the enclosing chunk


@dataclass
class ParsedFile:
    path: str
    chunks: list[Chunk] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tsp.language()))


def _text(node: Node | None) -> str:
    return node.text.decode("utf8", errors="replace") if node is not None and node.text else ""


def is_test_path(path: str) -> bool:
    p = path.replace("\\", "/")
    base = p.rsplit("/", 1)[-1]
    return "/tests/" in f"/{p}" or "/test/" in f"/{p}" or base.startswith("test_") or base.endswith("_test.py")


def _collect_calls(node: Node, out: list[tuple[str, int]], stop_at_defs: bool) -> None:
    stack = [node]
    while stack:
        n = stack.pop()
        if n is not node and stop_at_defs and n.type in ("function_definition", "class_definition"):
            continue
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier":
                    out.append((_text(fn), n.start_point[0] + 1))
                elif fn.type == "attribute":
                    attr = fn.child_by_field_name("attribute")
                    out.append((_text(attr), n.start_point[0] + 1))
        stack.extend(n.children)


def _collect_imports(node: Node, out: list[Ref]) -> None:
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "import_statement":
            for child in n.children_by_field_name("name"):
                target = child.child_by_field_name("name") if child.type == "aliased_import" else child
                module = _text(target)
                if module:
                    out.append(Ref(module.split(".")[-1], n.start_point[0] + 1, "import", module))
        elif n.type == "import_from_statement":
            module = _text(n.child_by_field_name("module_name")).lstrip(".")
            if module:
                out.append(Ref(module.split(".")[-1], n.start_point[0] + 1, "import", module))
            for child in n.children_by_field_name("name"):
                target = child.child_by_field_name("name") if child.type == "aliased_import" else child
                name = _text(target).split(".")[-1]
                if name:
                    out.append(Ref(name, n.start_point[0] + 1, "import", module))
        stack.extend(n.children)


def _signature(fn: Node) -> str:
    name = _text(fn.child_by_field_name("name"))
    params = " ".join(_text(fn.child_by_field_name("parameters")).split())
    ret = _text(fn.child_by_field_name("return_type"))
    return f"{name}{params}" + (f" -> {ret}" if ret else "")


def parse_python(path: str, source: str) -> ParsedFile:
    lines = source.splitlines()
    tree = _parser().parse(source.encode("utf8"))
    root = tree.root_node
    parsed = ParsedFile(path=path)
    test_file = is_test_path(path)
    covered: set[int] = set()

    def add_chunk(chunk: Chunk) -> int:
        parsed.chunks.append(chunk)
        return len(parsed.chunks) - 1

    def visit(container: Node, class_stack: list[str]) -> None:
        for child in container.children:
            outer = child
            node = child
            if child.type == "decorated_definition":
                inner = child.child_by_field_name("definition")
                if inner is None:
                    continue
                node = inner
            if node.type == "function_definition":
                name = _text(node.child_by_field_name("name"))
                start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
                idx = add_chunk(
                    Chunk(
                        name=name,
                        qualname=".".join([*class_stack, name]),
                        kind="function",
                        start_line=start,
                        end_line=end,
                        text="\n".join(lines[start - 1 : end]),
                        signature=_signature(node),
                        is_test=test_file or name.startswith("test"),
                    )
                )
                covered.update(range(start, end + 1))
                calls: list[tuple[str, int]] = []
                _collect_calls(node, calls, stop_at_defs=False)
                parsed.refs.extend(Ref(n, ln, "call", chunk_index=idx) for n, ln in calls)
            elif node.type == "class_definition":
                name = _text(node.child_by_field_name("name"))
                body = node.child_by_field_name("body")
                start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
                first_member = None
                if body is not None:
                    for member in body.children:
                        m = member.child_by_field_name("definition") if member.type == "decorated_definition" else member
                        if m is not None and m.type in ("function_definition", "class_definition"):
                            first_member = member.start_point[0] + 1
                            break
                header_end = (first_member - 1) if first_member else end
                header_end = max(start, min(header_end, start + 60))
                idx = add_chunk(
                    Chunk(
                        name=name,
                        qualname=".".join([*class_stack, name]),
                        kind="class",
                        start_line=start,
                        end_line=header_end,
                        text="\n".join(lines[start - 1 : header_end]),
                        signature=f"class {name}",
                        is_test=test_file,
                    )
                )
                covered.update(range(start, header_end + 1))
                calls = []
                if body is not None:
                    for member in body.children:
                        m = member.child_by_field_name("definition") if member.type == "decorated_definition" else member
                        if m is None or m.type not in ("function_definition", "class_definition"):
                            _collect_calls(member, calls, stop_at_defs=True)
                parsed.refs.extend(Ref(n, ln, "call", chunk_index=idx) for n, ln in calls)
                if body is not None:
                    visit(body, [*class_stack, name])

    visit(root, [])

    # Module-level chunk: top-level lines outside any def/class (imports, constants, scripts).
    module_lines = [
        i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()
    ]
    if module_lines:
        shown = module_lines[:MODULE_CHUNK_MAX_LINES]
        idx = add_chunk(
            Chunk(
                name="<module>",
                qualname="<module>",
                kind="module",
                start_line=shown[0],
                end_line=shown[-1],
                text="\n".join(lines[i - 1] for i in shown),
                signature="",
                is_test=test_file,
            )
        )
        top_calls: list[tuple[str, int]] = []
        for child in root.children:
            c = child.child_by_field_name("definition") if child.type == "decorated_definition" else child
            if c is not None and c.type in ("function_definition", "class_definition"):
                continue
            _collect_calls(child, top_calls, stop_at_defs=True)
        parsed.refs.extend(Ref(n, ln, "call", chunk_index=idx) for n, ln in top_calls)
        imports: list[Ref] = []
        _collect_imports(root, imports)
        for ref in imports:
            ref.chunk_index = idx
        parsed.refs.extend(imports)
    return parsed


def signatures(parsed: ParsedFile) -> dict[str, str]:
    """qualname -> signature, for functions only."""
    return {c.qualname: c.signature for c in parsed.chunks if c.kind == "function"}
