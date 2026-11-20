#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
REPORT_DIR = REPO_ROOT / "report"
BUILD_DIR = REPORT_DIR / "build"
DIAGRAM_DIR = BUILD_DIR / "diagrams"


MERMAID_IMAGE = (
    "minlag/mermaid-cli@sha256:"
    "18938e53497376d524e04b1fc0bf7b6d61a491ce9ff7032bb2855d57155acb10"
)
PANDOC_IMAGE = (
    "pandoc/core@sha256:"
    "62518bfb6833705dc90a1634d8d879299bd820b3f74ad3b32fba8b618452553a"
)
TYPST_IMAGE = (
    "ghcr.io/typst/typst@sha256:"
    "032e292249bcd378480cc7c142cfa324b63ef8aadeb88d7e7230320c4c9c422f"
)


SOURCE = REPORT_DIR / "report.md"

AUTHOR = "Alpoz Hazal Turkan"
GROUP = "11-413a"
TITLE = "task-blocks: a multi-agent assistant with LLM routing and a verification layer"

MERMAID_BLOCK = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def run(command: list[str], *, label: str) -> None:
    print(f"  {label} ... ", end="", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("FAILED")
        sys.stderr.write(result.stdout[-4000:] + "\n" + result.stderr[-4000:] + "\n")
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    print("ok")


def docker_run(image: str, args: list[str], *, mount: Path, label: str) -> None:
    run(
        [
            "docker", "run", "--rm",
            "-v", f"{mount}:/data",
            "-w", "/data",
            image,
            *args,
        ],
        label=label,
    )


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    return (
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:24], "big"),
    )


def _width_for(path: Path) -> int:
    width, height = _png_size(path)
    if not width or not height:
        return 80
    aspect = width / height
    if aspect >= 2.0:
        return 100
    if aspect >= 1.2:
        return 85
    if aspect >= 0.8:
        return 70
    return 55


def render_diagrams(text: str, source_name: str) -> str:
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def replace(match: re.Match[str]) -> str:
        counter["n"] += 1
        body = match.group(1)
        digest = hashlib.sha256(body.encode()).hexdigest()[:12]
        stem = f"{Path(source_name).stem.lower()}-{counter['n']:02d}-{digest}"
        png = DIAGRAM_DIR / f"{stem}.png"

        if not png.exists():
            mmd = DIAGRAM_DIR / f"{stem}.mmd"
            mmd.write_text(body, encoding="utf-8")
            docker_run(
                MERMAID_IMAGE,
                [
                    "-i", f"diagrams/{mmd.name}",
                    "-o", f"diagrams/{png.name}",
                    "-b", "white",


                    "-s", "2",
                    "-q",
                ],
                mount=BUILD_DIR,
                label=f"diagram {stem}",
            )
        else:
            print(f"  diagram {stem} ... cached")


        return f"\n![](diagrams/{png.name}){{width={_width_for(png)}%}}\n"

    return MERMAID_BLOCK.sub(replace, text)


def front_matter() -> str:
    return (
        "---\n"
        f'title: "{TITLE}"\n'
        f'author: "{AUTHOR}, group {GROUP}"\n'
        "lang: en\n"
        "papersize: a4\n"
        "fontsize: 10pt\n"
        "margin:\n"
        "  x: 2cm\n"
        "  y: 2cm\n"
        "---\n\n"
    )


def build(keep_intermediates: bool) -> int:
    if not SOURCE.exists():
        raise SystemExit(f"Missing report source: {SOURCE}")

    if BUILD_DIR.exists() and not keep_intermediates:
        for item in BUILD_DIR.iterdir():
            if item != DIAGRAM_DIR:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print("Rendering diagrams")
    body = render_diagrams(SOURCE.read_text(encoding="utf-8"), SOURCE.name)

    combined = BUILD_DIR / "report.md"
    combined.write_text(front_matter() + body, encoding="utf-8")
    print(f"  combined {combined.stat().st_size // 1024} KiB")

    print("Converting")
    docker_run(
        PANDOC_IMAGE,
        [
            "--standalone",
            "--toc",
            "--toc-depth=2",


            "--from=markdown+pipe_tables+backtick_code_blocks+raw_attribute",
            "--to=typst",
            "-o", "report.typ",
            "report.md",
        ],
        mount=BUILD_DIR,
        label="pandoc markdown to typst",
    )


    typ = BUILD_DIR / "report.typ"
    source = typ.read_text(encoding="utf-8")
    marker = "  set document(\n    title: title,"
    if marker in source:
        typ.write_text(
            source.replace(marker, f"{marker}\n    date: none,", 1),
            encoding="utf-8",
        )
    else:
        print("  note: could not set 'date: none'; check the pandoc template")

    docker_run(
        TYPST_IMAGE,
        ["compile", "report.typ", "report.pdf"],
        mount=BUILD_DIR,
        label="typst compile",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final = REPORT_DIR / "report.pdf"
    shutil.copy2(BUILD_DIR / "report.pdf", final)
    size_kb = final.stat().st_size // 1024

    print(f"\nwrote {final.relative_to(REPO_ROOT)} ({size_kb} KiB)")
    if not keep_intermediates:
        print(f"intermediates in {BUILD_DIR.relative_to(REPO_ROOT)} (diagram cache kept)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PDF report.")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep report.md and report.typ for debugging the conversion.",
    )
    args = parser.parse_args()

    if not shutil.which("docker"):
        raise SystemExit(
            "docker is required: the toolchain (mermaid-cli, pandoc, typst) "
            "runs in containers so nothing has to be installed on the host."
        )
    return build(args.keep_intermediates)


if __name__ == "__main__":
    raise SystemExit(main())
