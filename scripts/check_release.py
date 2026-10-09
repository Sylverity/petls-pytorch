"""Check release metadata and optional build artifacts (Python 3.11+)."""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check_release(tag: str | None = None, dist_dir: Path | None = None) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    module = ast.parse((ROOT / "src/petls_pytorch/__init__.py").read_text())
    versions = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
    ]
    if versions != [version]:
        raise ValueError("Package __version__ does not match pyproject.toml")
    citation = (ROOT / "CITATION.cff").read_text()
    if not re.search(rf'^version: "{re.escape(version)}"$', citation, re.M):
        raise ValueError("CITATION.cff version does not match pyproject.toml")
    changelog = (ROOT / "CHANGELOG.md").read_text()
    released = re.search(rf"^## {re.escape(version)} - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.M)
    if not released and f"## Unreleased ({version})" not in changelog:
        raise ValueError("Changelog is missing the current version")
    readme = (ROOT / "README.md").read_text()
    if tag is not None:
        if tag != f"v{version}":
            raise ValueError(f"Tag {tag!r} does not match v{version}")
        if not released:
            raise ValueError("A release tag requires a dated changelog entry")
        if f'date-released: "{released[1]}"' not in citation:
            raise ValueError("Citation release date must match the changelog")
        if "unreleased" in readme.lower():
            raise ValueError("Release README still describes an unreleased version")
    for markdown, html in re.findall(r'\]\(([^)]+)\)|src="([^"]+)"', readme):
        target = markdown or html
        if not (target.startswith("https://") or target.startswith("#")):
            raise ValueError(f"README reference will not resolve on PyPI: {target}")
        for prefix in (
            "https://github.com/Sylverity/petls-pytorch/blob/main/",
            "https://raw.githubusercontent.com/Sylverity/petls-pytorch/main/",
        ):
            if (
                target.startswith(prefix)
                and not (ROOT / target[len(prefix) :].split("#")[0]).is_file()
            ):
                raise ValueError(f"README links to a missing repository file: {target}")
    if dist_dir is not None:
        expected = {f"petls_pytorch-{version}-py3-none-any.whl", f"petls_pytorch-{version}.tar.gz"}
        actual = {path.name for path in dist_dir.iterdir() if path.is_file()}
        if actual != expected:
            raise ValueError(
                f"Distribution directory must contain exactly {sorted(expected)}; got {sorted(actual)}"
            )
        with zipfile.ZipFile(dist_dir / f"petls_pytorch-{version}-py3-none-any.whl") as wheel:
            metadata = BytesParser().parsebytes(
                wheel.read(f"petls_pytorch-{version}.dist-info/METADATA")
            )
            if metadata["Version"] != version or metadata["Name"] != project["name"]:
                raise ValueError("Wheel metadata does not match the project")
            if metadata["Description-Content-Type"] != "text/markdown":
                raise ValueError("Wheel README is not declared as Markdown")
            if metadata.get_payload(decode=True).decode("utf-8").strip() != readme.strip():
                raise ValueError("Wheel README is stale")
            for source in (ROOT / "src/petls_pytorch").rglob("*.py"):
                if wheel.read(str(source.relative_to(ROOT / "src"))) != source.read_bytes():
                    raise ValueError(f"Stale wheel source: {source}")
        with tarfile.open(dist_dir / f"petls_pytorch-{version}.tar.gz") as archive:
            names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
            required = {"CITATION.cff", "CHANGELOG.md", "CONTRIBUTING.md", "uv.lock", "MANIFEST.in"}
            for folder in ("tests", "benchmark", "scripts"):
                required.update(
                    str(path.relative_to(ROOT)) for path in (ROOT / folder).rglob("*.py")
                )
            required.update(
                str(path.relative_to(ROOT))
                for path in (ROOT / "tests/variants/data").rglob("*")
                if path.is_file()
            )
            if missing := required - names:
                raise ValueError(f"Source distribution omits required files: {sorted(missing)}")
    print(f"Release metadata{' and distributions' if dist_dir else ''} validated for {version}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--dist-dir", type=Path)
    args = parser.parse_args()
    check_release(args.tag, args.dist_dir)
