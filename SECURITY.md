# Security policy

## Supported versions

Security reports are accepted for the latest release and the `main` branch.
Fixes are developed on `main`; older releases are not maintained as separate
security branches. Use the latest release containing the relevant fix.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/Sylverity/petls-pytorch/security/advisories/new).
Do not include exploit details, credentials, or private data in public issues.

Include the affected version or commit, Python version, operating system,
reproduction steps, and the security impact. Provide a minimal reproducer using
synthetic data where possible. Maintainers will investigate and coordinate a
fix and disclosure with the reporter; response times are not guaranteed.

## Dependency security

Routine dependency updates wait at least 14 days after release. Urgent security
fixes may receive an explicit maintainer-approved exception, recorded in the
pull request. This does not disable the default Dependabot cooldown.

Run `uv audit --locked` to check the locked dependencies, including optional and
development dependencies. CI audits each supported Python version on Linux,
Windows, and macOS dependency selections. Reports are not suppressed merely to
make an audit pass. A clean audit means no known findings from the configured
advisory service at that time; it is not a guarantee of security.

## Untrusted input

Only load trusted serialized Python or PyTorch objects. Do not use pickle-based
loading on untrusted files. Apply resource limits when processing large point
clouds, complexes, images, or other externally supplied scientific data.
