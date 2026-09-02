# Governance

`petls-pytorch` is maintained by Sylverity Research. Current default reviewers are listed in
[CODEOWNERS](.github/CODEOWNERS).

## Decision making

Routine changes are reviewed through pull requests. Maintainers evaluate changes for correctness,
numerical stability, PETLS compatibility, device and dtype behavior, performance, API continuity,
and maintenance cost. Significant API or algorithm proposals should begin in a GitHub Discussion
or issue so alternatives can be considered before implementation.

Maintainers seek consensus when practical and make the final decision when consensus is not
available. Security decisions may remain private until coordinated disclosure is safe.

## Releases

Maintainers publish releases according to semantic versioning, document user-visible changes in
`CHANGELOG.md`, and use the automated trusted-publishing workflow. Deprecations should provide a
reasonable migration path before removal except when an immediate security fix is necessary.

## Becoming a maintainer

Sustained contributors may be invited to maintain the project based on the quality and consistency
of their contributions, review participation, subject-matter knowledge, and adherence to the Code
of Conduct. Maintainer access may be removed for prolonged inactivity, security reasons, or
conduct violations.

