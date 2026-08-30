# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch. Historical commits and
archived experiment artifacts are not supported.

## Reporting a vulnerability

Please use the repository's private GitHub vulnerability-reporting channel when
it is available. If it is not available, open an issue titled **Security
contact requested** without including exploit details, credentials, personal
data, patient data, or sensitive images. A maintainer can then arrange a
private channel.

Do not publish a proof of concept or a secret in a public issue, pull request,
commit, log, dataset example, or model prompt.

## Handling secrets and data

Never commit API keys, access tokens, passwords, private endpoints, local
configuration files, or credential-bearing command output. Use the documented
example configuration and environment variables for local setup.

MedGEN-Bench is distributed separately. Contributors must follow its access
terms and must not add raw datasets, generated results, medical images, or
potentially identifiable clinical information to this repository.

## Scope

This repository provides research evaluation software, not clinical advice or
a medical device. Report security issues in the code and its documented local
deployment path; report dataset or model-access issues to their respective
providers.
