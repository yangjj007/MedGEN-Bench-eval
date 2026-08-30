# Contributing

Contributions should keep the repository reproducible from a clean checkout.

- Keep code, documentation, tests, and small deterministic fixtures under
  version control; keep datasets, model weights, caches, generated images,
  evaluation outputs, logs, and experiment runs out of Git.
- Never submit credentials, local configuration, patient information, or
  non-public data. Use `api/config.example.yaml` as the configuration template.
- Add or update tests for behavior changes and run the relevant test commands
  before opening a pull request.
- Preserve the public API and document any required environment variables,
  model access requirements, or external dataset downloads.

By submitting a contribution, you agree to license it under the repository's
Apache-2.0 license. The code license does not cover the externally distributed
MedGEN-Bench dataset or third-party model and service terms.
