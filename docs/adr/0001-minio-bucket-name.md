# ADR 0001: Use a DNS-Compatible MinIO Bucket Name

- **Status**: Accepted
- **Date**: 2026-11-05
- **Decision owners**: Project maintainers

## Context

The constitution names `zeroshot_learning_output` as the default prediction bucket.
S3-compatible bucket naming rules disallow underscores, so MinIO may reject that value
when the application creates the bucket automatically.

## Decision

Use `zeroshot-learning-output` as the default bucket. Keep the bucket configurable through
`MINIO_BUCKET`, and have the application create the configured bucket when it is missing.

## Reason

The hyphenated name complies with DNS-compatible S3 bucket naming rules and lets the
documented local workflow create its bucket without manual intervention.

## Benefits

- Clean local setup can create the prediction bucket automatically.
- The default is compatible with MinIO and S3-style naming constraints.
- A configurable bucket remains available for other environments.

## Risks

- The default differs from the name first stated in the constitution and from any existing
  storage using that underscore name.
- Operators must configure `MINIO_BUCKET` explicitly when connecting to a pre-existing
  bucket with a different name.

## Reproducibility Impact

The bucket name is part of the documented MinIO configuration and is pinned here as
`zeroshot-learning-output` for the default setup. Reproductions must use the same value or
document an intentional environment-specific override; prediction object paths and JSON
contents do not otherwise change.
