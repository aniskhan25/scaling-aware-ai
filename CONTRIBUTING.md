# Contributing

This guide should stay practical, measured, and LUMI-specific.

## Contribution Principles

- Prefer runnable examples over abstract advice.
- Include validation steps for every workflow.
- Record placement metadata before interpreting performance.
- Distinguish measured facts from recommendations.
- Explain when not to scale.
- Keep scripts simple enough for users to inspect and adapt.

## Content Standard

New guide sections should include:

- the problem being solved
- why the problem matters on LUMI
- required assumptions
- commands or scripts, when applicable
- expected artifacts
- interpretation guidance
- common failure modes
- next-step recommendations

## Script Standard

Scripts should:

- fail clearly when required inputs are missing
- write outputs to predictable directories
- emit machine-readable summaries where useful
- avoid hidden dependencies on the original AI Guide lesson order
- avoid hard-coded project accounts in reusable examples where possible
- document any LUMI-specific environment assumptions
- stay small enough that a reader can understand the bottleneck being demonstrated
- avoid extra knobs unless they directly support an observation in the guide

Prefer one clear script over a reusable mini-framework. These examples are teaching tools first.

## Job Script Standard

Job scripts should make these choices visible:

- partition
- account placeholder
- node count
- GPU count
- CPU count
- walltime
- container path or environment source
- rank launch pattern
- key communication environment variables

## Review Checklist

Before merging a substantial guide change, check:

- Does the section help users make a scaling decision?
- Does it define what success and failure look like?
- Does it avoid implying that larger jobs are automatically better?
- Are commands copyable and scoped to the guide directory?
- Are outputs named consistently?
- Are LUMI-specific assumptions explicit?
- Are links to official LUMI documentation used for platform facts?
