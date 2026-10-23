# Contributing

Thanks for helping. Bug reports with a short clip description (camera, gym, what went wrong and when) are as useful
as code.

## Pull requests

- Keep the Python reference (`python/directe_tracking`) and the Swift port (`Sources/`) in step: a change to the
  tracker's behaviour goes into both, and `tools/golden/tracker_golden.py` regenerates the fixtures the Swift tests
  replay. CI checks the regenerated fixtures match the committed ones and that `swift test` passes.
- Judge tracker changes on hand-checked labels (which box is the climber, every few seconds), not on how a run looks.
  Say in the PR which clips you checked and what changed.
- Don't commit footage, frames of people, or fixtures recorded on footage of people who haven't agreed to it. Point
  the tests at those with `DIRECTE_GOLDEN` instead.
- Weights don't go in the repo: `tools/export_coreml.py` fetches them from upstream.

## Contributor Licence Agreement

This repository is AGPL-3.0. The same code also ships inside the Directe app, which isn't open source, so every
contribution needs the [Contributor Licence Agreement](CLA.md). A bot asks on your first pull request: reply
"I have read the CLA Document and I hereby sign the CLA". You keep the copyright in your contribution.
