# Public-release runbook

State of the repo's publication hardening, and the steps that remain gated on
the visibility flip.

## Done (verifiable)

- **History rewritten PII-free (2026-07-03).** Audit found PII in exactly two
  early commits (author name in the seed README/final-report; internal
  hostnames/LAN IPs/user@host in an early `docs/AUDIT.md` revision) — all
  already scrubbed at HEAD by the earlier working-tree scrub, then removed
  from *history* with `git-filter-repo --replace-text`. Post-rewrite
  verification: zero matches for all audit patterns (name, usernames,
  emails, home paths, LAN/tailnet IPs, hostnames) across every ref;
  `gitleaks` clean over all commits. The GitHub repo was deleted and
  recreated so no pre-rewrite object is reachable by SHA, PR ref, or
  force-push timeline event.
- **CI:** `ci.yml` (pytest matrix 3.12/3.13 + pyright strict-on-`tbot/`) and
  `secrets-scan.yml` (gitleaks full-history + private-pattern PII job that
  no-ops without the org secret).
- **LICENSE (MIT), CONTRIBUTING.md.**

## At/after the visibility flip

Free-tier org repos only get branch protection once public — apply then:

1. **Apply the main ruleset:**
   ```
   gh api repos/robotrocketscience/trajectory-bot/rulesets \
     --method POST --input .github/rulesets/main-protection.json
   ```
2. **Enable secret scanning + push protection** (Settings → Code security),
   and Dependabot alerts.
3. **Set the PII pattern secret** (`PII_PATTERNS_SECRET`, base64 pattern
   list, same value as the aelfrice staging gate) so `pii-pattern-scan`
   arms itself.
4. **Verify Actions are green on main** before announcing; the status checks
   in the ruleset must match the job names (`tests (3.12)`, `tests (3.13)`,
   `pyright`, `gitleaks`).
5. **Website cross-link** once the writeup page is live.

## Claims discipline (for the README and the website writeup)

- Success/fuel numbers are quoted from the *fresh 4096-episode* evaluation
  set, never in-run telemetry (winner's-curse).
- Fuel-efficiency comparisons must survive `scripts/verify_probe.py`
  (float64, dt=1 s, clean latches, exact-circularization AND tolerance-box
  baselines). Current verified position: the best-fuel policy MATCHES the
  analytic transfer within ~3% median at ~92% success; it does not beat it.
- Capability scope: policies are trained for circularize-at-apoapsis with
  target ≡ initial apoapsis (rt-generalization is an open research lane).
