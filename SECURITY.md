# Security Policy

## Supported Versions

This is an academic project developed at Baku Higher Oil School. The following components are actively maintained:

| Component | Status |
|---|---|
| `pid_ml_pipeline.py` | ✅ Maintained |
| `expanded_dataset.csv` | ✅ Maintained |
| TIA Portal project (`.ap17`) | ⚠️ Best-effort |
| Factory I/O scene (`.factoryio`) | ⚠️ Best-effort |
| PDF report | 🔒 Frozen / read-only |

---

## Scope

This repository contains:

- **PLC project files** — intended for use in isolated simulation environments (TIA Portal + Factory I/O). They are **not designed for deployment on live industrial hardware** without a full safety review.
- **Python scripts** — connect to a simulated S7-1500 PLC via `python-snap7` over a local network. They are **not intended to connect to production PLCs**.
- **Machine learning pipeline** — trained on synthetic fault data. It is **not certified for safety-critical decision making**.

> ⚠️ **Do not deploy any part of this project to a physical PLC, live control system, or operational plant without independent safety validation and compliance review.**

---

## Reporting a Vulnerability

We take security seriously even in an academic context — especially because industrial control system (ICS) code can cause real harm if misused.

### What to report

Please report any of the following:

- Hardcoded credentials, IP addresses, or secrets accidentally committed to the repository.
- Unsafe use of `python-snap7` that could allow unintended write access to a PLC.
- ML model behaviour that could produce dangerously misleading maintenance predictions.
- Dependency vulnerabilities in the Python stack (`scikit-learn`, `python-snap7`, `pandas`, etc.).
- Any code pattern that could be exploited if this project were adapted for real hardware.

### What NOT to report here

- Bugs unrelated to security (use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template instead).
- General PLC programming questions.
- Factory I/O or TIA Portal software vulnerabilities — report those directly to their respective vendors.

---

## How to Report

**Please do not open a public GitHub Issue for security vulnerabilities.**

Instead, report privately using one of the following methods:

1. **GitHub Private Vulnerability Reporting** *(preferred)*
   Navigate to the **Security** tab of this repository → **Advisories** → **Report a vulnerability**.

2. **Email**
   Send a detailed report to the project maintainer. Include:
   - A clear description of the vulnerability.
   - Steps to reproduce or a proof-of-concept (where safe to share).
   - The potential impact if exploited.
   - Any suggested remediation.

We will acknowledge your report within **5 business days** and aim to resolve confirmed vulnerabilities within **30 days**.

---

## Responsible Disclosure

We follow a **coordinated disclosure** approach:

1. Reporter submits vulnerability privately.
2. Maintainers confirm receipt and assess severity within 5 days.
3. A fix is developed and tested.
4. The fix is released and the reporter is credited (unless they prefer anonymity).
5. A public security advisory is published after the fix is available.

---

## ICS / OT-Specific Considerations

Because this project involves industrial control system tooling, contributors and users should be aware of the following:

- **Network isolation** — when running the simulation, keep the TIA Portal / Factory I/O environment on an isolated network or loopback interface. Do not expose the S7 TCP port (102) to the internet.
- **Snap7 write access** — `python-snap7` can read *and* write PLC memory. Audit any modifications to `pid_ml_pipeline.py` or the logger to ensure only intended memory areas are accessed.
- **Synthetic data risk** — the ML model was trained on synthetic faults. Overconfidence in its predictions in a real environment could mask genuine equipment degradation.
- **No authentication by default** — the S7 protocol used here does not require authentication in simulation mode. This is acceptable for a local digital twin but would be unacceptable in any networked deployment.

---

## Dependency Security

To check for known vulnerabilities in Python dependencies:

```bash
pip install pip-audit
pip-audit
```

Please open a regular Issue if you find an outdated or vulnerable dependency so it can be updated promptly.

---

*This policy was last updated: April 2026.*
