Here's a contributing guideline for the repo:

---

# Contributing Guidelines

Thank you for your interest in contributing to the **PID Tank Level Control & Predictive Maintenance** project! Whether you're fixing a bug, improving the ML pipeline, enhancing PLC logic, or adding documentation — all contributions are welcome.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Areas We'd Love Help With](#areas-wed-love-help-with)

---

## Code of Conduct

This project follows a simple rule: **be respectful and constructive**. Harassment, dismissive language, or unconstructive criticism will not be tolerated. This is a student engineering project — we welcome contributors of all experience levels.

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Digital-Twin-Predictive-Maintenance---PID-Tank-Level-Control-with-TIA-Portal-and-Factory-I-O.git
   cd Digital-Twin-Predictive-Maintenance---PID-Tank-Level-Control-with-TIA-Portal-and-Factory-I-O
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## How to Contribute

There are several ways to contribute:

| Type | Examples |
|---|---|
| 🐛 Bug fix | Fix incorrect scaling logic, broken CSV logging, model prediction errors |
| ✨ New feature | Add a new fault type, extend the HMI, implement online inference |
| 📖 Documentation | Improve README, add inline comments, write tutorials |
| 🧪 Testing | Add unit tests for the Python pipeline or ML model |
| 🗂️ Data | Contribute real or higher-fidelity synthetic fault datasets |
| 🔧 Tooling | CI/CD setup, linting config, dependency management |

---

## Development Setup

### Python Environment

Requires **Python 3.8+**.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install python-snap7 scikit-learn pandas numpy matplotlib
```

### TIA Portal / Factory I/O

- **TIA Portal v17** is required to open or modify the `.ap17` PLC project file.
- **Factory I/O** is required to open the `.factoryio` scene file.
- If you are only contributing to the Python ML pipeline, neither tool is required.

### Running the ML Pipeline

```bash
python pid_ml_pipeline.py
```

This will train the Random Forest classifier on `expanded_dataset.csv` and output model metrics and figures.

---

## Project Structure

```
├── FactoryIO_Template_S7-1500_V15_V17.ap17   # TIA Portal PLC project
├── PID_Tank_Level.factoryio                  # Factory I/O scene
├── pid_ml_pipeline.py                        # Python ML training pipeline
├── expanded_dataset.csv                      # Training/test dataset
├── fig1_dataset_overview.png                 # Generated figures
├── fig2_confusion_matrices.png
├── fig3_roc_curves.png
├── fig4_feature_importance.png
├── fig5_model_comparison.png
├── fig6_online_predictions.png
├── PID_Tank_Level_Control_Report_UPDATED.pdf # Project report
└── README.md
```

---

## Coding Standards

### Python

- Follow **PEP 8** style conventions.
- Use **descriptive variable names** — avoid single-letter names outside of loop indices.
- Add **docstrings** to all functions:
  ```python
  def bytes_to_real(raw: bytes) -> float:
      """Convert a 4-byte big-endian S7 REAL to a Python float."""
      return struct.unpack(">f", raw)[0]
  ```
- Keep functions **focused and small** — one responsibility per function.
- Do not commit **hardcoded IP addresses or credentials**; use environment variables or a config file.

### PLC / TIA Portal

- Follow Siemens naming conventions for tags and blocks (e.g., `DB1.Level_PV`, `OB30_PID_Cycle`).
- Comment all Function Blocks and ladder rungs clearly.
- Do not change the PROFINET device names or I/O addresses without updating the README accordingly.

### Data & CSV

- Any new dataset columns must be documented in the README with units and description.
- Do not commit large raw data files (> 10 MB); link to external storage instead.

---

## Commit Message Guidelines

Use the following format:

```
<type>(<scope>): <short summary>
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `refactor` | Code restructuring without behaviour change |
| `data` | Dataset additions or modifications |
| `test` | Adding or updating tests |
| `chore` | Tooling, CI, dependency updates |

**Examples:**

```
feat(ml): add SVM classifier as an alternative to Random Forest
fix(logger): handle snap7 connection timeout gracefully
docs(readme): clarify DAC scaling formula
data: add pump cavitation fault samples to expanded_dataset.csv
```

---

## Pull Request Process

1. Ensure your branch is **up to date** with `main` before opening a PR:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. **Describe your changes** clearly in the PR description — what problem it solves, how it was tested.
3. If your PR touches the ML pipeline, include **before/after metrics** (precision, recall, F1).
4. If your PR touches PLC logic, describe the change in plain language — not everyone has TIA Portal.
5. Keep PRs **focused** — one logical change per PR makes review faster.
6. A maintainer will review and may request changes. Please respond constructively to feedback.
7. Once approved, your PR will be **squash-merged** into `main`.

---

## Reporting Bugs

Open a GitHub Issue with the following information:

- **Environment** — OS, Python version, TIA Portal/Factory I/O version if relevant.
- **Steps to reproduce** — be as specific as possible.
- **Expected behaviour** vs **actual behaviour**.
- **Logs or screenshots** if available (especially for snap7 connection errors or incorrect model outputs).

---

## Suggesting Enhancements

Open a GitHub Issue with the label `enhancement` and include:

- A clear description of the proposed feature and the problem it solves.
- Any relevant references, papers, or prior art.
- Whether you'd be willing to implement it yourself.

---

## Areas We'd Love Help With

These are open items from the project's own Future Work section — ideal starting points for contributors:

- **Online PdM inference** — stream each new CSV row to the classifier in real time and raise an HMI alarm on fault detection.
- **Expanded fault library** — add pump cavitation, pipe blockage, and actuator stiction fault modes to the dataset and retrain.
- **2oo3 voting safety** — implement three-sensor redundancy with two-out-of-three voting logic.
- **Auto-tuning** — implement relay feedback or Ziegler–Nichols auto-tuning for the PID parameters.
- **Unit tests** — add `pytest` tests for the data logging and ML pipeline modules.
- **CI pipeline** — set up GitHub Actions to lint Python code and run tests on every push.

---

*Built with ❤️ at Baku Higher Oil School — contributions from anywhere in the world are welcome.*
