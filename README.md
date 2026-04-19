# PID Tank Level Control System
### Baku Higher Oil School — Process Control Course Project

> **Automatic Tank Level Control** using a Siemens S7-1500 PLC, Factory I/O digital twin, WinCC HMI, Python data logging, and a Random Forest predictive maintenance engine.
  
**Supervisor:** Nijat Hasanov  
**Date:** April 2026

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware & Software Stack](#hardware--software-stack)
- [PLC Logic & Control Design](#plc-logic--control-design)
- [Python Data Logger](#python-data-logger)
- [Predictive Maintenance](#predictive-maintenance)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Manual valve operation in industrial fluid handling is neither precise nor inherently safe. This project replaces manual control with a fully automated, closed-loop **PID feedback system** that maintains a liquid tank at any operator-specified level between 0 and 300 cm.

The system is built entirely within a **digital twin** environment — no physical hardware required — while faithfully replicating real industrial practices: deterministic PLC scan cycles, hardwired safety trips, PROFINET networking, and a WinCC HMI panel for live operator interaction.

Beyond core control, the project extends into two advanced domains:

1. **Python Data Logging** — a real-time telemetry historian that records every PLC variable to structured CSV files via the `python-snap7` S7 protocol library.
2. **Predictive Maintenance** — a `scikit-learn` Random Forest classifier that analyses the logged data to detect valve leakage and sensor drift faults before they escalate.

---

## System Architecture

```
┌─────────────────┐      Raw Analog      ┌──────────────────────┐
│   Factory I/O   │ ──── Level Signal ──▶ │                      │
│  (Digital Twin) │                       │  Siemens S7-1500 PLC │
│                 │ ◀─── Valve Command ── │   (TIA Portal v17)   │
└─────────────────┘    (0–27648 counts)   └──────────┬───────────┘
                                                     │ PROFINET
                                          ┌──────────▼───────────┐
                                          │   WinCC HMI Panel    │
                                          │  SP / PV / OUT / ESD │
                                          └──────────────────────┘
                                                     │ S7 TCP/IP
                                          ┌──────────▼───────────┐
                                          │   Python Snap7 Client│
                                          │  (500 ms poll cycle) │
                                          └──────────┬───────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │     CSV Dataset      │
                                          │  → Random Forest PdM │
                                          └──────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| **PID Control** | `PID_Compact` in OB30 (500 ms cyclic interrupt), tunable Kc / Ti / Td via HMI |
| **Signal Scaling** | DAC Function Block: raw 0–27648 → NORM_X → SCALE_X → 0–300 cm |
| **Emergency Shutdown** | SR flip-flop ESD; auto-trips at ≥ 290 cm; disables & resets PID accumulator |
| **WinCC HMI** | Live SP / PV / OUT display, PID parameter inputs, trend graph, ESD button |
| **Python Logger** | Snap7 client polls MD100–MD124 at 500 ms; M20.0 rising/falling edge triggers CSV open/close |
| **Predictive Maintenance** | Random Forest (200 trees, balanced weights); detects valve leakage & sensor drift |

---

## Hardware & Software Stack

**Simulated Hardware**
- Siemens S7-1500 CPU 1511-1 PN/DP
- Ultrasonic level sensor (0–300 cm, 4–20 mA)
- Analog fill valve and discharge valve
- Operator panel (Start / Stop / E-Stop / Reset)

**Software**
- TIA Portal v17 (Ladder Diagram programming)
- SIMATIC WinCC RunTime Advanced
- Factory I/O (3D physics / digital twin engine)
- Python 3 + `python-snap7` (data historian)
- `scikit-learn` (Random Forest classifier)

---

## PLC Logic & Control Design

### Signal Conversion (DAC Function Block)

Raw Factory I/O integers (0 → 27648) are normalised and scaled in two steps:

```
Normalized = (Raw − 0) / (27648 − 0)          [NORM_X]
Engineering = (Normalized × (300 − 0)) + 0     [SCALE_X]
```

### PID Loop

The `PID_Compact` technology object runs inside **OB30** (500 ms cyclic interrupt), ensuring deterministic execution twice per second. The standard parallel PID formula:

```
u(t) = Kc·e(t)  +  (Kc/Ti)·∫e(t)dt  +  Kc·Td·(de/dt)
```

The computed output percentage is written directly to `%QW30` (fill valve analog word).

### Safety Trip System

- **SR Flip-Flop ESD** — latches immediately on Stop or Emergency button press.
- **High-Level Hardware Trip** — valve forced to 0 % when tank ≥ 290 cm, independent of PID state.
- **PID Disable on Trip** — the PID accumulator is cleared on shutdown to prevent integral-windup spikes on restart.
- **Reset Path** — operator must acknowledge and press Reset before normal control resumes.

---

## Python Data Logger

The logger acts as an external data historian sitting alongside the PLC simulation.

```python
# Address map — extend without touching the poll loop
TAG_MAP = [
    (M_AREA, 0, 100, "REAL", "Setpoint_%"),
    (M_AREA, 0, 104, "REAL", "Level_PV_%"),
    (M_AREA, 0, 108, "REAL", "PID_Output_%"),
    (M_AREA, 0, 112, "REAL", "Error"),
    # ... P_Term, I_Term, D_Term
]

def bytes_to_real(raw: bytes) -> float:
    """4-byte big-endian S7 REAL → Python float."""
    return struct.unpack(">f", raw)[0]
```

**Recording trigger:** rising edge on `M20.0` (the HMI Start bit) opens a new timestamped CSV; the falling edge closes it cleanly. `csv_file.flush()` is called every cycle so data survives a crash.

---

## Predictive Maintenance

### Problem: Healthy-Bias Dataset

~2,000 samples logged from fault-free operation gave the classifier no failure examples to learn from. A Python digital-twin simulation was used to inject two synthetic fault modes:

- **Valve Leakage** — fill valve passes fluid at 0 % command → erratic PID output
- **Sensor Drift** — progressive ultrasonic offset → PV diverges from true level

### Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # handles residual class imbalance
    random_state=42
)
model.fit(X_train, y_train)
```

### Results

| Metric | Score |
|---|---|
| Precision | **0.93** |
| Recall | **0.90** |
| F1-Score | **0.91** |

The confusion matrix on the unseen test set: 1,500 true-healthy, 1,600 true-fault, 120 false alarms, 180 missed faults. The high recall (0.90) is critical — in a safety context, a missed fault is always more costly than a false alarm.

---

## Results

The system was validated across three setpoints under P-only control:

| Setpoint | Steady-State PV | Steady-State Error |
|---|---|---|
| 50 cm | 50.00 cm | 0 cm |
| 100 cm | 100.00 cm | 0 cm |
| 150 cm | 150.00 cm | 0 cm |

Emergency shutdown and the 290 cm hardware trip both functioned correctly in all fault-injection tests.

---

## Limitations

- **Integral Windup** — low Ti values caused ~10–12 % transient error during PI testing.
- **Derivative Kick** — D-on-PV (rather than D-on-error) was required to prevent aggressive valve spikes.
- **Double Integrator** — the tank is a naturally integrating process; adding a software integral term can push phase beyond −180°, causing instability.
- **Synthetic Fault Data** — the PdM classifier was trained on simulated faults; real-world degradation is more gradual and stochastic, requiring retraining on empirical data before physical deployment.

---

## Future Work

- **2oo3 Voting Safety** — three redundant sensors with two-out-of-three voting for the hardware trip.
- **Online PdM Inference** — stream each new CSV row to the classifier in real time and raise an HMI alarm on fault detection.
- **Expanded Fault Library** — add pump cavitation, pipe blockage, and actuator stiction to improve diagnostic coverage.
- **Multi-Loop Architecture** — distribute parallel PID branches with per-loop auto-tuning options.

---

## References

1. Ahmed, S. (2020). *PID Control utilizing TIA Portal methodologies.* https://www.youtube.com/watch?v=cMUmANsi5gw
2. GeeksforGeeks. (2023). *Data Analysis: Normalization and Scaling.* https://www.geeksforgeeks.org/data-analysis/normalization-and-scaling/
3. Control Station. *Options for PID Controller Tuning.* https://controlstation.com/options-exist-pid-controller-tuning/
4. Control Station. *Basics of Control Loop Optimization.* https://controlstation.com/control-loop-optimization/
5. python-snap7 Project. *python-snap7 Documentation.* https://python-snap7.readthedocs.io/
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

---

## About This Project

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██████╗ ██╗██████╗      ████████╗ █████╗ ███╗   ██╗██╗  ██╗  ║
║   ██╔══██╗██║██╔══██╗        ██╔══╝██╔══██╗████╗  ██║██║ ██╔╝  ║
║   ██████╔╝██║██║  ██║        ██║   ███████║██╔██╗ ██║█████╔╝   ║
║   ██╔═══╝ ██║██║  ██║        ██║   ██╔══██║██║╚██╗██║██╔═██╗   ║
║   ██║     ██║██████╔╝        ██║   ██║  ██║██║ ╚████║██║  ██╗  ║
║   ╚═╝     ╚═╝╚═════╝         ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝  ║
║                                                                  ║
║         Automatic Tank Level Control — BHOS 2026                ║
╚══════════════════════════════════════════════════════════════════╝
```

This project was developed as part of the **Process Control** course at **Baku Higher Oil School**, Department of Process Automation Engineering. It demonstrates how classical control theory, modern PLC programming, industrial simulation, and machine learning can be brought together into a single cohesive engineering workflow.

**What makes it interesting:**

- 🏭 **Real industrial toolchain** — TIA Portal, WinCC, PROFINET, and Factory I/O are the same tools used in live oil & gas facilities.
- 🔗 **Bridging OT and IT** — `python-snap7` breaks the boundary between the PLC world and the Python data-science ecosystem without modifying a single line of PLC code.
- 🤖 **AI on top of control** — the Random Forest layer shows how a digital twin can double as a fault-data generator, solving the cold-start problem that plagues industrial ML deployments.
- 🛡️ **Safety-first design** — the SR flip-flop ESD, PID disable-on-trip, and hardwired 290 cm limit reflect real functional-safety principles, not just classroom exercises.

```
Siemens S7-1500  ──►  Factory I/O  ──►  Python Snap7  ──►  CSV  ──►  Random Forest
     PLC               Digital Twin       Data Logger       Data       PdM Classifier
   (OT Layer)         (Simulation)        (IT Bridge)    (History)    (AI Layer)
```

> Built with ❤️ in Baku, Azerbaijan.  
