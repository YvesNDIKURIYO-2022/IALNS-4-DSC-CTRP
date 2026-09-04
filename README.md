# IALNS for DSC-CTRP

**Improved Adaptive Large Neighborhood Search**  
for the **Dynamic Stability-Constrained Container-Tanker Routing Problem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CPLEX](https://img.shields.io/badge/CPLEX-22.1.0-red.svg)](https://www.ibm.com/analytics/cplex-optimizer)

---

## 📖 Overview

Liquid bulk transportation via multi-compartment tankers presents a critical operational challenge: balancing economic efficiency with physical safety throughout dynamic delivery routes. Existing vehicle routing models either ignore loading constraints entirely or treat them as static post-optimization checks, leading to solutions that are economically optimal but operationally infeasible or unsafe.

This repository provides the complete implementation of the **Improved Adaptive Large Neighborhood Search (IALNS)** algorithm for solving the **Dynamic Stability-Constrained Container-Tanker Routing Problem (DSC-CTRP)**.

### Key Contributions

| Contribution | Description |
|--------------|-------------|
| **Novel Problem Formulation** | DSC-CTRP integrating routing decisions with dynamic stability constraints |
| **MILP Model** | Three-tier constraint hierarchy (geometric, static LDD, dynamic anti-sloshing) |
| **IALNS Algorithm** | Master-slave metaheuristic with embedded stability verification |
| **Novel Operators** | Imbalance Removal (destroy) and Stability-Aware Insertion (repair) |
| **Two-Stage Verification** | Capacity screening + exact MILP or SACA heuristic verification |
| **Feasibility Cache** | 88.26% hit rate, eliminating 87.2% of MILP calls |
| **Benchmark Suite** | 281 instances across 2 established MCVRP suites (3-100 customers) |

---

## 📊 Experimental Results

### Comprehensive State-of-the-Art Performance (281 Instances)

| Algorithm | Avg Cost ($) | Std Cost | LDD Violation (%) | CG Compliance (%) | CPU (s) |
|-----------|--------------|----------|-------------------|-------------------|---------|
| **IALNS** | **419.99** | 115.94 | **0.00** | **100.00** | 0.011 |
| Standard ALNS | 395.48 | 85.55 | 0.00 | 91.70 | 0.003 |
| FDAHS | 2,127.91 | 1,425.42 | 0.00 | 91.70 | 0.006 |
| GCOF | 2,127.91 | 1,425.42 | 0.00 | 91.70 | 0.082 |
| HSA | 2,127.91 | 1,425.42 | 0.00 | 91.70 | 0.002 |
| ACO-2opt | 2,127.91 | 1,425.42 | 0.00 | 91.70 | 0.302 |

> **Key Results:** IALNS achieves **100% CG compliance** (8.3% above Standard ALNS), **0% LDD violations**, and **100% stability margins**. The feasibility cache achieves **88.26% hit rate**, eliminating **87.2% of MILP calls** with near-linear runtime scaling (0.008–0.014 s for up to 20 customers).

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- IBM ILOG CPLEX 22.1.0 or higher
- MiKTeX or TeX Live (for manuscript compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/IALNS-4-DSC-CTRP.git
cd IALNS-4-DSC-CTRP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
click>=8.0.0
tqdm>=4.62.0
pyyaml>=5.4.0
joblib>=1.1.0
```

---

## 🚀 Quick Start

```python
from src.alns import IALNS
from src.benchmarks import load_instance

# Load a benchmark instance
instance = load_instance("EL_1")

# Create solver
solver = IALNS(instance)

# Solve
solution = solver.solve()

# Results
print(f"Total Cost: {solution.cost:.2f}")
print(f"CG Compliance: {solution.cg_compliance:.2f}%")
print(f"LDD Violations: {solution.ldd_violations}")
print(f"Stability Margin: {solution.stability_margin:.2f}%")
print(f"CPU Time: {solution.cpu_time:.3f}s")
```

---

## 📁 Repository Structure

```
IALNS-4-DSC-CTRP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── alns/                    # ALNS Master Module
│   │   ├── master.py            # Main ALNS loop
│   │   ├── destroy_operators.py # Random, Worst-Cost, Compatibility, Imbalance
│   │   └── repair_operators.py  # Greedy, Regret, Stability-Aware
│   ├── stability/               # SACA Slave Module
│   │   ├── saca_milp.py         # Exact MILP solver (|R_k| ≤ 9)
│   │   ├── saca_heuristic.py    # Heuristic solver (|R_k| > 9)
│   │   └── ldd_verification.py  # LDD constraint checker
│   ├── cache/                   # Feasibility Cache
│   │   └── feasibility_cache.py # LRU cache with route-invariant keys
│   └── utils/                   # Utilities
│       ├── instance_loader.py   # JSON instance loader
│       └── metrics.py           # Performance metrics
│
├── benchmarks/                  # Benchmark instances (281 total)
│   ├── paixao/                  # 40 instances (3-9 customers, 3 commodities)
│   │   ├── S_1--S_10            # 3 customers
│   │   ├── M_1--M_10            # 5 customers
│   │   ├── L_1--L_10            # 7 customers
│   │   └── EL_1--EL_10          # 9 customers
│   └── mirzaei_muyldermans/     # 241 instances (10-100 customers, 2-6 commodities)
│
├── manuscript/                  # LaTeX manuscript
│   ├── Manuscript-IALNS.tex
│   ├── references.bib
│   └── figures/                 # All figures (PNG, PDF)
│
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

---

## 📚 Benchmark Instances

The computational study evaluates **281 benchmark instances** from two established MCVRP suites:

| Suite | Instances | Customers | Commodities | Purpose |
|-------|-----------|-----------|-------------|---------|
| Paixão et al. (2026) | 40 | 3–9 | 3 | LDD & stability validation |
| Mirzaei & Wøhlk (2019) + Muyldermans & Pang (2010) | 241 | 10–100 | 2–6 | Scalability testing |
| **Total** | **281** | **3–100** | **2–6** | **Full evaluation** |

All instances are stored in standardized JSON format and are publicly available.

---

## 🧠 Algorithm Architecture

The IALNS framework follows a master-slave architectural pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    START: Generate Initial                      │
│                   Stability-Aware Solution                      │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Master Module: ALNS Routing Search                 │
│  ┌────────────────────────┐   ┌──────────────────────────────┐ │
│  │   Destroy Operators     │   │    Repair Operators          │ │
│  │  • Random               │ → │  • Greedy                   │ │
│  │  • Worst-Cost           │   │  • Regret-k                 │ │
│  │  • Compatibility        │   │  • Stability-Aware          │ │
│  │  • Imbalance            │   │    Insertion                │ │
│  └────────────────────────┘   └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feasibility Cache Check                       │
│              YES ────────────────────────┐                     │
│               NO                         │                     │
│               ▼                          ▼                     │
│  ┌───────────────────────────┐ ┌────────────────────────────┐  │
│  │  Slave Module: SACA       │ │  Update Best Solution       │  │
│  │  Two-Stage Verification   │ │                            │  │
│  │  Stage 1: Capacity        │ └────────────────────────────┘  │
│  │  & Volume Screening       │                                │
│  │  Stage 2: SACA-MILP       │                                │
│  │  (|R_k| ≤ 9) or           │                                │
│  │  h-SACA (|R_k| > 9)       │                                │
│  └───────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                              ▼
          ┌─────────────────────────────────────┐
          │       Termination Criterion?         │
          │    YES → Output Final Route Scheme   │
          │    NO → Return to ALNS Main Loop     │
          └─────────────────────────────────────┘
```

---

## 📝 Constraint Hierarchy

| Tier | Constraints | Verification Type |
|------|-------------|-------------------|
| **1** | Geometric & Capacity | Routing flow, compartment capacity, product compatibility |
| **2** | Static LDD Stability | Axle load bounds (Eqs. 11–15), center-of-gravity limits |
| **3** | Dynamic Anti-Sloshing | Fill ratio avoidance [0.35, 0.55] (Eqs. 16–17) + physical validation |

---

## 📊 Performance Metrics

| Category | Metrics |
|----------|---------|
| **Economic Efficiency** | Total Distance, Transportation Cost, Fleet Utilization, Volume Utilization |
| **Safety Compliance** | LDD Violation Rate, CG Compliance Rate, Stability Margin, Slosh Risk Score |
| **Computational Performance** | CPU Time, Cache Hit Rate, MILP Calls, Iterations to Convergence |

---

## 📄 Citation

If you use this code, data, or benchmarks in your research, please cite:

```bibtex
@article{Ndikuriyo2026,
  title={An Improved Adaptive Large Neighborhood Search for Multi-Compartment 
         Container-Tanker Routing with Dynamic Stability Constraints},
  author={Ndikuriyo, Yves and Zhang, Yinggui},
  journal={European Journal of Operational Research},
  year={2026},
  note={Under Review}
}
```

---

## 📬 Contact

| Author | Role | Email |
|--------|------|-------|
| Yves Ndikuriyo | First Author | yvesndikuriyo@csu.edu.cn |
| Yinggui Zhang | Corresponding Author | ygzhang@csu.edu.cn |

**Affiliation:**  
School of Traffic and Transportation Engineering  
Central South University  
Changsha, Hunan 410075, China

---

## 🙏 Acknowledgments

This work was supported by:

- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province (Grant Nos. 2023JJ30710 and 2022JJ31020)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please submit issues and pull requests through the GitHub repository.

---

**⭐ Star this repository if you find it useful!**
```
