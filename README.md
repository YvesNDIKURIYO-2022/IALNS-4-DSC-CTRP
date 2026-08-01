# IALNS-ESV for DSC-CTRP

**Improved Adaptive Large Neighborhood Search with Embedded Stability Verification**  
for the **Dynamic Stability-Constrained Container-Tanker Routing Problem**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![CPLEX](https://img.shields.io/badge/CPLEX-22.1.0-red.svg)](https://www.ibm.com/analytics/cplex-optimizer)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

---

## 📖 Overview

Liquid bulk transportation via multi-compartment tankers presents a critical operational challenge: balancing economic efficiency with physical safety throughout dynamic delivery routes. Existing vehicle routing models either ignore loading constraints entirely or treat them as static post-optimization checks, leading to solutions that are economically optimal but operationally infeasible or unsafe.

This repository provides the complete implementation of the **Improved Adaptive Large Neighborhood Search with Embedded Stability Verification (IALNS-ESV)** algorithm for solving the **Dynamic Stability-Constrained Container-Tanker Routing Problem (DSC-CTRP)**.

### Key Contributions

| Contribution | Description |
|--------------|-------------|
| **Novel Problem Formulation** | DSC-CTRP integrating routing decisions with dynamic stability constraints |
| **MILP Model** | Three-tier constraint hierarchy (geometric, static LDD, dynamic slosh) |
| **IALNS-ESV Algorithm** | Master-slave metaheuristic with embedded stability verification |
| **Novel Operators** | Imbalance Removal (destroy) and Stability-Aware Insertion (repair) |
| **Two-Stage Verification** | Capacity screening + full SACA MILP/heuristic verification |
| **Feasibility Cache** | 98.5% cache hit rate, 60-80% reduction in redundant checks |
| **Benchmark Suite** | 139 instances across 8 categories (3-60 customers) |

---

## 📊 Experimental Results

### Overall Performance (139 Instances)

| Algorithm | Avg Cost | Fleet | LDD Viol (%) | Stab Margin (%) | CPU (s) |
|-----------|----------|-------|--------------|-----------------|---------|
| **IALNS-ESV (IRSO)** | **497.33** | **3.0** | **0.00** | **7.85** | **0.367** |
| MILP | 835.39 | 7.0 | 0.00 | 15.12 | 0.049 |
| ALNS (No Stability) | 867.68 | 4.0 | 87.62 | 68.35 | 0.050 |
| ALNS (With LDD) | 1,236.57 | 6.0 | 0.00 | 3.00 | 0.262 |
| Sequential | 1,517.53 | 8.0 | 0.00 | 5.77 | 0.064 |

> **Cost reductions:** 69.31% vs Sequential, 61.82% vs ALNS (LDD), 45.59% vs ALNS (No Stability)

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- IBM ILOG CPLEX 22.1.0 or higher
- MiKTeX or TeX Live (for manuscript compilation)

### Setup

```bash
# Clone the repository
git clone https://github.com/username/IALNS-ESV-for-DSC-CTRP.git
cd IALNS-ESV-for-DSC-CTRP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
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
from src.alns import IALNS_ESV
from src.benchmarks import load_instance

# Load a benchmark instance
instance = load_instance("M_1_R25_random")

# Create solver
solver = IALNS_ESV(instance)

# Solve
solution = solver.solve()

# Results
print(f"Total Cost: {solution.cost:.2f}")
print(f"Routes: {solution.routes}")
print(f"Stability Margin: {solution.stability_margin:.2f}%")
print(f"LDD Violations: {solution.ldd_violations}")
print(f"CPU Time: {solution.cpu_time:.3f}s")
```

---

## 📁 Repository Structure

```
IALNS-ESV-for-DSC-CTRP/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── alns/                    # ALNS Master Module
│   │   ├── __init__.py
│   │   ├── master.py            # Main ALNS loop
│   │   ├── destroy_operators.py # Random, Worst-Cost, Compatibility, Imbalance
│   │   └── repair_operators.py  # Greedy, Regret, Stability-Aware
│   ├── stability/               # SACA Slave Module
│   │   ├── __init__.py
│   │   ├── saca_milp.py         # Exact MILP solver (n ≤ 9)
│   │   ├── saca_heuristic.py    # Heuristic solver (n > 9)
│   │   └── ldd_verification.py  # LDD constraint checker
│   ├── cache/                   # Feasibility Cache
│   │   ├── __init__.py
│   │   └── feasibility_cache.py
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── instance_loader.py
│       └── metrics.py
│
├── benchmarks/                  # Benchmark instances
│   ├── instances/
│   │   ├── small/               # n = 3-5
│   │   ├── medium/              # n = 7-9
│   │   ├── large/               # n = 10-25
│   │   └── real_world/          # n = 25-60
│   └── results/
│       └── experimental_results.csv
│
├── manuscript/                  # LaTeX manuscript
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
└── docs/                        # Documentation
```

---

## 📚 Benchmark Instances

| Category | Customers | Products | Compartments | Instances |
|----------|-----------|----------|--------------|-----------|
| Small (S) | 3 | 3 | 7 | 10 |
| Medium (M) | 5 | 3 | 7 | 10 |
| Large (L) | 7 | 3 | 7 | 5 |
| Extra-Large (EL) | 9 | 3 | 7 | 5 |
| Routing Small | 3-5 | 3 | 7 | 15 |
| Routing Medium | 5-10 | 3 | 7 | 30 |
| Routing Large | 10-25 | 3 | 7 | 48 |
| Real-World | 25-60 | 3 | 7 | 16 |

---

## 🧠 Algorithm Architecture

The IALNS-ESV framework follows a master-slave architectural pattern:

```
┌──────────────────────────────────────────────────────────────┐
│                    START: Generate Initial                   │
│                   Stability-Aware Solution                   │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              Master Module: ALNS Routing Search              │
│  ┌──────────────────────┐    ┌──────────────────────────┐   │
│  │   Destroy Operators   │    │    Repair Operators      │   │
│  │  • Random             │ →  │  • Greedy               │   │
│  │  • Worst-Cost         │    │  • Regret               │   │
│  │  • Compatibility      │    │  • Stability-Aware      │   │
│  │  • Imbalance          │    │    Insertion            │   │
│  └──────────────────────┘    └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   Feasibility Cache Check                    │
│              YES ──────────────────┐                         │
│               NO                   │                         │
│               ▼                    ▼                         │
│  ┌─────────────────────────┐ ┌──────────────────────────┐   │
│  │  Slave Module:           │ │  Update Global Optimal   │   │
│  │  Two-Stage Verification  │ │  Solution                │   │
│  │  Stage 1: Capacity &     │ └──────────────────────────┘   │
│  │  Volume Screening        │                               │
│  │  Stage 2: SACA MILP/     │                               │
│  │  Heuristic              │                               │
│  └─────────────────────────┘                               │
└──────────────────────────────────────────────────────────────┘
                              ▼
          ┌─────────────────────────────────────┐
          │       Termination Criterion?         │
          │    YES → Output Final Route Scheme   │
          │    NO → Return to ALNS Main Loop     │
          └─────────────────────────────────────┘
```

---

## 📝 Constraint Hierarchy

| Tier | Constraints | Description |
|------|-------------|-------------|
| **1** | Geometric & Capacity | Routing flow, compartment capacity, product compatibility |
| **2** | Static LDD | Axle load bounds, center-of-gravity limits |
| **3** | Dynamic Sloshing | Fill ratio avoidance [0.35, 0.55] + physical validation |

---

## 📄 Citation

If you use this code, data, or benchmarks in your research, please cite:

```bibtex
@article{Ndikuriyo2026,
  title={Container Route Optimization with Multi-Compartment Liquid Loading 
         and Dynamic Stability Constraints: An Improved ALNS Framework 
         with Embedded Stability Verification},
  author={Ndikuriyo, Yves and Zhang, Yinggui},
  journal={Transportation Research Part E: Logistics and Transportation Review},
  volume={xxx},
  pages={xxx-xxx},
  year={2026},
  doi={10.1016/j.tre.2026.xxxxxx}
}
```

---

## 📬 Contact

| Author | Role | Email |
|--------|------|-------|
| Yves Ndikuriyo | First Author | yvesndikuriyo@csu.edu.cn|yves.ndikuriyo@outlook.fr |
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

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**⭐ Star this repository if you find it useful!**
```
