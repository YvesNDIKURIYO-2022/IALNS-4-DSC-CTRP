"""
DSC-CTRP Standard Instance Generator
=====================================

Generates the standard benchmark instances used for the
Dynamic Stability-Constrained Container-Tanker Routing Problem
(DSC-CTRP).

The generator reproduces two layers:

  * Paixao layer  -- 15 instances (P20_1--P20_5, P25_1--P25_5,
                    P30_1--P30_5), 20--30 customers, 3 products.
  * Mirzaei layer -- 85 instances (vrpnc1--vrpnc15, extra33--extra87),
                    20--100 customers, 2--6 products.

Total: 100 instances. Of these, 30 (20--25 customers) are used for
the state-of-the-art comparison and the ablation study reported in
the manuscript; the remaining 70 are reserved for the scalability
study.

Author: Yves Ndikuriyo
Affiliation: Central South University
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np


# ============================================================================
# PATHS (portable; override with --output on the command line)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_instances")


# ============================================================================
# SOURCE METADATA
# ============================================================================

SOURCE_INFO = {
    "paixao": {
        "name": "Paixao et al. (2026) - LDD and stability validation",
        "reference": "Paixao, R., Soares, A., Ramos, A. G., & Silva, E. (2026)",
        "doi": "10.1016/j.apm.2026.116860",
    },
    "mirzaei": {
        "name": "Mirzaei & Wohlk (2019) - Large-scale MCVRP",
        "reference": "Mirzaei, S., & Wohlk, S. (2019)",
        "doi": "10.1007/s13676-016-0096-x",
    },
}


# ============================================================================
# VEHICLE CONFIGURATION
# ============================================================================

VEHICLE_CONFIG = {
    "front_axle_max": 10076,
    "rear_axle_max": 24000,
    "wheelbase": 7.70,
    "unladen_weight": 5900,
    "max_payload": 30176,
    "max_volume": 38000,
    "steering_min_ratio": 10,
    "driving_min_ratio": 25,
    "fixed_cost": 100.0,
    "cost_per_km": 1.0,
}

DEFAULT_COMPARTMENTS_7 = [
    {"id": 0, "capacity": 5000, "position_x": -4.5, "position_y": 0.0},
    {"id": 1, "capacity": 5000, "position_x": -3.5, "position_y": 0.0},
    {"id": 2, "capacity": 5000, "position_x": -2.5, "position_y": 0.0},
    {"id": 3, "capacity": 5750, "position_x": -1.5, "position_y": 0.0},
    {"id": 4, "capacity": 5750, "position_x": 0.5, "position_y": 0.0},
    {"id": 5, "capacity": 5750, "position_x": 1.5, "position_y": 0.0},
    {"id": 6, "capacity": 5750, "position_x": 2.5, "position_y": 0.0},
]


def get_product_densities(n_commodities: int) -> Dict[int, float]:
    """Return product densities for the given number of commodities."""
    if n_commodities <= 2:
        return {1: 0.72, 2: 0.84}
    if n_commodities == 3:
        return {1: 0.72, 2: 0.84, 3: 0.80}
    return {1: 0.72, 2: 0.84, 3: 0.80, 4: 0.76}


def create_compartments(n_compartments: int, total_capacity: float) -> List[Dict]:
    """Build a compartment configuration with the requested number of compartments."""
    if n_compartments == 2:
        return [
            {"id": 0, "capacity": total_capacity * 0.25, "position_x": -2.0, "position_y": 0.0},
            {"id": 1, "capacity": total_capacity * 0.75, "position_x": 2.0, "position_y": 0.0},
        ]
    if n_compartments == 7:
        return [c.copy() for c in DEFAULT_COMPARTMENTS_7]
    positions = np.linspace(-n_compartments / 2, n_compartments / 2, n_compartments)
    return [
        {"id": i, "capacity": total_capacity / n_compartments,
         "position_x": float(pos), "position_y": 0.0}
        for i, pos in enumerate(positions)
    ]


def compute_distance_matrix(coords: List[tuple]) -> List[List[float]]:
    """Compute the Euclidean distance matrix for the given coordinates."""
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                matrix[i][j] = round(float(np.sqrt(dx * dx + dy * dy)), 2)
    return matrix


# ============================================================================
# PAIXAO LAYER: 15 INSTANCES (P20, P25, P30)
# ============================================================================

class PaixaoInstanceGenerator:
    """Generate the Paixao layer used for LDD and stability validation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_all(self) -> List[Dict]:
        instances = []
        for n_customers in (20, 25, 30):
            for rep in range(1, 6):
                name = f"P{n_customers}_{rep}"
                instances.append(self._build_instance(name, n_customers, rep))
        return instances

    def _build_instance(self, name: str, n_customers: int, rep: int) -> Dict:
        """Build a single Paixao-layer instance with reproducible coordinates."""
        rng = random.Random(hash((name, rep)) & 0xFFFFFFFF)

        customers = []
        for cid in range(n_customers):
            x = rng.uniform(-35, 35)
            y = rng.uniform(-35, 35)
            demand = {
                1: rng.randint(3000, 5000),
                2: rng.randint(2500, 4000),
                3: rng.randint(1500, 3000),
            }
            customers.append({
                "id": cid,
                "x": x,
                "y": y,
                "demand": demand,
                "time_window": [0, 1000],
                "service_time": 10,
            })

        coords = [(0, 0)] + [(c["x"], c["y"]) for c in customers]
        vehicle = {
            "compartments": [c.copy() for c in DEFAULT_COMPARTMENTS_7],
            **VEHICLE_CONFIG,
            "product_densities": get_product_densities(3),
        }

        return {
            "name": name,
            "customers": customers,
            "time_windows": [[0.0, 1000.0, 10.0]] * n_customers,
            "vehicle": vehicle,
            "distance_matrix": compute_distance_matrix(coords),
            "metadata": {
                "source": SOURCE_INFO["paixao"]["name"],
                "source_doi": SOURCE_INFO["paixao"]["doi"],
                "layer": "paixao",
                "n_customers": n_customers,
                "n_commodities": 3,
                "n_compartments": 7,
                "generated": datetime.now().isoformat(),
            },
        }


# ============================================================================
# MIRZAEI LAYER: 85 INSTANCES (vrpnc1--vrpnc15, extra33--extra87)
# ============================================================================

class MirzaeiInstanceGenerator:
    """Generate the Mirzaei layer used for scalability testing."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def generate_all(self) -> List[Dict]:
        instances = []

        # vrpnc1--vrpnc15: 20--35 customers, 2--3 products
        for idx in range(1, 16):
            n_customers = 20 + (idx - 1) * 1  # 20, 21, ..., 34
            n_products = 2 if idx % 2 == 1 else 3
            name = f"vrpnc{idx}"
            instances.append(self._build_instance(
                name=name,
                n_customers=n_customers,
                n_products=n_products,
                seed=idx,
            ))

        # extra33--extra87: 36--100 customers, 2--6 products
        for idx, n_customers in enumerate(range(36, 101), start=33):
            n_products = 2 + (idx % 5)  # 2--6
            name = f"extra{idx}"
            instances.append(self._build_instance(
                name=name,
                n_customers=n_customers,
                n_products=n_products,
                seed=1000 + idx,
            ))

        return instances

    def _build_instance(self, name: str, n_customers: int, n_products: int, seed: int) -> Dict:
        rng = random.Random(seed)
        scaled_capacity = n_products * 500

        coords = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(n_customers)]

        customers = []
        for cid in range(n_customers):
            demand = {p + 1: rng.randint(1, 5) * 50 for p in range(n_products)}
            customers.append({
                "id": cid,
                "x": coords[cid][0],
                "y": coords[cid][1],
                "demand": demand,
                "time_window": [0, 1000],
                "service_time": 10,
            })

        vehicle = {
            "compartments": create_compartments(n_products, scaled_capacity),
            **VEHICLE_CONFIG,
            "max_payload": scaled_capacity,
            "max_volume": scaled_capacity,
            "product_densities": get_product_densities(n_products),
        }

        return {
            "name": name,
            "customers": customers,
            "time_windows": [[0.0, 1000.0, 10.0]] * n_customers,
            "vehicle": vehicle,
            "distance_matrix": compute_distance_matrix([(0, 0)] + coords),
            "metadata": {
                "source": SOURCE_INFO["mirzaei"]["name"],
                "source_doi": SOURCE_INFO["mirzaei"]["doi"],
                "layer": "mirzaei",
                "n_customers": n_customers,
                "n_commodities": n_products,
                "n_compartments": n_products,
                "scaled_capacity": scaled_capacity,
                "generated": datetime.now().isoformat(),
            },
        }


# ============================================================================
# MAIN GENERATOR
# ============================================================================

class DSCCTRPStandardGenerator:
    """Orchestrate generation of the full 100-instance benchmark."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.summary = {
            "paixao": {"count": 0, "names": []},
            "mirzaei": {"count": 0, "names": []},
            "total": 0,
        }

    def generate_all(self) -> None:
        print("=" * 80)
        print("DSC-CTRP STANDARD INSTANCE GENERATOR")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print("-" * 80)

        print("\n[1/2] Generating Paixao layer (15 instances)...")
        self._save(PaixaoInstanceGenerator(self.output_dir).generate_all(), "paixao")

        print("\n[2/2] Generating Mirzaei layer (85 instances)...")
        self._save(MirzaeiInstanceGenerator(self.output_dir).generate_all(), "mirzaei")

        self._print_summary()
        self._save_summary()

    def _save(self, instances: List[Dict], layer: str) -> None:
        for instance in instances:
            path = os.path.join(self.output_dir, f"{instance['name']}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(instance, fh, indent=2)
            self.summary[layer]["count"] += 1
            self.summary[layer]["names"].append(instance["name"])
            self.summary["total"] += 1
        print(f"  Generated {len(instances)} {layer} instances")

    def _print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"Total instances: {self.summary['total']}")
        print(f"  Paixao layer:  {self.summary['paixao']['count']}")
        print(f"  Mirzaei layer: {self.summary['mirzaei']['count']}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)

    def _save_summary(self) -> None:
        path = os.path.join(self.output_dir, "generation_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "total": self.summary["total"],
                "by_layer": {
                    "paixao": self.summary["paixao"],
                    "mirzaei": self.summary["mirzaei"],
                },
                "generated": datetime.now().isoformat(),
            }, fh, indent=2)
        print(f"Summary written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DSC-CTRP benchmark.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Directory in which to write the JSON instances.")
    args = parser.parse_args()

    DSCCTRPStandardGenerator(args.output).generate_all()


if __name__ == "__main__":
    main()