"""
DSC-CTRP v8.9.1 — SA-ALNS + deterministic seed + cache-hit reporting
=====================================================================

This script reproduces the experiments reported in:

    Ndikuriyo, Y., & Zhang, Y. (2026). A Stability-Aware Adaptive Large
    Neighborhood Search for Multi-Compartment Container-Tanker Routing with
    Dynamic Stability Constraints. Computers & Operations Research
    (under review).

It runs the state-of-the-art comparison, the ablation study, the
penalty-coefficient sensitivity sweep, the multi-compartment comparison, the
statistical tests, and the figures.

Reproducibility
---------------
All random seeds are derived from _stable_seed(), which uses SHA-1 on a
canonical string encoding rather than Python's built-in hash(). The built-in
hash() on strings is randomised per process (PYTHONHASHSEED), so results were
not reproducible across launches before v8.9.1. The current seed is stable
across processes, machines, and Python versions; the only source of variation
between runs is wall-clock time.

Cache accounting
----------------
PerformanceMetrics reports two distinct cache hit rates:

  * route_penalty_cache_hit_rate -- hit rate of BaseAlgorithm._route_penalty,
    the cache that the No Cache ablation variant exercises. This is the
    figure to quote in the manuscript.
  * cache_hit_rate -- SACA-level hit rate. Because SACA's LP fallback is never
    called on this benchmark, saca.milp_calls = 0 and this ratio saturates at
    100% for any run with at least one SACA hit. It is retained for backward
    compatibility but overstates the cache's contribution.

Author: Yves Ndikuriyo
Affiliation: Central South University
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

warnings.filterwarnings("ignore")

try:
    import pulp
except ImportError as exc:
    raise ImportError("Install PuLP: pip install pulp") from exc


# ============================================================================
# PATHS (portable; override with --instances and --output)
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INSTANCE_DIR = os.path.join(SCRIPT_DIR, "benchmark_instances")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "SA-ALNS_results_v8_9_1")

INSTANCE_GENERATOR_VERSION = "v8.7"  # instances unchanged since v8.7

_GLOBAL_SACA_CACHE = {}


# ============================================================================
# DETERMINISTIC SEED
# ============================================================================

def _stable_seed(*parts) -> int:
    """
    Deterministic replacement for hash((...)) % 2**31.

    Python's built-in hash() on strings is randomised per process
    (PYTHONHASHSEED), which made experimental runs non-reproducible across
    launches. This helper uses SHA-1 on a canonical string encoding of the
    parts, which is stable across processes, machines, and Python versions.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:8], 16) % (2**31)


# ============================================================================
# STYLE
# ============================================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["pdf.fonttype"] = 42

COLORS_SOTA = {
    "SA-ALNS": "#2E86AB", "SA-ALNS_CostOnly": "#7FB3D5",
    "Standard_ALNS": "#1A7A5A", "FDAHS": "#E76F51",
    "GCOF": "#F4A261", "HSA": "#E9C46A", "ACO_2opt": "#2A9D8F",
}
SOTA_DISPLAY_NAMES = {
    "SA-ALNS": "SA-ALNS (Proposed)",
    "SA-ALNS_CostOnly": "SA-ALNS (Cost-Only Ablation)",
    "Standard_ALNS": "Standard ALNS",
    "FDAHS": "FDAHS (2026)", "GCOF": "GCOF (2026)",
    "HSA": "HSA (2023)", "ACO_2opt": "ACO-2opt (2015)",
}
MARKERS = {
    "SA-ALNS": "o", "SA-ALNS_CostOnly": "X", "Standard_ALNS": "s",
    "FDAHS": "^", "GCOF": "D", "HSA": "v", "ACO_2opt": "p",
    "SA-ALNS_NS": "*", "SA-ALNS_NA": "h", "SA-ALNS_SEQ": "<",
    "SA-ALNS_NC": ">", "SA-ALNS_IMB": "P",
}
COLORS_ABLATION = {
    "SA-ALNS": "#2E86AB", "SA-ALNS_NS": "#F18F01", "SA-ALNS_NA": "#C73E1D",
    "SA-ALNS_SEQ": "#73AB84", "SA-ALNS_NC": "#6A4C93", "SA-ALNS_IMB": "#D4A373",
}
ABLATION_DISPLAY_NAMES = {
    "SA-ALNS": "SA-ALNS (Full)", "SA-ALNS_NS": "No Stability",
    "SA-ALNS_NA": "No Adaptive", "SA-ALNS_SEQ": "Sequential",
    "SA-ALNS_NC": "No Cache", "SA-ALNS_IMB": "No Imbalance",
}


# ============================================================================
# SAFE SAVE
# ============================================================================

def safe_savefig(fig, path, retries=3, sleep=1.0):
    for attempt in range(retries):
        try:
            fig.savefig(path)
            return path
        except PermissionError:
            time.sleep(sleep)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    base, ext = os.path.splitext(path)
    alt = f"{base}_{int(time.time())}{ext}"
    try:
        fig.savefig(alt)
        print(f"  [warn] {path} locked; wrote {alt} instead")
        return alt
    except Exception as e:
        print(f"  [error] could not save {path}: {e}")
        return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Compartment:
    id: int
    capacity: float
    position_x: float
    position_y: float
    load_density: float


@dataclass
class Vehicle:
    id: int
    compartments: List[Compartment]
    max_payload: float
    front_axle_max: float
    rear_axle_max: float
    wheelbase: float
    unladen_weight: float
    max_volume: float
    steering_min_ratio: float
    driving_min_ratio: float
    product_densities: Dict[int, float] = field(default_factory=dict)
    fixed_cost: float = 100.0
    cost_per_km: float = 1.0
    front_axle_min: float = 0.0
    rear_axle_min: float = 0.0
    cg_min: float = 0.0
    cg_max: float = 0.0
    compartment_positions: Dict[int, float] = field(default_factory=dict)
    is_multi_compartment: bool = True


@dataclass
class LDDResult:
    feasible: bool
    front_load: float
    rear_load: float
    center_of_gravity: float
    stability_margin: float
    violations: List[str]
    cg_position: float = 0.0
    cg_min: float = 0.0
    cg_max: float = 0.0
    cg_compliant: bool = True
    total_mass: float = 0.0
    longitudinal_moment: float = 0.0


@dataclass
class PerformanceMetrics:
    total_distance: float
    transportation_cost: float
    fleet_size: int
    fleet_utilization: float
    volume_utilization: float
    ldd_violation_rate: float
    stability_margin: float
    minimum_load_ratio: float
    slosh_risk_score: float
    cpu_time: float
    saca_calls: int
    cache_hit_rate: float
    iterations_to_convergence: int = 0
    milp_calls: int = 0
    cache_hits: int = 0
    milp_time: float = 0.0
    routing_time: float = 0.0
    feasibility_failures: int = 0
    feasibility_attempts: int = 0
    cg_position_mean: float = 0.0
    cg_position_std: float = 0.0
    cg_position_min: float = 0.0
    cg_position_max: float = 0.0
    cg_compliance_rate: float = 100.0
    cg_deviation_from_optimum: float = 0.0
    heeling_moment_estimate: float = 0.0
    lateral_acceleration_margin: float = 0.0
    penalised_cost: float = 0.0
    avg_penalty: float = 0.0
    n_total_routes: int = 0
    n_verified_routes: int = 0
    route_penalty_cache_hits: int = 0
    route_penalty_cache_hit_rate: float = 0.0


@dataclass
class SensitivityResult:
    lambda_W: float
    lambda_S: float
    avg_cost: float
    avg_stability: float
    avg_ldd_violation: float
    avg_fleet_size: float
    avg_cg_compliance: float
    avg_cpu_time: float
    std_cost: float
    std_stability: float
    convergence_iterations: int
    parameter_combination_id: str


# ============================================================================
# LDD VERIFIER
# ============================================================================

class LDDVerifier:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.compartment_positions = {
            c.id: float(np.clip(c.position_x, 0.0, vehicle.wheelbase))
            for c in vehicle.compartments
        }
        self.front_axle_min = getattr(vehicle, "front_axle_min", 0.0)
        self.rear_axle_min = getattr(vehicle, "rear_axle_min", 0.0)
        self.cg_min = getattr(vehicle, "cg_min", 0.3 * vehicle.wheelbase)
        self.cg_max = getattr(vehicle, "cg_max", 0.7 * vehicle.wheelbase)

    def verify_load(self, compartment_loads: Dict[int, float]) -> LDDResult:
        loads = {k: v for k, v in compartment_loads.items() if v > 1e-9}
        total_weight = sum(loads.values())

        if total_weight <= 1e-9:
            unladen = self.vehicle.unladen_weight
            unladen_cg = 0.5 * self.vehicle.wheelbase
            rear = unladen * unladen_cg / self.vehicle.wheelbase
            front = unladen - rear
            violations = []
            if front > self.vehicle.front_axle_max:
                violations.append("front_overload")
            if rear > self.vehicle.rear_axle_max:
                violations.append("rear_overload")
            return LDDResult(
                feasible=len(violations) == 0,
                front_load=front, rear_load=rear,
                center_of_gravity=unladen_cg,
                stability_margin=100.0, violations=violations,
                cg_position=unladen_cg, cg_min=self.cg_min, cg_max=self.cg_max,
                cg_compliant=True, total_mass=0.0, longitudinal_moment=0.0,
            )

        moment = sum(load * self.compartment_positions.get(cid, 0.5 * self.vehicle.wheelbase)
                     for cid, load in loads.items())
        cg_cargo = moment / total_weight

        unladen = self.vehicle.unladen_weight
        unladen_cg = 0.5 * self.vehicle.wheelbase
        total_mass = total_weight + unladen
        cg = (total_weight * cg_cargo + unladen * unladen_cg) / total_mass

        rear = total_mass * cg / self.vehicle.wheelbase
        front = total_mass - rear

        violations = []
        if front > self.vehicle.front_axle_max:
            violations.append("front_overload")
        if front < self.front_axle_min:
            violations.append("front_underload")
        if rear > self.vehicle.rear_axle_max:
            violations.append("rear_overload")
        if rear < self.rear_axle_min:
            violations.append("rear_underload")
        if total_weight > self.vehicle.max_payload:
            violations.append("payload")

        cg_ok = True
        if cg < self.cg_min:
            violations.append("cg_forward")
            cg_ok = False
        if cg > self.cg_max:
            violations.append("cg_rearward")
            cg_ok = False

        if total_mass > 0:
            if front / total_mass < self.vehicle.steering_min_ratio:
                violations.append("steering")
            if rear / total_mass < self.vehicle.driving_min_ratio:
                violations.append("driving")

        margins = []
        if self.vehicle.front_axle_max > 0:
            margins.append((self.vehicle.front_axle_max - front) / self.vehicle.front_axle_max)
        if self.vehicle.rear_axle_max > 0:
            margins.append((self.vehicle.rear_axle_max - rear) / self.vehicle.rear_axle_max)
        if self.vehicle.max_payload > 0:
            margins.append((self.vehicle.max_payload - total_weight) / self.vehicle.max_payload)
        if self.cg_max > self.cg_min:
            margins.append((cg - self.cg_min) / (self.cg_max - self.cg_min))
            margins.append((self.cg_max - cg) / (self.cg_max - self.cg_min))
        margin = max(0.0, min(margins)) * 100 if margins else 100.0

        return LDDResult(
            feasible=len(violations) == 0,
            front_load=front, rear_load=rear,
            center_of_gravity=cg, stability_margin=margin,
            violations=violations,
            cg_position=cg, cg_min=self.cg_min, cg_max=self.cg_max,
            cg_compliant=cg_ok, total_mass=total_weight,
            longitudinal_moment=moment,
        )


# ============================================================================
# SACA — sequence-aware + split allocation
# ============================================================================

class SACA:
    RISKY_BAND = (0.35, 0.55)
    MILP_SIZE_LIMIT = 9
    MILP_TIME_LIMIT = 10.0

    W_AXLE = 100.0
    W_CG = 500.0
    W_SLOSH = 2.0
    SLOSH_PENALTY_CAP_FRACTION = 0.20

    def __init__(self, vehicle: Vehicle, instance_data: Optional[Dict] = None,
                 stability_weight: float = 0.35, use_cache: bool = True):
        self.vehicle = vehicle
        self.instance_data = instance_data
        self.ldd_verifier = LDDVerifier(vehicle)
        self.stability_weight = stability_weight
        self._vehicle_sig = self._make_vehicle_signature(vehicle)
        self.use_cache = use_cache
        self.cache = _GLOBAL_SACA_CACHE if use_cache else {}
        self.milp_calls = 0
        self.cache_hits = 0
        self.milp_time = 0.0
        self.milp_status_log = []
        self.failure_reasons = defaultdict(int)

    @staticmethod
    def _make_vehicle_signature(vehicle):
        caps = tuple(round(c.capacity, 3) for c in vehicle.compartments)
        pos = tuple(round(c.position_x, 3) for c in vehicle.compartments)
        return (round(vehicle.wheelbase, 4),
                round(vehicle.max_volume, 3),
                round(vehicle.max_payload, 3),
                caps, pos)

    @staticmethod
    def _cache_key(route):
        return hashlib.sha1(repr(tuple(route)).encode()).hexdigest()

    def _sequence_feasible(self, route, demand_by_customer):
        k = len(route)
        if k == 0:
            return True
        suffix = [dict() for _ in range(k)]
        running = defaultdict(float)
        for i in range(k - 1, -1, -1):
            cust = route[i]
            for p, v in demand_by_customer[cust].items():
                running[p] += v
            suffix[i] = dict(running)
        for i in range(k):
            remaining = suffix[i]
            total_vol = sum(remaining.values())
            if total_vol > self.vehicle.max_volume + 1e-6:
                return False
            n_prod = sum(1 for v in remaining.values() if v > 1e-6)
            if n_prod > len(self.vehicle.compartments):
                return False
        return True

    def verify_route(self, route, demand_by_customer, transport_cost=None,
                     skip_sequence_check=False):
        if not route:
            return 0.0, {}, {}, False

        key = (self._vehicle_sig, SACA._cache_key(route))
        if self.use_cache and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        if not skip_sequence_check and not self._sequence_feasible(route, demand_by_customer):
            self.failure_reasons["sequence_infeasible"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[key] = res
            return res

        demand = defaultdict(float)
        for c in route:
            for p, v in demand_by_customer[c].items():
                demand[p] += v

        total_volume = sum(demand.values())
        if total_volume > self.vehicle.max_volume + 1e-6:
            self.failure_reasons["total_volume"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[key] = res
            return res

        n_required_products = sum(1 for v in demand.values() if v > 1e-6)
        if n_required_products > len(self.vehicle.compartments):
            self.failure_reasons["too_many_products"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[key] = res
            return res

        vol_alloc, prod_alloc = self._solve_heuristic(demand)
        if not vol_alloc:
            vol_alloc, prod_alloc = self._solve_lp(demand)

        if not vol_alloc:
            self.failure_reasons["allocation"] += 1
            res = (1e6, {}, {}, True)
            if self.use_cache:
                self.cache[key] = res
            return res

        penalty = self._penalty_from_states(
            prod_alloc, vol_alloc, route, demand_by_customer,
            transport_cost=transport_cost,
        )
        res = (penalty, vol_alloc, prod_alloc, False)
        if self.use_cache:
            self.cache[key] = res
        return res

    def _solve_heuristic(self, demand):
        comps = self.vehicle.compartments
        products = list(demand.keys())
        if not products or sum(demand.values()) <= 0:
            return {}, {}

        if sum(1 for v in demand.values() if v > 1e-6) > len(comps):
            return {}, {}

        sorted_products = sorted(demand.items(), key=lambda x: x[1], reverse=True)
        comp_state = {c.id: {"alloc": {}, "total": 0.0} for c in comps}

        for product_id, vol_needed in sorted_products:
            need = vol_needed
            for comp in comps:
                if need <= 1e-6:
                    break
                st = comp_state[comp.id]
                if product_id not in st["alloc"]:
                    continue
                avail = comp.capacity - st["total"]
                if avail <= 1e-6:
                    continue
                take = min(need, avail)
                st["alloc"][product_id] += take
                st["total"] += take
                need -= take

            for comp in comps:
                if need <= 1e-6:
                    break
                st = comp_state[comp.id]
                if st["total"] > 1e-6:
                    continue
                take = min(need, comp.capacity)
                st["alloc"][product_id] = take
                st["total"] = take
                need -= take

            for comp in comps:
                if need <= 1e-6:
                    break
                st = comp_state[comp.id]
                if product_id in st["alloc"]:
                    continue
                avail = comp.capacity - st["total"]
                if avail <= 1e-6:
                    continue
                take = min(need, avail)
                st["alloc"][product_id] = take
                st["total"] += take
                need -= take

            if need > 1e-3:
                return {}, {}

        vol_alloc, prod_alloc = {}, {}
        for cid, st in comp_state.items():
            if st["total"] <= 1e-6:
                vol_alloc[cid] = 0.0
                prod_alloc[cid] = {}
            else:
                vol_alloc[cid] = st["total"]
                prod_alloc[cid] = dict(st["alloc"])

        if abs(sum(vol_alloc.values()) - sum(demand.values())) > 1e-3:
            return {}, {}
        return vol_alloc, prod_alloc

    def _solve_lp(self, demand):
        veh = self.vehicle
        comps = veh.compartments
        products = list(demand.keys())
        if not products or sum(demand.values()) <= 0:
            return {}, {}

        alpha, beta = self.RISKY_BAND
        prob = pulp.LpProblem("SACA_LP", pulp.LpMinimize)

        q = {(p, c.id): pulp.LpVariable(f"q_{p}_{c.id}", lowBound=0)
             for p in products for c in comps}
        pi = {c.id: pulp.LpVariable(f"pi_{c.id}", lowBound=0) for c in comps}

        for c in comps:
            prob += pulp.lpSum(q[(p, c.id)] for p in products) <= c.capacity
        for p in products:
            prob += pulp.lpSum(q[(p, c.id)] for c in comps) >= demand[p]
        total_kg = pulp.lpSum(q[(p, c.id)] * 0.8
                              for p in products for c in comps)
        prob += total_kg <= veh.max_payload * 2.0
        for c in comps:
            f = pulp.lpSum(q[(p, c.id)] for p in products) / max(c.capacity, 1e-9)
            prob += f - pi[c.id] <= alpha
            prob += f + pi[c.id] >= beta
        prob += self.W_SLOSH * pulp.lpSum(pi.values())

        t0 = time.time()
        try:
            status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception as e:
            self.milp_time += time.time() - t0
            self.milp_calls += 1
            self.milp_status_log.append(f"ERROR:{e}")
            return {}, {}
        self.milp_time += time.time() - t0
        self.milp_calls += 1
        status = pulp.LpStatus[status_code]
        self.milp_status_log.append(status)
        if status not in ("Optimal", "Not Solved"):
            return {}, {}

        vol_alloc, prod_alloc = {}, {}
        try:
            for c in comps:
                comp_alloc = {}
                for p in products:
                    v = pulp.value(q[(p, c.id)])
                    if v is not None and v > 1e-6:
                        comp_alloc[p] = v
                vol_alloc[c.id] = sum(comp_alloc.values())
                prod_alloc[c.id] = comp_alloc
        except Exception:
            return {}, {}

        assigned_per_product = {p: 0.0 for p in products}
        for cid, alloc in prod_alloc.items():
            for p, v in alloc.items():
                assigned_per_product[p] += v
        for p, needed in demand.items():
            if assigned_per_product[p] < needed - 1e-3:
                return {}, {}
        return vol_alloc, prod_alloc

    def _penalty_from_states(self, prod_alloc, vol_alloc, route, demand_by_customer,
                              transport_cost=None):
        load_kg = {}
        for cid, vol in vol_alloc.items():
            if vol > 1e-6:
                load_kg[cid] = vol * 0.8
        if not load_kg:
            return 0.0

        ldd = self.ldd_verifier.verify_load(load_kg)
        penalty = 0.0
        for v in ldd.violations:
            if v == "front_overload":
                penalty += self.W_AXLE * (ldd.front_load - self.vehicle.front_axle_max)
            elif v == "rear_overload":
                penalty += self.W_AXLE * (ldd.rear_load - self.vehicle.rear_axle_max)
            elif v == "payload":
                penalty += self.W_AXLE * (ldd.total_mass - self.vehicle.max_payload)
            elif v == "cg_forward":
                penalty += self.W_CG * (self.vehicle.cg_min - ldd.cg_position)
            elif v == "cg_rearward":
                penalty += self.W_CG * (ldd.cg_position - self.vehicle.cg_max)

        slosh_penalty = 0.0
        for cid, kg in load_kg.items():
            comp = next((c for c in self.vehicle.compartments if c.id == cid), None)
            if comp is None or comp.capacity <= 0:
                continue
            fill = (kg / 0.8) / comp.capacity
            if self.RISKY_BAND[0] <= fill <= self.RISKY_BAND[1]:
                slosh_penalty += self.W_SLOSH * 1.0
            elif 0.25 <= fill < self.RISKY_BAND[0] or self.RISKY_BAND[1] < fill <= 0.65:
                slosh_penalty += self.W_SLOSH * 0.5

        if transport_cost is not None and transport_cost > 0:
            cap = self.SLOSH_PENALTY_CAP_FRACTION * transport_cost
            slosh_penalty = min(slosh_penalty, cap)

        penalty += slosh_penalty
        return penalty


class SACA_SingleProduct(SACA):
    def verify_route(self, route, demand_by_customer, transport_cost=None,
                     skip_sequence_check=False):
        if not route:
            return 0.0, {}, {}, False
        demand = defaultdict(float)
        for c in route:
            for p, v in demand_by_customer[c].items():
                demand[p] += v
        n_required = sum(1 for v in demand.values() if v > 1e-6)
        if n_required > 1:
            self.failure_reasons["multi_product_route"] += 1
            return (1e6, {}, {}, True)
        return super().verify_route(route, demand_by_customer, transport_cost,
                                    skip_sequence_check)


# ============================================================================
# OBJECTIVE CALCULATOR
# ============================================================================

class ObjectiveCalculator:
    def __init__(self, instance_data: Dict, vehicle: Vehicle,
                 stability_weight: float = 0.35,
                 saca: Optional[SACA] = None):
        self.instance_data = instance_data
        self.vehicle = vehicle
        self.stability_weight = stability_weight
        self.ldd_verifier = LDDVerifier(vehicle)
        if saca is not None:
            self.saca = saca
        else:
            self.saca = SACA(vehicle, instance_data,
                             stability_weight=stability_weight)
        self.distance_matrix = instance_data["distance_matrix"]

    def compute_total_distance(self, solution):
        total = 0.0
        for route in solution.values():
            if not route:
                continue
            nodes = [0] + [c + 1 for c in route] + [0]
            total += sum(self.distance_matrix[nodes[i]][nodes[i + 1]]
                         for i in range(len(nodes) - 1))
        return total

    def compute_transportation_cost(self, solution):
        return (self.vehicle.cost_per_km * self.compute_total_distance(solution)
                + self.vehicle.fixed_cost * len(solution))

    def _verify_with_cost(self, route):
        demands = {c: self.instance_data["customers"][c]["demand"] for c in route}
        travel = self.compute_transportation_cost({0: route})
        return self.saca.verify_route(route, demands, transport_cost=travel)

    def _iter_verified_routes(self, solution):
        n_total = sum(1 for r in solution.values() if r)
        verified = []
        for route in solution.values():
            if not route:
                continue
            res = self._verify_with_cost(route)
            verified.append((route, res))
        n_verified = sum(1 for (_, (_, _, _, hard)) in verified if not hard)
        return n_total, n_verified, verified

    def compute_penalised_cost(self, solution, lambda_W=1.0, lambda_S=0.5):
        travel = self.compute_transportation_cost(solution)
        total_penalty = 0.0
        for route in solution.values():
            if not route:
                continue
            penalty, _, _, hard = self._verify_with_cost(route)
            if hard:
                return float("inf")
            total_penalty += penalty
        return travel + (lambda_W + lambda_S) * total_penalty

    def compute_fleet_size(self, solution):
        return len(solution)

    def compute_fleet_utilization(self, solution):
        if not solution:
            return 0.0
        utilized = 0
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            util = sum(demands.values()) / max(self.vehicle.max_volume, 1e-9)
            if util >= 0.8:
                utilized += 1
        return utilized / len(solution) * 100

    def compute_volume_utilization(self, solution):
        total_volume, total_capacity = 0.0, 0.0
        for route in solution.values():
            if not route:
                continue
            total_volume += sum(self._extract_demands(route).values())
            total_capacity += self.vehicle.max_volume
        return (total_volume / total_capacity * 100) if total_capacity > 0 else 0.0

    def compute_ldd_violation_rate(self, solution):
        n_total, n_verified, verified = self._iter_verified_routes(solution)
        if n_verified == 0:
            return 0.0
        v = 0
        for (_, (penalty, _, _, hard)) in verified:
            if hard:
                continue
            if penalty > 1e-6:
                v += 1
        return v / n_verified * 100

    def compute_avg_penalty(self, solution):
        total, n = 0.0, 0
        for route in solution.values():
            if not route:
                continue
            penalty, _, _, hard = self._verify_with_cost(route)
            if not hard:
                total += penalty
                n += 1
        return total / max(n, 1)

    def compute_stability_margin(self, solution):
        n_total, n_verified, verified = self._iter_verified_routes(solution)
        if n_verified == 0:
            return 0.0
        margins = []
        for (route, (penalty, vol_alloc, _, hard)) in verified:
            if hard:
                continue
            load = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
            ldd = self.ldd_verifier.verify_load(load)
            if ldd.feasible:
                margins.append(ldd.stability_margin)
            else:
                margins.append(0.0)
        return float(np.mean(margins)) if margins else 0.0

    def compute_minimum_load_ratio(self, solution):
        n_total, n_verified, verified = self._iter_verified_routes(solution)
        if n_verified == 0:
            return 0.0
        ratios = []
        for (route, (penalty, vol_alloc, _, hard)) in verified:
            if hard:
                continue
            load = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
            ldd = self.ldd_verifier.verify_load(load)
            if ldd.feasible and ldd.total_mass > 0:
                total_vw = ldd.total_mass + self.vehicle.unladen_weight
                if total_vw > 0:
                    ratios.append(ldd.front_load / total_vw * 100)
        return float(np.mean(ratios)) if ratios else 0.0

    def compute_slosh_risk_score(self, solution):
        total_risk, n = 0.0, 0
        for route in solution.values():
            if not route:
                continue
            penalty, _, _, hard = self._verify_with_cost(route)
            if not hard:
                total_risk += penalty
                n += 1
        return total_risk / max(n, 1)

    def compute_cg_metrics(self, solution):
        n_total, n_verified, verified = self._iter_verified_routes(solution)
        if n_verified == 0:
            return {"cg_position_mean": 0.0, "cg_position_std": 0.0,
                    "cg_position_min": 0.0, "cg_position_max": 0.0,
                    "cg_compliance_rate": 0.0,
                    "cg_deviation_from_optimum": 0.0}
        cgs = []
        n_compliant = 0
        for (route, (penalty, vol_alloc, _, hard)) in verified:
            if hard:
                continue
            load = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
            ldd = self.ldd_verifier.verify_load(load)
            if ldd.cg_compliant:
                n_compliant += 1
            if ldd.feasible:
                cgs.append(ldd.cg_position)
        rate = 100.0 * n_compliant / n_verified
        if not cgs:
            return {"cg_position_mean": 0.0, "cg_position_std": 0.0,
                    "cg_position_min": 0.0, "cg_position_max": 0.0,
                    "cg_compliance_rate": rate,
                    "cg_deviation_from_optimum": 0.0}
        mean = float(np.mean(cgs))
        opt = 0.5 * self.vehicle.wheelbase
        return {"cg_position_mean": mean,
                "cg_position_std": float(np.std(cgs)) if len(cgs) > 1 else 0.0,
                "cg_position_min": float(min(cgs)), "cg_position_max": float(max(cgs)),
                "cg_compliance_rate": rate,
                "cg_deviation_from_optimum": abs(mean - opt)}

    def compute_heeling_moment_estimate(self, solution, lateral_acceleration=0.4):
        max_m = 0.0
        for route in solution.values():
            if not route:
                continue
            penalty, vol_alloc, _, hard = self._verify_with_cost(route)
            if not hard and vol_alloc:
                load = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
                total_mass = sum(load.values())
                cg_h = 0.5 + float(np.mean(
                    [self.ldd_verifier.compartment_positions.get(cid, 0) for cid in load]))
                max_m = max(max_m, total_mass * lateral_acceleration * cg_h)
        return max_m

    def compute_lateral_acceleration_margin(self, solution, max_heeling=1000):
        hm = self.compute_heeling_moment_estimate(solution)
        return (max_heeling - hm) / max_heeling * 100 if hm > 0 else 100.0

    def _extract_demands(self, route):
        demands = defaultdict(float)
        for cid in route:
            for p, v in self.instance_data["customers"][cid]["demand"].items():
                demands[p] += v
        return dict(demands)

    def compute_all_metrics(self, solution, iterations_to_convergence=0,
                            routing_time=0.0, milp_time=0.0,
                            lambda_W=1.0, lambda_S=0.5):
        start = time.time()
        total_distance = self.compute_total_distance(solution)
        transportation_cost = self.compute_transportation_cost(solution)
        penalised_cost = self.compute_penalised_cost(solution, lambda_W, lambda_S)
        avg_penalty = self.compute_avg_penalty(solution)
        fleet_size = self.compute_fleet_size(solution)
        fleet_util = self.compute_fleet_utilization(solution)
        vol_util = self.compute_volume_utilization(solution)
        ldd_v = self.compute_ldd_violation_rate(solution)
        stability = self.compute_stability_margin(solution)
        min_ratio = self.compute_minimum_load_ratio(solution)
        slosh = self.compute_slosh_risk_score(solution)
        cg = self.compute_cg_metrics(solution)
        hm = self.compute_heeling_moment_estimate(solution)
        lm = self.compute_lateral_acceleration_margin(solution)
        wall = time.time() - start

        n_total, n_verified, _ = self._iter_verified_routes(solution)

        return PerformanceMetrics(
            total_distance=total_distance, transportation_cost=transportation_cost,
            fleet_size=fleet_size, fleet_utilization=fleet_util,
            volume_utilization=vol_util, ldd_violation_rate=ldd_v,
            stability_margin=stability, minimum_load_ratio=min_ratio,
            slosh_risk_score=slosh,
            cpu_time=routing_time + milp_time + wall,
            saca_calls=self.saca.milp_calls,
            cache_hit_rate=(self.saca.cache_hits /
                            max(1, self.saca.milp_calls + self.saca.cache_hits)) * 100,
            iterations_to_convergence=iterations_to_convergence,
            milp_calls=self.saca.milp_calls,
            cache_hits=self.saca.cache_hits,
            milp_time=milp_time, routing_time=routing_time,
            cg_position_mean=cg["cg_position_mean"],
            cg_position_std=cg["cg_position_std"],
            cg_position_min=cg["cg_position_min"],
            cg_position_max=cg["cg_position_max"],
            cg_compliance_rate=cg["cg_compliance_rate"],
            cg_deviation_from_optimum=cg["cg_deviation_from_optimum"],
            heeling_moment_estimate=hm,
            lateral_acceleration_margin=lm,
            penalised_cost=penalised_cost,
            avg_penalty=avg_penalty,
            n_total_routes=n_total,
            n_verified_routes=n_verified)


# ============================================================================
# BENCHMARK GENERATION
# ============================================================================

def generate_single_instance(n_customers, layer, seed, name=None,
                             n_products=4, n_compartments=3):
    random.seed(seed)
    np.random.seed(seed)
    if name is None:
        name = f"{layer}_{n_customers}"

    n_products = max(2, min(n_products, n_compartments + 2))
    wb = random.uniform(3.0, 5.0)

    comps = []
    comp_pos = {}
    for i in range(n_compartments):
        frac = 0.30 + 0.40 * (i + 0.5) / n_compartments
        px = wb * frac
        comp_pos[i] = px
        comps.append({
            "id": i,
            "capacity": random.uniform(120.0, 200.0),
            "position_x": px, "position_y": 0.0, "load_density": 1.0,
        })

    total_capacity = sum(c["capacity"] for c in comps)
    max_payload = total_capacity * 0.8
    unladen = max_payload * random.uniform(0.30, 0.50)
    total_gvw = max_payload + unladen
    fx = total_gvw * 0.50
    rx = total_gvw * 0.65

    avg_comp_cap = total_capacity / n_compartments
    demand_max = avg_comp_cap * random.uniform(0.60, 0.90)
    demand_min = demand_max * 0.40

    customers = []
    for i in range(n_customers):
        p = random.randint(0, n_products - 1)
        demand = {p: random.uniform(demand_min, demand_max)}
        customers.append({
            "id": i,
            "x": random.uniform(0, 100),
            "y": random.uniform(0, 100),
            "demand": demand,
            "time_window": [0.0, 1440.0],
            "service_time": random.uniform(5, 15),
        })

    tws = {}
    for i in range(n_customers):
        s = random.uniform(0, 100)
        tws[i] = [s, s + random.uniform(20, 60)]
        customers[i]["time_window"] = tws[i]

    prod_dens = {p: 0.8 for p in range(n_products)}
    vehicle = {
        "id": 0, "compartments": comps, "compartment_positions": comp_pos,
        "max_payload": max_payload,
        "front_axle_max": fx, "rear_axle_max": rx,
        "front_axle_min": fx * 0.10, "rear_axle_min": rx * 0.10,
        "wheelbase": wb, "unladen_weight": unladen,
        "max_volume": total_capacity,
        "steering_min_ratio": 0.25, "driving_min_ratio": 0.55,
        "product_densities": prod_dens,
        "cg_min": 0.3 * wb, "cg_max": 0.7 * wb,
    }

    coords = [[0, 0]] + [[c["x"], c["y"]] for c in customers]
    dm = [[math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                     (coords[i][1] - coords[j][1]) ** 2)
           for j in range(n_customers + 1)] for i in range(n_customers + 1)]

    return {
        "name": name, "layer": layer, "customers": customers,
        "time_windows": tws, "vehicle": vehicle,
        "distance_matrix": dm, "n_products": n_products,
        "n_customers": n_customers, "n_compartments": n_compartments,
    }


def generate_benchmark_instances(output_dir, num_instances=100):
    print(f"\nGenerating {num_instances} synthetic benchmark instances ...")
    all_instances = []

    paixao = [
        ("P20", 20, 4, 3),
        ("P25", 25, 4, 3),
        ("P30", 30, 4, 4),
    ]
    mirzaei = [
        (40, 4, 3), (40, 5, 3), (40, 4, 4),
        (50, 4, 3), (50, 5, 3), (50, 5, 4),
        (60, 5, 3), (60, 5, 4), (60, 4, 4),
        (75, 5, 4), (75, 5, 4), (75, 4, 5),
        (100, 5, 4), (100, 5, 5), (100, 4, 5),
    ]

    ctr = 1
    for prefix, n, npr, ncmp in paixao:
        for i in range(1, 6):
            name = f"{prefix}_{i}"
            all_instances.append((name, generate_single_instance(
                n_customers=n, layer="Paixao", seed=ctr, name=name,
                n_products=npr, n_compartments=ncmp)))
            ctr += 1

    idx = 1
    for n, npr, ncmp in mirzaei:
        name = f"vrpnc{idx}"
        all_instances.append((name, generate_single_instance(
            n_customers=n, layer="Mirzaei", seed=ctr, name=name,
            n_products=npr, n_compartments=ncmp)))
        ctr += 1
        idx += 1

    while len(all_instances) < num_instances:
        n = random.choice([20, 25, 30, 40, 50, 60, 75, 100])
        npr = random.choice([3, 4, 5])
        ncmp = random.choice([3, 4, 5])
        name = f"extra{len(all_instances)+1}"
        all_instances.append((name, generate_single_instance(
            n_customers=n, layer="Mirzaei", seed=ctr, name=name,
            n_products=npr, n_compartments=ncmp)))
        ctr += 1

    for name, inst in all_instances:
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(inst, f, indent=2)
    print(f"Generated {len(all_instances)} instances in {output_dir}")
    return all_instances


def _instance_generator_is_current(base_dir):
    marker = os.path.join(base_dir, ".instance_generator_version")
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, "r") as f:
            version = f.read().strip()
        return version == INSTANCE_GENERATOR_VERSION
    except Exception:
        return False


def _write_instance_generator_marker(base_dir):
    marker = os.path.join(base_dir, ".instance_generator_version")
    with open(marker, "w") as f:
        f.write(INSTANCE_GENERATOR_VERSION)


def ensure_instances(base_dir):
    current = _instance_generator_is_current(base_dir)
    if current:
        js = [f for _, _, fs in os.walk(base_dir)
              for f in fs if f.endswith(".json")]
        if js:
            print(f"Instances are current ({INSTANCE_GENERATOR_VERSION}); "
                  f"found {len(js)} .json files. Skipping regeneration.")
            return

    print(f"\nInstance generator marker missing or stale. "
          f"Regenerating all instances with {INSTANCE_GENERATOR_VERSION} ...")
    removed = 0
    for root, dirs, files in os.walk(base_dir):
        if "experimental_results" in root or "SA-ALNS_results" in root:
            continue
        for f in files:
            if f.endswith(".json"):
                os.remove(os.path.join(root, f))
                removed += 1
    if removed:
        print(f"  Removed {removed} stale .json files.")
    generate_benchmark_instances(base_dir, num_instances=100)
    _write_instance_generator_marker(base_dir)
    print(f"  Wrote version marker {INSTANCE_GENERATOR_VERSION}.")


# ============================================================================
# VEHICLE CREATORS
# ============================================================================

def create_vehicle_from_instance(instance, multi_compartment=True):
    vd = instance["vehicle"]
    raw = vd["compartments"]
    if multi_compartment:
        compartments = [Compartment(id=c["id"], capacity=c["capacity"],
                                    position_x=c["position_x"],
                                    position_y=c["position_y"],
                                    load_density=1.0) for c in raw]
        positions = {c["id"]: c["position_x"] for c in raw}
        total_cap = sum(c.capacity for c in compartments)
    else:
        total_cap = sum(c["capacity"] for c in raw)
        weighted_x = sum(c["capacity"] * c["position_x"] for c in raw) / max(total_cap, 1e-9)
        compartments = [Compartment(id=0, capacity=total_cap,
                                    position_x=weighted_x, position_y=0.0,
                                    load_density=1.0)]
        positions = {0: weighted_x}
    return Vehicle(
        id=0, compartments=compartments,
        max_payload=vd["max_payload"], front_axle_max=vd["front_axle_max"],
        rear_axle_max=vd["rear_axle_max"], wheelbase=vd["wheelbase"],
        unladen_weight=vd["unladen_weight"], max_volume=total_cap,
        steering_min_ratio=vd["steering_min_ratio"],
        driving_min_ratio=vd["driving_min_ratio"],
        product_densities=vd.get("product_densities", {}),
        fixed_cost=100.0, cost_per_km=1.0,
        front_axle_min=vd.get("front_axle_min", vd["front_axle_max"] * 0.1),
        rear_axle_min=vd.get("rear_axle_min", vd["rear_axle_max"] * 0.1),
        cg_min=vd.get("cg_min", 0.3 * vd["wheelbase"]),
        cg_max=vd.get("cg_max", 0.7 * vd["wheelbase"]),
        compartment_positions=positions,
        is_multi_compartment=multi_compartment)


def build_instance_data(instance):
    n = len(instance["customers"])
    coords = [[0, 0]] + [[c["x"], c["y"]] for c in instance["customers"]]
    dm = [[math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                     (coords[i][1] - coords[j][1]) ** 2)
           for j in range(n + 1)] for i in range(n + 1)]
    return {
        "name": instance["name"],
        "customers": instance["customers"],
        "time_windows": instance["time_windows"],
        "distance_matrix": dm,
        "vehicle": instance["vehicle"],
        "n_products": instance.get("n_products", 4),
        "n_customers": n,
        "n_compartments": instance.get("n_compartments",
                                        len(instance["vehicle"]["compartments"])),
    }


# ============================================================================
# BASE ALGORITHM
# ============================================================================

class BaseAlgorithm:
    def __init__(self, instance_data, vehicle, stability_weight=0.35):
        self.instance_data = instance_data
        self.vehicle = vehicle
        self.stability_weight = stability_weight

        self.saca = SACA(vehicle, instance_data,
                         stability_weight=stability_weight)
        self.calc = ObjectiveCalculator(instance_data, vehicle,
                                        stability_weight=stability_weight,
                                        saca=self.saca)

        self.distance_matrix = instance_data["distance_matrix"]
        self.feasibility_cache = {}
        self.cache_hits = 0
        self.n_customers = len(instance_data["customers"])
        self.convergence_history = []
        self.feasibility_attempts = 0
        self.feasibility_failures = 0
        self.lambda_W = 1.0
        self.lambda_S = 0.5

    def _is_complete(self, solution):
        """
        True iff every customer 0..n_customers-1 appears exactly once across
        all routes. Rejects duplicated and missing customers.
        """
        seen = set()
        for route in solution.values():
            for c in route:
                if c in seen:
                    return False
                seen.add(c)
        return len(seen) == self.n_customers

    def _route_penalty(self, route):
        if not route:
            return 0.0, False
        key = SACA._cache_key(route)
        if key in self.feasibility_cache:
            self.cache_hits += 1
            return self.feasibility_cache[key]
        self.feasibility_attempts += 1

        demands_by_cust = {c: self.instance_data["customers"][c]["demand"] for c in route}

        if not self.saca._sequence_feasible(route, demands_by_cust):
            self.feasibility_cache[key] = (1e6, True)
            self.feasibility_failures += 1
            return 1e6, True

        travel = self.calc.compute_transportation_cost({0: route})
        penalty, _, _, hard = self.saca.verify_route(
            route, demands_by_cust, transport_cost=travel,
            skip_sequence_check=True)
        self.feasibility_cache[key] = (penalty, hard)
        if hard or penalty > 1e-6:
            self.feasibility_failures += 1
        return penalty, hard

    def _route_score(self, route, vid=0):
        penalty, hard = self._route_penalty(route)
        if hard:
            return 1e9
        travel = self.calc.compute_transportation_cost({vid: route})
        return travel + (self.lambda_W + self.lambda_S) * penalty

    def _extract_demands(self, route):
        d = defaultdict(float)
        for cid in route:
            for p, v in self.instance_data["customers"][cid]["demand"].items():
                d[p] += v
        return dict(d)

    def _generate_initial_solution(self):
        solution = self._greedy_insert(list(range(self.n_customers)))
        if self._is_trivial(solution):
            ordered = sorted(
                range(self.n_customers),
                key=lambda c: -sum(
                    self.instance_data["customers"][c]["demand"].values()
                ),
            )
            solution = self._greedy_insert(ordered)
        if self._is_trivial(solution):
            for _ in range(3):
                perm = list(range(self.n_customers))
                random.shuffle(perm)
                solution = self._greedy_insert(perm)
                if not self._is_trivial(solution):
                    break
        return solution

    def _greedy_insert(self, order):
        solution = {}
        for customer in order:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [customer] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen)
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                solution[best_vid].insert(best_pos, customer)
            else:
                solution[len(solution)] = [customer]
        return solution

    @staticmethod
    def _is_trivial(solution):
        return bool(solution) and all(len(r) == 1 for r in solution.values())

    def _two_opt_improvement(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        for vid in list(new.keys()):
            route = new[vid]
            if len(route) < 3:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        pen, hard = self._route_penalty(nr)
                        if hard:
                            continue
                        cur_score = (self.calc.compute_transportation_cost({vid: route})
                                     + (self.lambda_W + self.lambda_S) *
                                     self._route_penalty(route)[0])
                        new_score = (self.calc.compute_transportation_cost({vid: nr})
                                     + (self.lambda_W + self.lambda_S) * pen)
                        if new_score < cur_score:
                            route = nr
                            improved = True
                            break
                    if improved:
                        break
            new[vid] = route
        return new

    def _finalize(self, solution, iterations, wall_clock):
        m = self.calc.compute_all_metrics(
            solution, iterations_to_convergence=iterations,
            routing_time=wall_clock, milp_time=self.saca.milp_time,
            lambda_W=self.lambda_W, lambda_S=self.lambda_S)
        m.feasibility_attempts = self.feasibility_attempts
        m.feasibility_failures = self.feasibility_failures
        total_lookups = self.feasibility_attempts + self.cache_hits
        m.route_penalty_cache_hits = self.cache_hits
        m.route_penalty_cache_hit_rate = (
            100.0 * self.cache_hits / total_lookups if total_lookups > 0 else 0.0)
        return m


# ============================================================================
# STANDARD ALNS
# ============================================================================

class StandardALNS(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {
            "max_iterations": 150, "initial_temperature": 100.0,
            "cooling_rate": 0.99, "destruction_rate": 0.3,
            "max_no_improvement": 50,
        }

    def solve(self):
        t0 = time.time()
        solution = self._generate_initial_solution()
        current = self._solution_score(solution)
        best = solution.copy()
        best_cost = current
        T = self.params["initial_temperature"]
        it, noimp = 0, 0
        while it < self.params["max_iterations"]:
            it += 1
            d_op = random.choice(["random", "worst", "shaw"])
            r_op = random.choice(["greedy", "regret"])
            destroyed, removed = self._destroy(solution, d_op)
            repaired = self._repair(destroyed, removed, r_op)
            new_cost = self._solution_score(repaired)
            if new_cost < current or random.random() < math.exp(
                    -(new_cost - current) / max(T, 1e-10)):
                solution = repaired
                current = new_cost
                if new_cost < best_cost:
                    best = solution.copy()
                    best_cost = new_cost
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            T *= self.params["cooling_rate"]
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it, wall_clock),
                                 "convergence": self.convergence_history}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _destroy(self, solution, op):
        new = {k: v.copy() for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]

        if op == "random":
            n_remove = max(2, int(len(all_c) * self.params["destruction_rate"]))
            if len(all_c) <= n_remove:
                return new, []
            to_remove = set(random.sample(all_c, n_remove))
            removed = list(to_remove)
            for vid in list(new.keys()):
                new[vid] = [c for c in new[vid] if c not in to_remove]
                if not new[vid]:
                    del new[vid]
            return new, removed

        if op == "worst":
            marginal = {}
            for vid, route in new.items():
                if len(route) <= 1:
                    continue
                for i, c in enumerate(route):
                    rw = route[:i] + route[i + 1:]
                    if rw:
                        marginal[(vid, c)] = (
                            self.calc.compute_transportation_cost({vid: route})
                            - self.calc.compute_transportation_cost({vid: rw})
                        )
            if not marginal:
                return new, []
            k = max(2, int(len(marginal) * self.params["destruction_rate"] * 0.8))
            kept = sorted(marginal.items(), key=lambda x: x[1], reverse=True)[:k]
            to_remove = {key for key, _ in kept}
            removed = [c for _, c in to_remove]
            for vid in list(new.keys()):
                new[vid] = [c for c in new[vid] if (vid, c) not in to_remove]
                if not new[vid]:
                    del new[vid]
            return new, removed

        if len(all_c) < 3:
            return self._destroy(solution, "random")
        seed = random.choice(all_c)
        sp = set(self.instance_data["customers"][seed]["demand"].keys())
        similar = []
        for c in all_c:
            if c == seed:
                continue
            ov = len(sp.intersection(
                set(self.instance_data["customers"][c]["demand"].keys())))
            if ov > 0:
                similar.append((c, ov))
        if not similar:
            return self._destroy(solution, "random")
        similar.sort(key=lambda x: x[1], reverse=True)
        k = max(2, min(len(similar),
                       int(len(all_c) * self.params["destruction_rate"] * 0.7)))
        to_remove = {seed} | {c for c, _ in similar[:k]}
        removed = list(to_remove)
        for vid in list(new.keys()):
            new[vid] = [c for c in new[vid] if c not in to_remove]
            if not new[vid]:
                del new[vid]
        return new, removed

    def _repair(self, solution, unassigned, op):
        new = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        random.shuffle(remaining)
        for c in remaining[:]:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen)
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                new[best_vid].insert(best_pos, c)
                remaining.remove(c)
        for c in remaining:
            new[len(new)] = [c]
        return new


# ============================================================================
# FDAHS / GCOF / HSA / ACO-2opt
# ============================================================================

class FDAHSAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "population_size": 10,
                                 "crossover_rate": 0.8, "mutation_rate": 0.1,
                                 "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        pop = []
        for _ in range(self.params["population_size"]):
            s = self._two_opt_improvement(self._generate_initial_solution())
            pop.append(s)
        costs = [self._solution_score(s) for s in pop]
        bi = int(np.argmin(costs))
        best, best_cost = pop[bi].copy(), costs[bi]
        it, noimp = 0, 0
        while it < self.params["max_iterations"]:
            it += 1
            new_pop = []
            for _ in range(self.params["population_size"]):
                p1 = random.choice(pop)
                p2 = random.choice(pop)
                child = self._crossover(p1, p2)
                if random.random() < self.params["mutation_rate"]:
                    child = self._mutate(child)
                child = self._two_opt_improvement(child)
                new_pop.append(child)
            pop = new_pop
            costs = [self._solution_score(s) for s in pop]
            bi = int(np.argmin(costs))
            if costs[bi] < best_cost - 1e-9:
                best, best_cost = pop[bi].copy(), costs[bi]
                noimp = 0
            else:
                noimp += 1
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it, wall_clock)}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _crossover(self, p1, p2):
        c1 = [c for r in p1.values() for c in r]
        c2 = [c for r in p2.values() for c in r]
        if len(c1) < 2:
            return {k: v.copy() for k, v in p1.items()}
        i = random.randint(0, len(c1) - 2)
        j = random.randint(i + 1, len(c1) - 1)
        seq = [None] * len(c1)
        seq[i:j] = c1[i:j]
        j2 = 0
        for k in range(len(seq)):
            if seq[k] is None:
                while j2 < len(c2) and c2[j2] in seq:
                    j2 += 1
                if j2 < len(c2):
                    seq[k] = c2[j2]
                    j2 += 1
        child = {}
        for c in seq:
            if c is None:
                continue
            placed = False
            for vid, route in child.items():
                test = route + [c]
                pen, hard = self._route_penalty(test)
                if not hard:
                    child[vid].append(c)
                    placed = True
                    break
            if not placed:
                child[len(child)] = [c]
        return child

    def _mutate(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        routes = list(new.keys())
        if len(routes) < 2:
            return new
        v1, v2 = random.sample(routes, 2)
        r1, r2 = new[v1], new[v2]
        if not r1 or not r2:
            return new
        i1 = random.randint(0, len(r1) - 1)
        i2 = random.randint(0, len(r2) - 1)
        nr1 = r1[:i1] + [r2[i2]] + r1[i1 + 1:]
        nr2 = r2[:i2] + [r1[i1]] + r2[i2 + 1:]
        pen1, hard1 = self._route_penalty(nr1)
        pen2, hard2 = self._route_penalty(nr2)
        if not hard1 and not hard2:
            new[v1], new[v2] = nr1, nr2
        return new


class GCOFAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "population_size": 10,
                                 "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        pop = []
        for _ in range(self.params["population_size"]):
            s = self._two_opt_improvement(self._generate_initial_solution())
            pop.append(s)
        costs = [self._solution_score(s) for s in pop]
        bi = int(np.argmin(costs))
        best, best_cost = pop[bi].copy(), costs[bi]
        it, noimp = 0, 0
        while it < self.params["max_iterations"]:
            it += 1
            new_pop = []
            for sol in pop:
                g = {k: v.copy() for k, v in sol.items()}
                if random.random() < 0.3:
                    other = random.choice(pop)
                    g = self._cgpa_adjust(g, other)
                g = self._two_opt_improvement(g)
                new_pop.append(g)
            pop = new_pop
            costs = [self._solution_score(s) for s in pop]
            bi = int(np.argmin(costs))
            if costs[bi] < best_cost - 1e-9:
                best, best_cost = pop[bi].copy(), costs[bi]
                noimp = 0
            else:
                noimp += 1
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it, wall_clock)}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _cgpa_adjust(self, sol, other):
        new = {k: v.copy() for k, v in sol.items()}
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    test = route[:i] + route[j:j + 1] + route[i:j] + route[j + 1:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    cur_score = (self.calc.compute_transportation_cost({vid: route})
                                 + (self.lambda_W + self.lambda_S) *
                                 self._route_penalty(route)[0])
                    new_score = (self.calc.compute_transportation_cost({vid: test})
                                 + (self.lambda_W + self.lambda_S) * pen)
                    if new_score < cur_score:
                        new[vid] = test
                        break
        return new


class HSAAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60,
                                 "initial_temperature": 40.0,
                                 "cooling_rate": 0.98, "max_no_improvement": 25}

    def solve(self):
        t0 = time.time()
        solution = self._two_opt_improvement(self._generate_initial_solution())
        current = self._solution_score(solution)
        best, best_cost = solution.copy(), current
        T = self.params["initial_temperature"]
        it, noimp = 0, 0
        while T > 0.001 and it < self.params["max_iterations"]:
            it += 1
            nb = self._neighbor(solution)
            nc = self._solution_score(nb)
            if nc < current or random.random() < math.exp(-(nc - current) / max(T, 1e-10)):
                solution = nb
                current = nc
                if nc < best_cost:
                    best, best_cost = solution.copy(), nc
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            T *= self.params["cooling_rate"]
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it, wall_clock)}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _neighbor(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        routes = list(new.keys())
        if len(routes) < 2:
            return new
        v1, v2 = random.sample(routes, 2)
        r1, r2 = new[v1], new[v2]
        if not r1 or not r2:
            return new
        i1 = random.randint(0, len(r1) - 1)
        i2 = random.randint(0, len(r2) - 1)
        nr1 = r1[:i1] + [r2[i2]] + r1[i1 + 1:]
        nr2 = r2[:i2] + [r1[i1]] + r2[i2 + 1:]
        pen1, hard1 = self._route_penalty(nr1)
        pen2, hard2 = self._route_penalty(nr2)
        if not hard1 and not hard2:
            new[v1], new[v2] = nr1, nr2
        return new


class ACO2optAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {"max_iterations": 60, "num_ants": 10,
                                 "alpha": 1.0, "beta": 2.0, "rho": 0.9,
                                 "max_no_improvement": 25}
        self.pheromone = None

    def solve(self):
        t0 = time.time()
        n = self.n_customers
        self.pheromone = np.full((n + 1, n + 1), 1.0 / (n + 1))
        best = self._two_opt_improvement(self._generate_initial_solution())
        best_cost = self._solution_score(best)
        it, noimp = 0, 0
        for it in range(self.params["max_iterations"]):
            sols, costs = [], []
            for _ in range(self.params["num_ants"]):
                s = self._ant_construct()
                s = self._two_opt_improvement(s)
                c = self._solution_score(s)
                sols.append(s)
                costs.append(c)
                if c < best_cost:
                    best, best_cost = s.copy(), c
                    noimp = 0
                else:
                    noimp += 1
            self._update_pheromone(sols, costs)
            self.pheromone *= (1 - self.params["rho"])
            self.pheromone += 1e-9
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it + 1, wall_clock)}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _ant_construct(self):
        order = list(range(self.n_customers))
        random.shuffle(order)
        solution = {}
        for c in order:
            best = None
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    if pos == 0:
                        prev = 0
                        nxt = route[0] + 1 if route else 0
                    elif pos == len(route):
                        prev = route[-1] + 1
                        nxt = 0
                    else:
                        prev = route[pos - 1] + 1
                        nxt = route[pos] + 1
                    h = 1.0 / (self.distance_matrix[prev][c + 1] +
                               self.distance_matrix[c + 1][nxt] + 1e-6)
                    pher = 0.5 * (self.pheromone[prev][c + 1] +
                                  self.pheromone[c + 1][nxt])
                    sc = (pher ** self.params["alpha"]) * (h ** self.params["beta"])
                    if best is None or sc > best[0]:
                        best = (sc, vid, pos)
            if best is None:
                solution[len(solution)] = [c]
            else:
                _, vid, pos = best
                solution[vid].insert(pos, c)
        return solution

    def _update_pheromone(self, sols, costs):
        for s, c in zip(sols, costs):
            if c <= 0 or not np.isfinite(c):
                continue
            d = 1.0 / c
            for route in s.values():
                if not route:
                    continue
                nodes = [0] + [x + 1 for x in route] + [0]
                for i in range(len(nodes) - 1):
                    self.pheromone[nodes[i]][nodes[i + 1]] += d
                    self.pheromone[nodes[i + 1]][nodes[i]] += d


# ============================================================================
# SA-ALNS and ablation variants
# ============================================================================

class SAALNSAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, stability_weight=stability_weight)
        self.params = params or {
            "max_iterations": 150, "initial_temperature": 100.0,
            "cooling_rate": 0.99, "destruction_rate": 0.3,
            "learning_rate": 0.15, "stability_weight": stability_weight,
            "tabu_tenure": 10, "adaptive_frequency": 10,
            "max_no_improvement": 50,
        }
        self.saca.stability_weight = self.params["stability_weight"]
        self.calc.stability_weight = self.params["stability_weight"]
        self.destroy_weights = {"random": 1.0, "worst": 1.2, "compatibility": 1.0,
                                "imbalance": 1.3, "shaw": 1.1}
        self.repair_weights = {"greedy": 1.0, "regret": 1.3,
                               "stability_aware": 1.5, "balance": 1.1}

    def solve(self):
        t0 = time.time()
        solution = self._enhanced_initial()
        current = self._solution_score(solution)
        best, best_cost = solution.copy(), current
        T = self.params["initial_temperature"]
        it, noimp = 0, 0
        dperf, rperf = defaultdict(list), defaultdict(list)
        while it < self.params["max_iterations"]:
            it += 1
            d_op = self._select(self.destroy_weights)
            r_op = self._select(self.repair_weights)
            destroyed, removed = self._destroy(solution, d_op)
            repaired = self._repair(destroyed, removed, r_op)
            new_cost = self._solution_score(repaired)
            if current > 0 and np.isfinite(current) and np.isfinite(new_cost):
                imp = (current - new_cost) / current
                dperf[d_op].append(imp)
                rperf[r_op].append(imp)
            if (new_cost < current or
                    random.random() < math.exp(-(new_cost - current) / max(T, 1e-10))):
                solution = repaired
                current = new_cost
                if new_cost < best_cost:
                    best, best_cost = solution.copy(), new_cost
                    noimp = 0
                else:
                    noimp += 1
            else:
                noimp += 1
            if it % self.params["adaptive_frequency"] == 0:
                self._update_weights(dperf, rperf)
                dperf.clear()
                rperf.clear()
            T *= self.params["cooling_rate"]
            self.convergence_history.append(best_cost)
            if noimp > self.params["max_no_improvement"]:
                break
        wall_clock = time.time() - t0
        return best, best_cost, {"metrics": self._finalize(best, it, wall_clock),
                                 "convergence": self.convergence_history}

    def _solution_score(self, solution):
        if not self._is_complete(solution):
            return 1e9
        travel = self.calc.compute_transportation_cost(solution)
        total_pen = 0.0
        for route in solution.values():
            if not route:
                continue
            pen, hard = self._route_penalty(route)
            if hard:
                return 1e9
            total_pen += pen
        return travel + (self.lambda_W + self.lambda_S) * total_pen

    def _enhanced_initial(self):
        order = sorted(range(self.n_customers),
                       key=lambda c: sum(self.instance_data["customers"][c]["demand"].values()),
                       reverse=True)
        solution = {}
        for c in order:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen
                             + 0.5 * self._slosh_component(test))
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                solution[best_vid].insert(best_pos, c)
            else:
                solution[len(solution)] = [c]
        if self._is_trivial(solution):
            return self._generate_initial_solution()
        return solution

    def _slosh_component(self, route):
        demands = {c: self.instance_data["customers"][c]["demand"] for c in route}
        travel = self.calc.compute_transportation_cost({0: route})
        pen, _, _, hard = self.saca.verify_route(
            route, demands, transport_cost=travel, skip_sequence_check=True)
        return 0.0 if hard else pen

    @property
    def ldd_verifier(self):
        return self.saca.ldd_verifier

    def _select(self, weights):
        if random.random() < 0.15:
            return random.choice(list(weights.keys()))
        total = sum(weights.values()) or 1.0
        r = random.random() * total
        acc = 0
        for op, w in weights.items():
            acc += w
            if r <= acc:
                return op
        return list(weights.keys())[-1]

    def _destroy(self, solution, op):
        if op == "random":
            return self._destroy_random(solution)
        if op == "worst":
            return self._destroy_worst(solution)
        if op == "compatibility":
            return self._destroy_compat(solution)
        if op == "imbalance":
            return self._destroy_imbalance(solution)
        return self._destroy_shaw(solution)

    def _destroy_random(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]
        n_remove = max(2, int(len(all_c) * self.params["destruction_rate"]))
        if len(all_c) <= n_remove:
            return new, []
        to_remove = set(random.sample(all_c, n_remove))
        for vid in list(new.keys()):
            new[vid] = [c for c in new[vid] if c not in to_remove]
            if not new[vid]:
                del new[vid]
        return new, list(to_remove)

    def _destroy_worst(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        marginal = {}
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i, c in enumerate(route):
                rw = route[:i] + route[i + 1:]
                if rw:
                    marginal[(vid, c)] = (self.calc.compute_transportation_cost({vid: route}) -
                                          self.calc.compute_transportation_cost({vid: rw}))
        if not marginal:
            return new, []
        k = max(2, int(len(marginal) * self.params["destruction_rate"] * 0.8))
        to_remove = {x[0] for x in sorted(marginal.items(), key=lambda x: x[1], reverse=True)[:k]}
        removed = [c for _, c in to_remove]
        for vid in list(new.keys()):
            new[vid] = [c for c in new[vid] if (vid, c) not in to_remove]
            if not new[vid]:
                del new[vid]
        return new, removed

    def _destroy_shaw(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        all_c = [c for r in new.values() for c in r]
        if len(all_c) < 3:
            return self._destroy_random(solution)
        seed = random.choice(all_c)
        sp = set(self.instance_data["customers"][seed]["demand"].keys())
        similar = []
        for c in all_c:
            if c == seed:
                continue
            ov = len(sp.intersection(set(self.instance_data["customers"][c]["demand"].keys())))
            if ov > 0:
                similar.append((c, ov))
        similar.sort(key=lambda x: x[1], reverse=True)
        k = max(2, min(len(similar), int(len(all_c) * self.params["destruction_rate"] * 0.7)))
        if not similar:
            return new, []
        to_remove = {seed} | {c for c, _ in similar[:k]}
        for vid in list(new.keys()):
            new[vid] = [c for c in new[vid] if c not in to_remove]
            if not new[vid]:
                del new[vid]
        return new, list(to_remove)

    def _destroy_compat(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        inc = defaultdict(list)
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            for i, c1 in enumerate(route):
                for c2 in route[i + 1:]:
                    p1 = set(self.instance_data["customers"][c1]["demand"].keys())
                    p2 = set(self.instance_data["customers"][c2]["demand"].keys())
                    if p1 != p2:
                        inc[vid].extend([c1, c2])
        if not inc:
            return new, []
        target = random.choice(list(inc.keys()))
        unique = list(set(inc[target]))
        k = max(2, min(len(unique), int(len(unique) * 0.5)))
        to_remove = set(random.sample(unique, k))
        new[target] = [c for c in new[target] if c not in to_remove]
        if not new[target]:
            del new[target]
        return new, list(to_remove)

    def _destroy_imbalance(self, solution):
        new = {k: v.copy() for k, v in solution.items()}
        unstable = []
        for vid, route in new.items():
            if len(route) <= 1:
                continue
            pen, hard = self._route_penalty(route)
            if hard or pen > 1e-6:
                unstable.append(vid)
        if not unstable:
            return self._destroy_random(solution)
        target = random.choice(unstable)
        route = new[target]
        for i, c in enumerate(route):
            test = route[:i] + route[i + 1:]
            pen, hard = self._route_penalty(test)
            if not hard:
                new[target] = [x for x in route if x != c]
                if not new[target]:
                    del new[target]
                return new, [c]
        return self._destroy_random(solution)

    def _repair(self, solution, unassigned, op):
        if op == "greedy":
            return self._repair_greedy(solution, unassigned)
        if op == "regret":
            return self._repair_regret(solution, unassigned)
        if op == "stability_aware":
            return self._repair_stab(solution, unassigned)
        return self._repair_balance(solution, unassigned)

    def _repair_greedy(self, solution, unassigned):
        new = {k: v.copy() for k, v in solution.items()}
        rem = unassigned[:]
        random.shuffle(rem)
        for c in rem[:]:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen)
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                new[best_vid].insert(best_pos, c)
                rem.remove(c)
        for c in rem:
            new[len(new)] = [c]
        return new

    def _repair_regret(self, solution, unassigned):
        new = {k: v.copy() for k, v in solution.items()}
        rem = unassigned[:]
        while rem:
            regrets = []
            for c in rem:
                costs = []
                for vid, route in new.items():
                    for pos in range(len(route) + 1):
                        test = route[:pos] + [c] + route[pos:]
                        pen, hard = self._route_penalty(test)
                        if not hard:
                            costs.append(self.calc.compute_transportation_cost({vid: test})
                                         + (self.lambda_W + self.lambda_S) * pen)
                if len(costs) >= 2:
                    costs.sort()
                    regrets.append((costs[1] - costs[0], c))
                elif costs:
                    regrets.append((costs[0], c))
                else:
                    regrets.append((float("inf"), c))
            if not regrets:
                break
            regrets.sort(key=lambda x: x[0], reverse=True)
            if regrets[0][0] == float("inf"):
                c = rem.pop(0)
                new[len(new)] = [c]
                continue
            c = regrets[0][1]
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen)
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                new[best_vid].insert(best_pos, c)
            else:
                new[len(new)] = [c]
            rem.remove(c)
        return new

    def _repair_stab(self, solution, unassigned):
        new = {k: v.copy() for k, v in solution.items()}
        rem = unassigned[:]
        for c in rem[:]:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in new.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    pen, hard = self._route_penalty(test)
                    if hard:
                        continue
                    score = (self.calc.compute_transportation_cost({vid: test})
                             + (self.lambda_W + self.lambda_S) * pen
                             + self.params["stability_weight"] * self._slosh_component(test))
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                new[best_vid].insert(best_pos, c)
                rem.remove(c)
        for c in rem:
            new[len(new)] = [c]
        return new

    def _repair_balance(self, solution, unassigned):
        new = {k: v.copy() for k, v in solution.items()}
        rem = unassigned[:]
        random.shuffle(rem)
        for c in rem[:]:
            min_load = float("inf")
            best_vid, best_pos = None, None
            for vid, route in new.items():
                load = sum(self._extract_demands(route).values())
                if load < min_load:
                    for pos in range(len(route) + 1):
                        test = route[:pos] + [c] + route[pos:]
                        pen, hard = self._route_penalty(test)
                        if not hard:
                            min_load, best_vid, best_pos = load, vid, pos
                            break
            if best_vid is not None:
                new[best_vid].insert(best_pos, c)
                rem.remove(c)
        for c in rem:
            new[len(new)] = [c]
        return new

    def _update_weights(self, dperf, rperf):
        lr = self.params["learning_rate"]
        for op, s in dperf.items():
            if s:
                avg = float(np.mean(s))
                self.destroy_weights[op] = ((1 - lr) * self.destroy_weights[op] +
                                            lr * max(0.1, avg * 2))
        for op, s in rperf.items():
            if s:
                avg = float(np.mean(s))
                self.repair_weights[op] = ((1 - lr) * self.repair_weights[op] +
                                           lr * max(0.1, avg * 2))
        td = sum(self.destroy_weights.values()) or 1.0
        tr = sum(self.repair_weights.values()) or 1.0
        for op in self.destroy_weights:
            self.destroy_weights[op] /= td
        for op in self.repair_weights:
            self.repair_weights[op] /= tr


class SAALNS_NoStability(SAALNSAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, params, stability_weight=0.0)
        self.stability_weight = 0.0
        self.saca.stability_weight = 0.0
        self.params["stability_weight"] = 0.0

    def _slosh_component(self, route):
        return 0.0


class SAALNS_NoAdaptive(SAALNSAlgorithm):
    def _select(self, weights):
        return random.choice(list(weights.keys()))

    def _update_weights(self, dperf, rperf):
        pass


class SAALNS_Sequential(SAALNSAlgorithm):
    def solve(self):
        t0 = time.time()
        routing_solution = self._solve_routing_only()
        final = self._solve_loading_for_routes(routing_solution)
        wall_clock = time.time() - t0
        m = self._finalize(final, len(self.convergence_history), wall_clock)
        return final, m.penalised_cost, {"metrics": m,
                                         "convergence": self.convergence_history}

    def _total_demand(self, customer):
        return sum(self.instance_data["customers"][customer]["demand"].values())

    def _routing_only_cost(self, solution):
        return self.calc.compute_transportation_cost(solution)

    def _route_total(self, route):
        return sum(self._total_demand(c) for c in route)

    def _fits_capacity(self, route):
        return self._route_total(route) <= self.vehicle.max_volume + 1e-6

    def _routing_only_insert(self, order):
        solution = {}
        for c in order:
            best_vid, best_pos, best_score = None, None, float("inf")
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test = route[:pos] + [c] + route[pos:]
                    if not self._fits_capacity(test):
                        continue
                    score = self._routing_only_cost({vid: test})
                    if score < best_score:
                        best_score, best_vid, best_pos = score, vid, pos
            if best_vid is not None:
                solution[best_vid].insert(best_pos, c)
            else:
                solution[len(solution)] = [c]
        return solution

    def _solve_routing_only(self):
        order = sorted(range(self.n_customers),
                       key=lambda c: -self._total_demand(c))
        solution = self._routing_only_insert(order)
        if self._is_trivial(solution):
            ordered = list(range(self.n_customers))
            random.shuffle(ordered)
            solution = self._routing_only_insert(ordered)
        for vid in list(solution.keys()):
            route = solution[vid]
            if len(route) < 3:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if not self._fits_capacity(nr):
                            continue
                        if (self._routing_only_cost({vid: nr}) <
                                self._routing_only_cost({vid: route})):
                            solution[vid] = nr
                            route = nr
                            improved = True
                            break
                    if improved:
                        break
        self.convergence_history.append(self._routing_only_cost(solution))
        return solution

    def _solve_loading_for_routes(self, routing_solution):
        final = {}
        for _, route in routing_solution.items():
            if not route:
                continue
            demands = {c: self.instance_data["customers"][c]["demand"]
                       for c in route}
            travel = self.calc.compute_transportation_cost({0: route})
            pen, _, _, hard = self.saca.verify_route(route, demands,
                                                     transport_cost=travel,
                                                     skip_sequence_check=True)
            if not hard and pen < 1e-6:
                final[len(final)] = route
                continue
            split = self._best_feasible_split(route)
            if split is not None:
                final[len(final)] = split[0]
                final[len(final)] = split[1]
            else:
                for c in route:
                    final[len(final)] = [c]
        return final

    def _best_feasible_split(self, route):
        if len(route) <= 1:
            return None
        best, best_cost = None, float("inf")
        for i in range(1, len(route)):
            r1, r2 = route[:i], route[i:]
            d1 = {c: self.instance_data["customers"][c]["demand"] for c in r1}
            d2 = {c: self.instance_data["customers"][c]["demand"] for c in r2}
            t1 = self.calc.compute_transportation_cost({0: r1})
            t2 = self.calc.compute_transportation_cost({1: r2})
            p1, _, _, h1 = self.saca.verify_route(r1, d1, transport_cost=t1,
                                                   skip_sequence_check=True)
            p2, _, _, h2 = self.saca.verify_route(r2, d2, transport_cost=t2,
                                                   skip_sequence_check=True)
            if h1 or h2:
                continue
            cost_split = (t1 + t2
                          + (self.lambda_W + self.lambda_S) * (p1 + p2))
            if cost_split < best_cost:
                best_cost = cost_split
                best = [r1, r2]
        return best


class SAALNS_NoCache(SAALNSAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, params, stability_weight=stability_weight)
        self.feasibility_cache = {}
        new_saca = SACA(vehicle, instance_data,
                        stability_weight=self.stability_weight,
                        use_cache=False)
        self.saca = new_saca
        self.calc.saca = new_saca

    def _route_penalty(self, route):
        if not route:
            return 0.0, False
        self.feasibility_attempts += 1
        demands_by_cust = {c: self.instance_data["customers"][c]["demand"] for c in route}
        if not self.saca._sequence_feasible(route, demands_by_cust):
            self.feasibility_failures += 1
            return 1e6, True
        travel = self.calc.compute_transportation_cost({0: route})
        penalty, _, _, hard = self.saca.verify_route(
            route, demands_by_cust, transport_cost=travel,
            skip_sequence_check=True)
        if hard or penalty > 1e-6:
            self.feasibility_failures += 1
        return penalty, hard


class SAALNS_NoImbalance(SAALNSAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, params, stability_weight=stability_weight)
        self.destroy_weights = {"random": 1.0, "worst": 1.2,
                                "compatibility": 1.0, "shaw": 1.1}


class SAALNS_SingleProduct(SAALNSAlgorithm):
    def __init__(self, instance_data, vehicle, params=None, stability_weight=0.35):
        super().__init__(instance_data, vehicle, params,
                         stability_weight=stability_weight)
        new_saca = SACA_SingleProduct(
            vehicle, instance_data,
            stability_weight=self.stability_weight,
            use_cache=True)
        self.saca = new_saca
        self.calc.saca = new_saca


# ============================================================================
# SMOKE TEST
# ============================================================================

def smoke_test(benchmark_dir, min_customers=20, tol=1e-6):
    print("\nRunning smoke test ...")
    candidates = []
    for root, _, files in os.walk(benchmark_dir):
        for f in files:
            if f.endswith(".json"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise RuntimeError("No JSON instances found")
    candidates.sort(key=lambda p: (len(json.load(open(p))["customers"]),
                                   os.path.basename(p)))
    smoke_file = None
    for fp in candidates:
        if len(json.load(open(fp))["customers"]) >= min_customers:
            smoke_file = fp
            break
    if smoke_file is None:
        raise RuntimeError(f"No instance with >= {min_customers} customers")

    instance = json.load(open(smoke_file))
    n = len(instance["customers"])
    print(f"  Using {os.path.basename(smoke_file)} ({n} customers)")

    vehicle = create_vehicle_from_instance(instance)
    data = build_instance_data(instance)
    calc = ObjectiveCalculator(data, vehicle)

    trivial = sum(calc.compute_transportation_cost({i: [c]})
                  for i, c in enumerate(range(n)))
    print(f"  Trivial singleton cost: {trivial:.4f}")

    algos = [("Standard_ALNS", StandardALNS),
             ("FDAHS", FDAHSAlgorithm),
             ("GCOF", GCOFAlgorithm),
             ("HSA", HSAAlgorithm),
             ("ACO_2opt", ACO2optAlgorithm)]
    results = {}
    for name, cls in algos:
        seed = _stable_seed("smoke", name)
        random.seed(seed)
        np.random.seed(seed)
        algo = cls(data, vehicle)
        _, cost, _ = algo.solve()
        results[name] = cost
        flag = " <TRIVIAL>" if abs(cost - trivial) < 1e-3 else ""
        print(f"  [smoke] {name:14s} -> {cost:.6f}{flag}")

    names = list(results)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(results[names[i]] - results[names[j]]) < tol:
                print(f"  [WARN] {names[i]} and {names[j]} agree to < {tol}")
    return results


# ============================================================================
# LDD DIAGNOSTIC PROBE
# ============================================================================

def ldd_diagnostic_probe(benchmark_dir):
    print("\n" + "=" * 80)
    print("LDD DIAGNOSTIC PROBE")
    print("=" * 80)
    target = None
    for root, _, files in os.walk(benchmark_dir):
        if "P20_1.json" in files:
            target = os.path.join(root, "P20_1.json")
            break
    if target is None:
        print("  P20_1.json not found; skipping probe.")
        return
    inst = json.load(open(target))
    vehicle = create_vehicle_from_instance(inst)
    data = build_instance_data(inst)
    saca = SACA(vehicle, data)

    n = len(inst["customers"])
    route = list(range(min(4, n)))
    demands = {c: data["customers"][c]["demand"] for c in route}
    travel = vehicle.fixed_cost
    penalty, vol_alloc, prod_alloc, hard = saca.verify_route(
        route, demands, transport_cost=travel)
    print(f"  Route: {route}")
    print(f"  hard-infeasible: {hard}")
    print(f"  penalty: {penalty:.4f}")
    load_kg = {cid: vol_alloc[cid] * 0.8 for cid in vol_alloc if vol_alloc[cid] > 0}
    ldd = saca.ldd_verifier.verify_load(load_kg)
    print(f"  cg_position      : {ldd.cg_position:.4f}")
    print(f"  cg_min / cg_max  : {ldd.cg_min:.4f} / {ldd.cg_max:.4f}")
    print(f"  cg_compliant     : {ldd.cg_compliant}")
    print(f"  stability_margin : {ldd.stability_margin:.2f}%")
    print(f"  violations       : {ldd.violations}")
    print("=" * 80 + "\n")


# ============================================================================
# SENSITIVITY ANALYZER
# ============================================================================

class SensitivityAnalyzer:
    def __init__(self, benchmark_dir, output_dir):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.lambda_W_values = [0.50, 1.00, 1.50]
        self.lambda_S_values = [0.25, 0.50, 0.75]
        self.baseline_W, self.baseline_S = 1.00, 0.50

    def run_sensitivity_analysis(self, max_instances=5, num_runs=2):
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS (lambda_W, lambda_S)")
        print("=" * 80)
        files = []
        for root, _, fs in os.walk(self.benchmark_dir):
            for f in fs:
                if f.endswith(".json"):
                    files.append(os.path.join(root, f))
        files.sort(key=lambda x: os.path.basename(x))
        if max_instances:
            files = files[:max_instances]

        results = []
        total = len(self.lambda_W_values) * len(self.lambda_S_values)
        ctr = 0
        for lw in self.lambda_W_values:
            for ls in self.lambda_S_values:
                ctr += 1
                print(f"  [{ctr}/{total}] W={lw} S={ls}")
                costs, stabs, viols, fleets, cgs, cpus, convs = ([] for _ in range(7))
                n_inf = 0
                for fp in files:
                    try:
                        inst = json.load(open(fp))
                    except Exception:
                        continue
                    vehicle = create_vehicle_from_instance(inst)
                    data = build_instance_data(inst)
                    for run in range(num_runs):
                        seed = _stable_seed("sens", lw, ls, run, fp)
                        random.seed(seed)
                        np.random.seed(seed)
                        params = {
                            "max_iterations": 100,
                            "initial_temperature": 100.0,
                            "cooling_rate": 0.99,
                            "destruction_rate": 0.3,
                            "learning_rate": 0.15,
                            "stability_weight": ls,
                            "tabu_tenure": 10,
                            "adaptive_frequency": 10,
                            "max_no_improvement": 40,
                        }
                        try:
                            algo = SAALNSAlgorithm(data, vehicle, params,
                                                  stability_weight=ls)
                            algo.lambda_W = lw
                            algo.lambda_S = ls
                            t0 = time.time()
                            sol, cost, extra = algo.solve()
                            elapsed = time.time() - t0
                            if sol:
                                m = extra.get("metrics")
                                if m is None:
                                    m = algo.calc.compute_all_metrics(
                                        sol, lambda_W=lw, lambda_S=ls)
                                if np.isfinite(m.penalised_cost):
                                    costs.append(m.penalised_cost)
                                    stabs.append(m.stability_margin)
                                    viols.append(m.ldd_violation_rate)
                                    fleets.append(m.fleet_size)
                                    cgs.append(m.cg_compliance_rate)
                                    cpus.append(elapsed)
                                    convs.append(len(algo.convergence_history))
                                else:
                                    n_inf += 1
                        except Exception:
                            continue
                if costs:
                    results.append(SensitivityResult(
                        lambda_W=lw, lambda_S=ls,
                        avg_cost=float(np.mean(costs)),
                        avg_stability=float(np.mean(stabs)),
                        avg_ldd_violation=float(np.mean(viols)),
                        avg_fleet_size=float(np.mean(fleets)),
                        avg_cg_compliance=float(np.mean(cgs)),
                        avg_cpu_time=float(np.mean(cpus)),
                        std_cost=float(np.std(costs)),
                        std_stability=float(np.std(stabs)),
                        convergence_iterations=int(np.mean(convs)) if convs else 0,
                        parameter_combination_id=f"W{lw}_S{ls}"))
        df = pd.DataFrame([r.__dict__ for r in results])
        out_csv = os.path.join(self.output_dir, "part3_sensitivity",
                               "sensitivity_analysis_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved sensitivity results to {out_csv}")
        if not df.empty:
            print(df.to_string(index=False))
        return df


# ============================================================================
# SOTA COMPARISON
# ============================================================================

def _finite_mean(series):
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) > 0 else float("nan")


def _finite_std(series):
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0


class SOTAComparison:
    def __init__(self, benchmark_dir, output_dir):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.algorithms = {
            "SA-ALNS": SAALNSAlgorithm,
            "SA-ALNS_CostOnly": SAALNS_NoStability,
            "Standard_ALNS": StandardALNS,
            "FDAHS": FDAHSAlgorithm,
            "GCOF": GCOFAlgorithm,
            "HSA": HSAAlgorithm,
            "ACO_2opt": ACO2optAlgorithm,
        }

    def run_all_instances(self, max_instances=None):
        files = []
        for root, _, fs in os.walk(self.benchmark_dir):
            for f in fs:
                if f.endswith(".json"):
                    files.append(os.path.join(root, f))
        files.sort(key=lambda p: (
            len(json.load(open(p))["customers"]), os.path.basename(p)))
        if max_instances:
            files = files[:max_instances]

        smoke_test(self.benchmark_dir)
        ldd_diagnostic_probe(self.benchmark_dir)

        print(f"\nSOTA COMPARISON: {len(files)} instances")
        all_results = []
        for i, fp in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] {os.path.basename(fp)}")
            try:
                all_results.append(self.run_one(fp))
            except Exception as e:
                print(f"  Skipped: {e}")

        df = self._to_df(all_results)
        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, "sota_comparison_results.csv"),
                      index=False)
        return df

    def run_one(self, fp, num_runs=1):
        instance = json.load(open(fp))
        vehicle = create_vehicle_from_instance(instance)
        data = build_instance_data(instance)
        n_expected = len(instance["customers"])
        out = {"instance": instance["name"],
               "n_customers": n_expected,
               "n_compartments": instance.get("n_compartments",
                                               len(instance["vehicle"]["compartments"])),
               "layer": instance.get("layer", "Unknown"),
               "n_products": instance.get("n_products", 4),
               "vehicle_type": "Multi-Compartment",
               "results": {}}
        for name, cls in self.algorithms.items():
            print(f"    {name}...", end="", flush=True)
            costs, times, t_milp, metrics = [], [], [], []
            n_infeasible = 0
            for run in range(num_runs):
                seed = _stable_seed(name, run, instance["name"])
                random.seed(seed)
                np.random.seed(seed)
                try:
                    algo = cls(data, vehicle)
                    t0 = time.time()
                    sol, cost, extra = algo.solve()
                    elapsed = time.time() - t0
                    if sol:
                        n_in = sum(len(r) for r in sol.values() if r)
                        if n_in != n_expected:
                            print(f" [ALERT {name}: {n_in}/{n_expected} customers]",
                                  end="")
                        m = extra.get("metrics")
                        if m is None:
                            m = algo.calc.compute_all_metrics(
                                sol, lambda_W=algo.lambda_W, lambda_S=algo.lambda_S)
                        if np.isfinite(m.penalised_cost):
                            costs.append(m.penalised_cost)
                            times.append(elapsed)
                            t_milp.append(m.milp_time)
                            metrics.append(m)
                        else:
                            n_infeasible += 1
                            times.append(elapsed)
                            t_milp.append(m.milp_time)
                            metrics.append(m)
                except Exception as e:
                    print(f" [err {e}]", end="")
            if metrics:
                avg = self._avg_metrics(metrics)
                mean_calls = float(avg.milp_calls)
                mean_milp_time = float(np.mean(t_milp)) if t_milp else 0.0
                per_call = (mean_milp_time / mean_calls) if mean_calls > 0 else float("nan")
                finite_costs = [c for c in costs if np.isfinite(c)]
                infeas_rate = 100.0 * n_infeasible / max(len(metrics), 1)
                out["results"][name] = {
                    "avg_cost": float(np.mean(finite_costs)) if finite_costs else float("inf"),
                    "std_cost": float(np.std(finite_costs, ddof=1)) if len(finite_costs) > 1 else 0.0,
                    "best_cost": float(min(finite_costs)) if finite_costs else float("inf"),
                    "worst_cost": float(max(finite_costs)) if finite_costs else float("inf"),
                    "n_finite_runs": len(finite_costs),
                    "n_infeasible_runs": n_infeasible,
                    "infeasibility_rate_pct": infeas_rate,
                    "avg_routing_time": float(np.mean(times)) if times else 0.0,
                    "avg_milp_time": mean_milp_time,
                    "avg_total_time": float(np.mean(times)) if times else 0.0,
                    "avg_milp_calls": avg.milp_calls,
                    "avg_milp_time_per_call": per_call,
                    "milp_time_fraction": (
                        mean_milp_time /
                        max(float(np.mean(times)) if times else 1e-9, 1e-9)),
                    "avg_saca_calls": avg.milp_calls,
                    "avg_cache_hit_rate": avg.cache_hit_rate,
                    "avg_route_penalty_cache_hits": avg.route_penalty_cache_hits,
                    "avg_route_penalty_cache_hit_rate": avg.route_penalty_cache_hit_rate,
                    "avg_ldd_violation_rate": avg.ldd_violation_rate,
                    "avg_stability": avg.stability_margin,
                    "avg_slosh_risk": avg.slosh_risk_score,
                    "avg_cg_compliance_rate": avg.cg_compliance_rate,
                    "avg_cg_deviation": avg.cg_deviation_from_optimum,
                    "avg_volume_utilization": avg.volume_utilization,
                    "avg_fleet_size": avg.fleet_size,
                    "avg_feasibility_failures": avg.feasibility_failures,
                    "avg_feasibility_attempts": avg.feasibility_attempts,
                    "feasibility_failure_rate": (
                        100.0 * avg.feasibility_failures /
                        max(1, avg.feasibility_attempts)),
                    "avg_penalty": avg.avg_penalty,
                    "avg_n_total_routes": float(avg.n_total_routes),
                    "avg_n_verified_routes": float(avg.n_verified_routes),
                }
                c = out["results"][name]["avg_cost"]
                print(f" done ({c:.2f})" if np.isfinite(c) else " done (inf)", flush=True)
            else:
                print(" failed", flush=True)
        return out

    def _avg_metrics(self, metrics):
        if not metrics:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        avg = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        attrs = ["total_distance", "transportation_cost", "fleet_size",
                 "fleet_utilization", "volume_utilization", "ldd_violation_rate",
                 "stability_margin", "minimum_load_ratio", "slosh_risk_score",
                 "cpu_time", "saca_calls", "cache_hit_rate",
                 "iterations_to_convergence", "milp_calls", "cache_hits",
                 "milp_time", "routing_time", "feasibility_failures",
                 "feasibility_attempts", "cg_position_mean", "cg_position_std",
                 "cg_position_min", "cg_position_max", "cg_compliance_rate",
                 "cg_deviation_from_optimum", "heeling_moment_estimate",
                 "lateral_acceleration_margin", "penalised_cost", "avg_penalty",
                 "n_total_routes", "n_verified_routes",
                 "route_penalty_cache_hits", "route_penalty_cache_hit_rate"]
        n = len(metrics)
        for m in metrics:
            for a in attrs:
                setattr(avg, a, getattr(avg, a) + getattr(m, a, 0))
        for a in attrs:
            setattr(avg, a, getattr(avg, a) / n)
        for a in ["cg_compliance_rate", "fleet_utilization",
                  "volume_utilization", "ldd_violation_rate",
                  "stability_margin", "minimum_load_ratio", "cache_hit_rate",
                  "route_penalty_cache_hit_rate"]:
            setattr(avg, a, max(0.0, min(100.0, getattr(avg, a))))
        return avg

    def _to_df(self, results):
        rows = []
        for r in results:
            if not r["results"]:
                continue
            base = {k: r[k] for k in ("instance", "n_customers", "n_compartments",
                                       "layer", "n_products", "vehicle_type")}
            for algo, d in r["results"].items():
                row = dict(base)
                row["algorithm"] = algo
                row.update(d)
                rows.append(row)
        return pd.DataFrame(rows)


class AblationComparison(SOTAComparison):
    def __init__(self, benchmark_dir, output_dir):
        super().__init__(benchmark_dir, output_dir)
        self.algorithms = {
            "SA-ALNS": SAALNSAlgorithm,
            "SA-ALNS_NS": SAALNS_NoStability,
            "SA-ALNS_NA": SAALNS_NoAdaptive,
            "SA-ALNS_SEQ": SAALNS_Sequential,
            "SA-ALNS_NC": SAALNS_NoCache,
            "SA-ALNS_IMB": SAALNS_NoImbalance,
        }

    def run_all_instances(self, max_instances=None):
        files = []
        for root, _, fs in os.walk(self.benchmark_dir):
            for f in fs:
                if f.endswith(".json"):
                    files.append(os.path.join(root, f))
        files.sort(key=lambda p: (
            len(json.load(open(p))["customers"]), os.path.basename(p)))
        if max_instances:
            files = files[:max_instances]
        print(f"\nABLATION STUDY: {len(files)} instances")
        all_results = []
        for i, fp in enumerate(files):
            print(f"\n[{i+1}/{len(files)}] {os.path.basename(fp)}")
            try:
                all_results.append(self.run_one(fp))
            except Exception as e:
                print(f"  Skipped: {e}")
        df = self._to_df(all_results)
        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, "ablation_study_results.csv"),
                      index=False)
        return df


# ============================================================================
# MULTI-COMPARTMENT COMPARISON
# ============================================================================

def _metrics_infeasible(m):
    if m is None:
        return True
    return not np.isfinite(m.penalised_cost)


def run_multicomp_comparison(sota_df, benchmark_dir, output_dir):
    print("\n" + "=" * 80)
    print("MULTI vs SINGLE PRODUCT (Option B1)")
    print("=" * 80)
    out_csv = os.path.join(output_dir, "part4_statistics",
                           "multicomp_benefits.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if sota_df is None or sota_df.empty or "algorithm" not in sota_df.columns:
        print("  sota_df empty or malformed; writing empty CSV.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    sa_rows = sota_df[sota_df["algorithm"] == "SA-ALNS"]
    if sa_rows.empty:
        print("  No 'SA-ALNS' rows; writing empty CSV.")
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()

    print(f"  Instances: {len(sa_rows)}")
    rows, errors = [], 0
    for _, r in sa_rows.iterrows():
        name = r["instance"]
        fp = None
        for root, _, fs in os.walk(benchmark_dir):
            if f"{name}.json" in fs:
                fp = os.path.join(root, f"{name}.json")
                break
        if fp is None:
            continue
        try:
            inst = json.load(open(fp))
            n_cmp = inst.get("n_compartments",
                             len(inst["vehicle"]["compartments"]))
            data = build_instance_data(inst)
            seed = _stable_seed("multicomp", name)

            veh_A = create_vehicle_from_instance(inst, multi_compartment=True)
            random.seed(seed); np.random.seed(seed)
            algo_A = SAALNSAlgorithm(data, veh_A)
            sol_A, _, extra_A = algo_A.solve()
            m_A = extra_A.get("metrics") or \
                algo_A.calc.compute_all_metrics(
                    sol_A, lambda_W=algo_A.lambda_W, lambda_S=algo_A.lambda_S)

            veh_B = create_vehicle_from_instance(inst, multi_compartment=True)
            random.seed(seed); np.random.seed(seed)
            algo_B = SAALNS_SingleProduct(data, veh_B)
            sol_B, _, extra_B = algo_B.solve()
            m_B = extra_B.get("metrics") or \
                algo_B.calc.compute_all_metrics(
                    sol_B, lambda_W=algo_B.lambda_W, lambda_S=algo_B.lambda_S)

            veh_C = create_vehicle_from_instance(inst, multi_compartment=False)
            random.seed(seed); np.random.seed(seed)
            algo_C = SAALNSAlgorithm(data, veh_C)
            sol_C, _, extra_C = algo_C.solve()
            m_C = extra_C.get("metrics") or \
                algo_C.calc.compute_all_metrics(
                    sol_C, lambda_W=algo_C.lambda_W, lambda_S=algo_C.lambda_S)

            def pct(base, other):
                if not np.isfinite(base) or not np.isfinite(other):
                    return float("nan")
                if abs(other) < 1e-9:
                    return float("nan")
                return (other - base) / other * 100.0

            rows.append({
                "instance": name,
                "n_customers": r["n_customers"],
                "layer": r.get("layer", "Unknown"),
                "n_compartments": n_cmp,
                "A_full_cost": m_A.penalised_cost,
                "A_full_fleet": m_A.fleet_size,
                "A_full_util": m_A.volume_utilization,
                "A_full_cg": m_A.cg_compliance_rate,
                "A_full_ldd": m_A.ldd_violation_rate,
                "A_full_slosh": m_A.slosh_risk_score,
                "A_full_feasible": not _metrics_infeasible(m_A),
                "B_nomix_cost": m_B.penalised_cost,
                "B_nomix_fleet": m_B.fleet_size,
                "B_nomix_util": m_B.volume_utilization,
                "B_nomix_cg": m_B.cg_compliance_rate,
                "B_nomix_ldd": m_B.ldd_violation_rate,
                "B_nomix_slosh": m_B.slosh_risk_score,
                "B_nomix_feasible": not _metrics_infeasible(m_B),
                "C_forced_cost": m_C.penalised_cost,
                "C_forced_fleet": m_C.fleet_size,
                "C_forced_feasible": not _metrics_infeasible(m_C),
                "cost_improvement_AB": pct(m_A.penalised_cost, m_B.penalised_cost),
                "fleet_reduction_AB": pct(m_A.fleet_size, m_B.fleet_size),
                "util_improvement_AB": pct(m_A.volume_utilization,
                                           m_B.volume_utilization),
                "cost_improvement_AC": pct(m_A.penalised_cost, m_C.penalised_cost),
                "fleet_reduction_AC": pct(m_A.fleet_size, m_C.fleet_size),
            })
            print(f"    {name}: A={m_A.penalised_cost:.1f} "
                  f"B={m_B.penalised_cost:.1f} "
                  f"C={'inf' if not np.isfinite(m_C.penalised_cost) else f'{m_C.penalised_cost:.1f}'}")
        except Exception as e:
            errors += 1
            print(f"  Error on {name}: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    print(f"\n  Rows: {len(df)}, errors: {errors}")

    if not df.empty:
        n_inf_B = int((~np.isfinite(df["B_nomix_cost"])).sum())
        n_inf_C = int((~np.isfinite(df["C_forced_cost"])).sum())
        if n_inf_B > 0.2 * len(df):
            print("\n" + "!" * 80)
            print(f"  [WARNING] B_nomix_cost is inf on {n_inf_B}/{len(df)} "
                  f"instances ({100.0*n_inf_B/len(df):.1f}%).")
            print("!" * 80)
        if n_inf_C > 0:
            print(f"  [info] C_forced_cost is inf on {n_inf_C}/{len(df)} "
                  f"instances ({100.0*n_inf_C/len(df):.1f}%).")

        finite_B = df[np.isfinite(df["B_nomix_cost"])]
        if not finite_B.empty:
            print(f"\n  ---- A vs B (on the {len(finite_B)} finite-B instances) ----")
            print(f"  Mean cost improvement  : {_finite_mean(finite_B['cost_improvement_AB']):+.2f}%")
            print(f"  Mean fleet reduction   : {_finite_mean(finite_B['fleet_reduction_AB']):+.2f}%")
            print(f"  Mean util improvement  : {_finite_mean(finite_B['util_improvement_AB']):+.2f}%")
            if len(finite_B) > 0:
                wins_A = int((finite_B['cost_improvement_AB'] > 0).sum())
                wins_B = int((finite_B['cost_improvement_AB'] < 0).sum())
                print(f"  A beats B on cost      : {wins_A}/{len(finite_B)}")
                print(f"  B beats A on cost      : {wins_B}/{len(finite_B)}")
        print(f"  A feasible: {df['A_full_feasible'].sum()}/{len(df)}")
        print(f"  B feasible: {df['B_nomix_feasible'].sum()}/{len(df)}")
        print(f"  C feasible: {df['C_forced_feasible'].sum()}/{len(df)}")

    df.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv}")
    return df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def _wilcoxon_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return float("nan")
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except Exception:
        return float("nan")


def _mannwhitney_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        _, p = mannwhitneyu(a, b)
        return float(p)
    except Exception:
        return float("nan")


def run_statistical_tests(sota_df, output_dir):
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (SOTA, inf-filtered)")
    print("=" * 80)
    if sota_df is None or sota_df.empty:
        return None
    sa = sota_df[sota_df["algorithm"] == "SA-ALNS"]
    if sa.empty:
        return None
    rows = []
    scost = sa["avg_cost"].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(scost) < 5:
        print("  Insufficient finite SA-ALNS costs for statistics.")
        return None
    for algo in ["SA-ALNS_CostOnly", "Standard_ALNS", "FDAHS",
                 "GCOF", "HSA", "ACO_2opt"]:
        sub = sota_df[sota_df["algorithm"] == algo]
        if sub.empty:
            continue
        c = sub["avg_cost"].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(c) < 5:
            print(f"  {algo}: insufficient finite costs ({len(c)}); skipped.")
            continue
        wp = _wilcoxon_safe(scost, c)
        mp = _mannwhitney_safe(scost, c)
        m1, m2 = float(np.mean(scost)), float(np.mean(c))
        s1 = float(np.std(scost, ddof=1)) if len(scost) > 1 else 0.0
        s2 = float(np.std(c, ddof=1)) if len(c) > 1 else 0.0
        pooled = math.sqrt(((len(scost) - 1) * s1 ** 2 + (len(c) - 1) * s2 ** 2) /
                           max(len(scost) + len(c) - 2, 1))
        d = (m1 - m2) / pooled if pooled > 0 else 0
        imp = (m2 - m1) / m2 * 100 if m2 > 0 else 0
        rows.append({"Algorithm": SOTA_DISPLAY_NAMES.get(algo, algo),
                     "SAALNS_n_finite": len(scost),
                     "Comp_n_finite": len(c),
                     "SAALNS_mean": m1, "Comp_mean": m2,
                     "Improvement_pct": imp, "Wilcoxon_p": wp,
                     "MannWhitney_p": mp, "Cohen_d": d,
                     "Significant_5pct": bool(np.isfinite(mp) and mp < 0.05)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "part4_statistics",
                           "statistical_tests_results.csv"), index=False)
    print(df.to_string(index=False))
    return df


def paired_ablation_tests(output_dir):
    print("\n" + "=" * 80)
    print("PAIRED ABLATION WILCOXON TESTS (inf-filtered)")
    print("=" * 80)
    csv = os.path.join(output_dir, "ablation_study_results.csv")
    if not os.path.exists(csv):
        print(f"  Missing {csv}; skipping.")
        return None
    df = pd.read_csv(csv)
    if df.empty or "avg_cost" not in df.columns:
        return None
    df["avg_cost"] = df["avg_cost"].replace([np.inf, -np.inf], np.nan)
    pivot = df.pivot_table(index="instance", columns="algorithm",
                           values="avg_cost", aggfunc="mean")
    if "SA-ALNS" not in pivot.columns:
        return None
    rows = []
    for algo in pivot.columns:
        if algo == "SA-ALNS":
            continue
        pair = pivot[["SA-ALNS", algo]].dropna()
        if len(pair) < 5:
            continue
        p = _wilcoxon_safe(pair["SA-ALNS"].values, pair[algo].values)
        diff = pair["SA-ALNS"] - pair[algo]
        rows.append({
            "Variant": ABLATION_DISPLAY_NAMES.get(algo, algo),
            "n_pairs": len(pair),
            "SAALNS_mean": float(pair["SA-ALNS"].mean()),
            "Variant_mean": float(pair[algo].mean()),
            "Mean_delta": float(diff.mean()),
            "Median_delta": float(diff.median()),
            "Wilcoxon_p": p,
            "SAALNS_wins": int((diff < 0).sum()),
            "Variant_wins": int((diff > 0).sum()),
            "Ties": int((np.abs(diff) < 1e-9).sum()),
            "Significant_5pct": bool(np.isfinite(p) and p < 0.05),
        })
    if not rows:
        return None
    result = pd.DataFrame(rows).sort_values("Wilcoxon_p", na_position="last")
    out = os.path.join(output_dir, "part4_statistics",
                       "ablation_paired_tests.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print(f"  Wrote {out}")
    return result


def paired_multicomp_AB(output_dir):
    print("\n" + "=" * 80)
    print("PAIRED TEST: MULTICOMP A vs B (inf-filtered)")
    print("=" * 80)
    csv = os.path.join(output_dir, "part4_statistics", "multicomp_benefits.csv")
    if not os.path.exists(csv):
        print(f"  Missing {csv}; skipping.")
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None

    results = []
    for label, ca, cb in [("Cost", "A_full_cost", "B_nomix_cost"),
                          ("Fleet", "A_full_fleet", "B_nomix_fleet")]:
        if ca not in df.columns or cb not in df.columns:
            continue
        pair = df[[ca, cb]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 5:
            continue
        p = _wilcoxon_safe(pair[ca].values, pair[cb].values)
        diff = pair[ca] - pair[cb]
        results.append({
            "Metric": label, "n_pairs": len(pair),
            "A_mean": float(pair[ca].mean()), "B_mean": float(pair[cb].mean()),
            "Mean_delta": float(diff.mean()), "Median_delta": float(diff.median()),
            "Wilcoxon_p": p,
            "A_wins": int((diff < 0).sum()),
            "B_wins": int((diff > 0).sum()),
            "Ties": int((np.abs(diff) < 1e-9).sum()),
            "Significant_5pct": bool(np.isfinite(p) and p < 0.05),
        })
    if not results:
        return None
    out_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "part4_statistics",
                            "multicomp_paired_test_AB.csv")
    out_df.to_csv(out_path, index=False)
    print(out_df.to_string(index=False))
    print(f"  Wrote {out_path}")
    cost_row = next((r for r in results if r["Metric"] == "Cost"), None)
    if cost_row and cost_row["A_mean"] > cost_row["B_mean"]:
        print("\n  [VERDICT] On this benchmark, the FULL multi-compartment")
        print("  model (A) does NOT beat the NO_MIX baseline (B) on cost.")
        print("  The 'mixing benefit' claim is NOT supported by this data.")
    return out_df


# ============================================================================
# FIGURES
# ============================================================================

def fig_sota_cost(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_cost"]
             .agg(["mean", "std"]).reset_index().dropna())
        if g.empty:
            continue
        lw = 3 if algo == "SA-ALNS" else 2
        ax.errorbar(g["n_customers"], g["mean"], yerr=g["std"].fillna(0),
                    marker=MARKERS.get(algo, "o"),
                    color=COLORS_SOTA.get(algo, "#333"),
                    label=SOTA_DISPLAY_NAMES.get(algo, algo),
                    markersize=8, capsize=3, linewidth=lw)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Average Penalised Cost ($)")
    ax.set_title("SOTA Comparison: Penalised Cost vs Instance Size (finite runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part1_sota", "Figure_1_Cost.pdf"))
    plt.close(fig)


def fig_sota_cg(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_cg_compliance_rate"]
             .mean().reset_index().dropna())
        if g.empty:
            continue
        ax.plot(g["n_customers"], g["avg_cg_compliance_rate"], marker="o",
                color=COLORS_SOTA.get(algo, "#333"),
                label=SOTA_DISPLAY_NAMES.get(algo, algo), linewidth=2)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("CG Compliance Rate (%)")
    ax.set_title("CG Compliance vs Instance Size (over verified routes)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part1_sota", "Figure_2_CG.pdf"))
    plt.close(fig)


def fig_sota_timing(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_total_time"]
             .mean().reset_index().dropna())
        if g.empty:
            continue
        ax.plot(g["n_customers"], g["avg_total_time"],
                marker="o", color=COLORS_SOTA.get(algo, "#333"),
                label=SOTA_DISPLAY_NAMES.get(algo, algo), linewidth=2)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Wall-clock Time (s)")
    ax.set_yscale("log")
    ax.set_title("Wall-clock Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part1_sota", "Figure_3_Timing.pdf"))
    plt.close(fig)


def fig_sota_ldd(df, output_dir):
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        g = (sub.replace([np.inf, -np.inf], np.nan)
             .groupby("n_customers")["avg_ldd_violation_rate"]
             .agg(["mean", "std"]).reset_index().dropna())
        if g.empty:
            continue
        lw = 3 if algo == "SA-ALNS" else 2
        ax.errorbar(g["n_customers"], g["mean"], yerr=g["std"].fillna(0),
                    marker=MARKERS.get(algo, "o"),
                    color=COLORS_SOTA.get(algo, "#333"),
                    label=SOTA_DISPLAY_NAMES.get(algo, algo),
                    markersize=8, capsize=3, linewidth=lw)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("LDD Violation Rate (%)")
    ax.set_title("LDD Violation Rate vs. Instance Size")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part1_sota", "Figure_4_LDD.pdf"))
    plt.close(fig)


def fig_ablation_cost(df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for algo in df["algorithm"].unique():
        sub = df_clean[df_clean["algorithm"] == algo]
        if sub.empty:
            continue
        g = sub.groupby("n_customers")["avg_cost"].agg(["mean", "std"]).reset_index().dropna()
        if g.empty:
            continue
        lw = 3 if algo == "SA-ALNS" else 2
        ax.errorbar(g["n_customers"], g["mean"], yerr=g["std"].fillna(0),
                    marker=MARKERS.get(algo, "o"),
                    color=COLORS_ABLATION.get(algo, "#333"),
                    label=ABLATION_DISPLAY_NAMES.get(algo, algo),
                    markersize=8, capsize=3, linewidth=lw)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("Average Penalised Cost ($)")
    ax.set_title("Ablation Study: Penalised Cost vs Instance Size (finite runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part2_ablation",
                                   "Figure_Ablation_Cost.pdf"))
    plt.close(fig)


def fig_ablation_safety(df, output_dir):
    if df is None or df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("avg_cg_compliance_rate", "CG Compliance (%)", (0, 105)),
               ("avg_ldd_violation_rate", "LDD Violation Rate (%)", (0, 105)),
               ("avg_slosh_risk", "Slosh Risk Score", None)]
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        if col not in df_clean.columns:
            ax.set_visible(False)
            continue
        g = df_clean.groupby(["algorithm", "n_customers"])[col].mean().reset_index()
        for algo in df_clean["algorithm"].unique():
            sub = g[g["algorithm"] == algo]
            if sub.empty:
                continue
            ax.plot(sub["n_customers"], sub[col],
                    marker=MARKERS.get(algo, "o"),
                    color=COLORS_ABLATION.get(algo, "#333"),
                    label=ABLATION_DISPLAY_NAMES.get(algo, algo),
                    linewidth=2)
        ax.set_xlabel("Number of Customers")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Ablation Study: Safety Metrics", fontsize=12)
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part2_ablation",
                                   "Figure_Ablation_Safety.pdf"))
    plt.close(fig)


def fig_multicomp_AB(df, output_dir):
    if df is None or df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    df_clean = df.replace([np.inf, -np.inf], np.nan)

    gA = df_clean.groupby("n_customers")["A_full_cost"].mean()
    gB = df_clean.groupby("n_customers")["B_nomix_cost"].mean()
    axes[0].plot(gA.index, gA.values, marker="o", color="#2E86AB",
                 label="A: FULL (mixing)", linewidth=2)
    axes[0].plot(gB.index, gB.values, marker="s", color="#E76F51",
                 label="B: NO_MIX", linewidth=2)
    axes[0].set_xlabel("Number of Customers")
    axes[0].set_ylabel("Mean Penalised Cost ($)")
    axes[0].set_title("Cost: Mixing Benefit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    gA = df_clean.groupby("n_customers")["A_full_fleet"].mean()
    gB = df_clean.groupby("n_customers")["B_nomix_fleet"].mean()
    axes[1].plot(gA.index, gA.values, marker="o", color="#2E86AB",
                 label="A: FULL", linewidth=2)
    axes[1].plot(gB.index, gB.values, marker="s", color="#E76F51",
                 label="B: NO_MIX", linewidth=2)
    axes[1].set_xlabel("Number of Customers")
    axes[1].set_ylabel("Mean Fleet Size")
    axes[1].set_title("Fleet Size: Mixing Benefit")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    data = df_clean["cost_improvement_AB"].dropna()
    if len(data) > 0:
        axes[2].hist(data, bins=15, color="#2E86AB", edgecolor="black", alpha=0.8)
        axes[2].axvline(0, color="red", linestyle="--", linewidth=1.5,
                        label="No benefit")
        axes[2].axvline(data.mean(), color="black", linewidth=1.5,
                        label=f"Mean = {data.mean():+.1f}%")
        axes[2].set_xlabel("Cost Improvement A over B (%)")
        axes[2].set_ylabel("Instances")
        axes[2].set_title("Distribution of Mixing Benefit")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Multi-Compartment Mixing Benefit", fontsize=12)
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part2_ablation",
                                   "Figure_Multicomp_AB.pdf"))
    plt.close(fig)


def fig_multicomp_fleet_legacy(multicomp_df, output_dir):
    if multicomp_df is None or multicomp_df.empty:
        return
    if "fleet_reduction_AB" not in multicomp_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    df_clean = multicomp_df.replace([np.inf, -np.inf], np.nan)
    g = (df_clean.groupby("n_compartments")["fleet_reduction_AB"]
         .agg(["mean", "std"]).reset_index().dropna())
    if g.empty:
        plt.close(fig)
        return
    ax.bar(g["n_compartments"].astype(str), g["mean"],
           yerr=g["std"].fillna(0), capsize=4, color="#2E86AB", edgecolor="black")
    ax.set_xlabel("Number of Compartments")
    ax.set_ylabel("Fleet Reduction A over B (%)")
    ax.set_title("Multi-Compartment Fleet Reduction (Option B1)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    safe_savefig(fig, os.path.join(output_dir, "part2_ablation",
                                   "Figure_Compartment_Fleet.pdf"))
    plt.close(fig)


def generate_all_figures(sota_df, ablation_df, multicomp_df, output_dir):
    print("\nGENERATING FIGURES")
    if sota_df is not None and not sota_df.empty:
        for fn in (fig_sota_cost, fig_sota_cg, fig_sota_timing, fig_sota_ldd):
            try:
                fn(sota_df, output_dir)
            except Exception as e:
                print(f"  {fn.__name__} error: {e}")
    if ablation_df is not None and not ablation_df.empty:
        for fn in (fig_ablation_cost, fig_ablation_safety):
            try:
                fn(ablation_df, output_dir)
            except Exception as e:
                print(f"  {fn.__name__} error: {e}")
    if multicomp_df is not None and not multicomp_df.empty:
        for fn in (fig_multicomp_AB, fig_multicomp_fleet_legacy):
            try:
                fn(multicomp_df, output_dir)
            except Exception as e:
                print(f"  {fn.__name__} error: {e}")


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def print_diagnostics(sota_df, saca):
    print("\n" + "=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)
    if sota_df is None or sota_df.empty:
        print("No SOTA data.")
        return
    for algo in sota_df["algorithm"].unique():
        sub = sota_df[sota_df["algorithm"] == algo]
        if sub.empty:
            continue
        df_clean = sub.replace([np.inf, -np.inf], np.nan)
        avg_n = sub["n_customers"].mean()
        avg_fleet = df_clean["avg_fleet_size"].mean()
        ratio = avg_fleet / avg_n if avg_n > 0 else 0
        ff_rate = df_clean["feasibility_failure_rate"].mean() \
            if "feasibility_failure_rate" in sub.columns else 0.0
        avg_pen = df_clean["avg_penalty"].mean() if "avg_penalty" in sub.columns else 0.0
        saca_hit = df_clean["avg_cache_hit_rate"].mean() \
            if "avg_cache_hit_rate" in sub.columns else 0.0
        route_hit = df_clean["avg_route_penalty_cache_hit_rate"].mean() \
            if "avg_route_penalty_cache_hit_rate" in sub.columns else 0.0
        milp_frac = df_clean["milp_time_fraction"].mean() \
            if "milp_time_fraction" in sub.columns else 0.0
        milp_calls = df_clean["avg_milp_calls"].mean() \
            if "avg_milp_calls" in sub.columns else 0.0
        infeas = df_clean["infeasibility_rate_pct"].mean() \
            if "infeasibility_rate_pct" in sub.columns else 0.0
        avg_cost = df_clean["avg_cost"].mean()
        n_total = df_clean["avg_n_total_routes"].mean() \
            if "avg_n_total_routes" in sub.columns else float("nan")
        n_verif = df_clean["avg_n_verified_routes"].mean() \
            if "avg_n_verified_routes" in sub.columns else float("nan")
        print(f"\n{algo}")
        print(f"  avg penalised cost: ${avg_cost:.2f} (finite runs)")
        print(f"  run-infeasibility : {infeas:.1f}% of runs returned inf")
        print(f"  avg fleet size    : {avg_fleet:.2f}   "
              f"(fleet/customer = {ratio:.2f})")
        print(f"  routes verified   : {n_verif:.2f} / {n_total:.2f} "
              f"({100.0*n_verif/max(n_total,1e-9):.1f}%)")
        print(f"  avg CG compliance : {df_clean['avg_cg_compliance_rate'].mean():.2f}% "
              f"(over verified routes)")
        print(f"  avg stability     : {df_clean['avg_stability'].mean():.2f}% "
              f"(mean over verified routes)")
        print(f"  avg LDD violation : {df_clean['avg_ldd_violation_rate'].mean():.2f}%")
        if "avg_slosh_risk" in sub.columns:
            print(f"  avg slosh penalty : {df_clean['avg_slosh_risk'].mean():.3f}")
        print(f"  avg route penalty : {avg_pen:.3f}")
        print(f"  hard-infeasible %: {ff_rate:.1f}% (during search)")
        print(f"  route-penalty cache hit : {route_hit:.1f}%  "
              f"(quote this in the manuscript)")
        print(f"  SACA-level cache hit    : {saca_hit:.1f}%  "
              f"(saturates at 100 because LP fallback is never called)")
        print(f"  avg wall-clock    : {df_clean['avg_total_time'].mean():.4f}s")
        print(f"  MILP fraction     : {milp_frac*100:.1f}%")
        print(f"  avg MILP calls    : {milp_calls:.0f}")
    print("\nFailure-reason breakdown (SACA diagnostic):")
    total = sum(saca.failure_reasons.values())
    if total == 0:
        print("  (no failures recorded)")
    else:
        for reason, count in sorted(saca.failure_reasons.items(),
                                    key=lambda kv: -kv[1]):
            print(f"  {reason:22s}: {count:5d}  ({100.0*count/total:.1f}%)")


def probe_saca_failure_reasons(rep_fp, n_probes=200):
    inst = json.load(open(rep_fp))
    veh = create_vehicle_from_instance(inst)
    diag_saca = SACA(veh)
    data = build_instance_data(inst)
    n = len(inst["customers"])
    for _ in range(n_probes):
        k = random.randint(1, min(n, 9))
        route = random.sample(range(n), k)
        demands = {i: data["customers"][i]["demand"] for i in route}
        calc_tmp = ObjectiveCalculator(data, veh, saca=diag_saca)
        travel = calc_tmp.compute_transportation_cost({0: route})
        diag_saca.verify_route(route, demands, transport_cost=travel)
    return diag_saca


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the DSC-CTRP SA-ALNS experiments.")
    parser.add_argument("--instances", default=DEFAULT_INSTANCE_DIR,
                        help="Directory containing (or to receive) the JSON "
                             "benchmark instances.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Directory in which to write results and figures.")
    parser.add_argument("--max-instances", type=int, default=30,
                        help="Number of instances to use for the SOTA and "
                             "ablation runs (default: 30, as in the manuscript).")
    args = parser.parse_args()

    print("=" * 80)
    print("DSC-CTRP v8.9.1 — SA-ALNS with deterministic seed")
    print("=" * 80)
    print("Reproducibility: seeds derived via SHA-1 (_stable_seed).")
    print("Cache reporting: route-penalty hit rate is the manuscript figure;")
    print("                 SACA-level hit rate saturates and is retained only")
    print("                 for backward compatibility.")

    os.makedirs(args.output, exist_ok=True)
    for subdir in ("part1_sota", "part2_ablation", "part3_sensitivity",
                   "part4_statistics"):
        os.makedirs(os.path.join(args.output, subdir), exist_ok=True)

    print(f"\nInstances: {args.instances}")
    print(f"Output:    {args.output}")

    ensure_instances(args.instances)

    print("\n" + "=" * 80)
    print("PART 1: SOTA COMPARISON")
    print("=" * 80)
    sota = SOTAComparison(args.instances, args.output)
    sota_df = sota.run_all_instances(max_instances=args.max_instances)

    rep_fp = None
    for root, _, fs in os.walk(args.instances):
        for f in fs:
            if f.endswith(".json"):
                rep_fp = os.path.join(root, f)
                break
        if rep_fp:
            break
    if rep_fp:
        diag_saca = probe_saca_failure_reasons(rep_fp, n_probes=200)
        print_diagnostics(sota_df, diag_saca)
    else:
        print_diagnostics(sota_df, SACA(create_vehicle_from_instance(
            generate_single_instance(20, "Paixao", 1))))

    print("\n" + "=" * 80)
    print("PART 2: ABLATION STUDY")
    print("=" * 80)
    abl = AblationComparison(args.instances, args.output)
    ablation_df = abl.run_all_instances(max_instances=args.max_instances)

    print("\n" + "=" * 80)
    print("PART 3: SENSITIVITY ANALYSIS")
    print("=" * 80)
    sens = SensitivityAnalyzer(args.instances, args.output)
    sens.run_sensitivity_analysis(max_instances=5, num_runs=2)

    multicomp_df = run_multicomp_comparison(sota_df, args.instances, args.output)

    if sota_df is not None and not sota_df.empty:
        run_statistical_tests(sota_df, args.output)

    paired_ablation_tests(args.output)
    paired_multicomp_AB(args.output)

    generate_all_figures(sota_df, ablation_df, multicomp_df, args.output)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Outputs: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
