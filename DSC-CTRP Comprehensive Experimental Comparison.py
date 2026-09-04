"""
DSC-CTRP COMPREHENSIVE EXPERIMENTAL COMPARISON 
================================================================================

Algorithms compared:
1. IALNS (Proposed) - Full version with adaptive weights and enhanced operators
2. Standard ALNS - Baseline ALNS without adaptive selection
3. FDAHS - Feasibility-Driven Anytime Hybrid Search (Vu et al., 2026)
4. GCOF - Guided Collaborative Optimization Framework (Han et al., 2026)
5. HSA - Hybrid Simulated Annealing (Beneich & Douiri, 2023)
6. ACO-2opt - Ant Colony Optimization with 2-Opt (Abdulkader et al., 2015)

Performance Metrics Categories:
- Economic Efficiency: Total Distance, Transportation Cost, Fleet Size, Fleet Utilization, Volume Utilization
- Safety Compliance: LDD Violation Rate, Stability Margin, Minimum Load Ratio, Slosh Risk Score, CG Compliance
- Computational Performance: CPU Time, SACA Calls, Cache Hit Rate, Iterations to Convergence

Author: Yves Ndikuriyo
Affiliation: Central South University
"""

import json
import os
import time
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS - FIXED FOR INTERACTIVE ENVIRONMENTS
# ============================================================================

try:
    try:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                CURRENT_DIR = os.getcwd()
            else:
                CURRENT_DIR = os.getcwd()
        except:
            CURRENT_DIR = os.getcwd()
except:
    CURRENT_DIR = os.getcwd()

POSSIBLE_PATHS = [
    os.path.join(CURRENT_DIR, "benchmark_instances"),
    r".\benchmark_instances",
    r"../benchmark_instances",
    r"benchmark_instances"
]

BASE_DIR = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        BASE_DIR = path
        break

if BASE_DIR is None:
    BASE_DIR = os.path.join(CURRENT_DIR, 'benchmark_instances')
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Created benchmark directory at: {BASE_DIR}")

OUTPUT_DIR = os.path.join(BASE_DIR, 'experimental_results', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ['part1_sota', 'part2_ablation', 'part3_sensitivity', 'part4_statistics']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

print(f"Benchmark Directory: {BASE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['pdf.fonttype'] = 42

# Color palettes
COLORS_SOTA = {
    'IALNS': '#2E86AB',
    'Standard_ALNS': '#1A7A5A',
    'FDAHS': '#E76F51',
    'GCOF': '#F4A261',
    'HSA': '#E9C46A',
    'ACO_2opt': '#2A9D8F',
}

SOTA_DISPLAY_NAMES = {
    'IALNS': 'IALNS (Proposed)',
    'Standard_ALNS': 'Standard ALNS',
    'FDAHS': 'FDAHS (2026)',
    'GCOF': 'GCOF (2026)',
    'HSA': 'HSA (2023)',
    'ACO_2opt': 'ACO-2opt (2015)',
}

COLORS_ABLATION = {
    'IALNS': '#2E86AB',
    'IALNS_NS': '#F18F01',
    'IALNS_NA': '#C73E1D',
    'IALNS_SEQ': '#73AB84',
    'IALNS_NC': '#6A4C93',
    'IALNS_IMB': '#D4A373'
}

ABLATION_DISPLAY_NAMES = {
    'IALNS': 'IALNS (Full)',
    'IALNS_NS': 'No Stability',
    'IALNS_NA': 'No Adaptive',
    'IALNS_SEQ': 'Sequential',
    'IALNS_NC': 'No Cache',
    'IALNS_IMB': 'No Imbalance'
}

# Compartment colors
COLORS_COMPARTMENTS = {
    1: '#2E86AB',
    2: '#E76F51',
    3: '#1A7A5A',
    4: '#F4A261',
}

COMPARTMENT_LABELS = {
    1: 'Single-Compartment',
    2: '2 Compartments',
    3: '3 Compartments',
    4: '4 Compartments',
}

MARKERS = {
    'IALNS': 'o',
    'Standard_ALNS': 's',
    'FDAHS': '^',
    'GCOF': 'D',
    'HSA': 'v',
    'ACO_2opt': 'p',
    'IALNS (Full)': 'o',
    'No Stability': '*',
    'No Adaptive': 'h',
    'Sequential': '<',
    'No Cache': '>',
    'No Imbalance': 'P',
    'Single-Compartment': 'x',
    'Multi-Compartment': 'o'
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: Dict[int, float]
    time_window: Tuple[float, float]
    service_time: float

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
    # Economic Efficiency Metrics
    total_distance: float
    transportation_cost: float
    fleet_size: int
    fleet_utilization: float
    volume_utilization: float
    
    # Safety Compliance Metrics
    ldd_violation_rate: float
    stability_margin: float
    minimum_load_ratio: float
    slosh_risk_score: float
    
    # Computational Performance Metrics
    cpu_time: float
    saca_calls: int
    cache_hit_rate: float
    iterations_to_convergence: int = 0  # Default value added
    
    # CG Metrics (additional safety metrics)
    cg_position_mean: float = 0.0
    cg_position_std: float = 0.0
    cg_position_min: float = 0.0
    cg_position_max: float = 0.0
    cg_compliance_rate: float = 100.0
    cg_deviation_from_optimum: float = 0.0
    heeling_moment_estimate: float = 0.0
    lateral_acceleration_margin: float = 0.0
    route_cg_positions: List[float] = field(default_factory=list)
    route_stability_margins: List[float] = field(default_factory=list)
    route_fill_ratios: List[float] = field(default_factory=list)
    parameter_combination: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """Results from a single sensitivity analysis run."""
    lambda_H: float  # Handling cost weight
    lambda_W: float  # Overload penalty weight
    lambda_S: float  # Stability penalty weight
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
# BENCHMARK INSTANCE GENERATOR
# ============================================================================

def generate_benchmark_instances(output_dir: str, num_instances: int = 281):
    print(f"\nGenerating {num_instances} synthetic benchmark instances...")
    all_instances = []
    instance_counter = 1
    
    # Paixão Layer with varying compartments
    paixao_configs = [
        ('S', 3, 10, 3, 1),   # Single compartment
        ('M', 5, 10, 3, 2),   # 2 compartments
        ('L', 7, 10, 3, 3),   # 3 compartments
        ('EL', 9, 10, 3, 4),  # 4 compartments
    ]
    
    for prefix, n_customers, count, n_products, n_compartments in paixao_configs:
        for i in range(1, count + 1):
            name = f"{prefix}_{i}"
            instance = generate_single_instance(
                n_customers=n_customers,
                layer='Paixão',
                seed=instance_counter,
                name=name,
                n_products=n_products,
                n_compartments=n_compartments
            )
            all_instances.append((name, instance))
            instance_counter += 1
    
    mirzaei_configs = [
        (10, 2, 15, 3), (10, 3, 10, 3), (10, 4, 5, 4), (10, 5, 3, 4), (10, 6, 2, 4),
        (12, 2, 12, 3), (12, 3, 8, 3), (12, 4, 4, 4), (12, 5, 3, 4),
        (15, 2, 10, 3), (15, 3, 8, 3), (15, 4, 5, 4), (15, 5, 3, 4),
        (20, 2, 10, 3), (20, 3, 8, 3), (20, 4, 5, 4), (20, 5, 3, 4), (20, 6, 2, 4),
        (25, 2, 8, 3), (25, 3, 6, 3), (25, 4, 4, 4), (25, 5, 3, 4),
        (30, 2, 8, 3), (30, 3, 6, 3), (30, 4, 4, 4), (30, 5, 2, 4),
        (40, 2, 6, 3), (40, 3, 5, 4), (40, 4, 3, 4), (40, 5, 2, 4),
        (50, 2, 6, 3), (50, 3, 5, 4), (50, 4, 3, 4), (50, 5, 2, 4),
        (60, 2, 5, 3), (60, 3, 4, 4), (60, 4, 3, 4),
        (75, 2, 5, 3), (75, 3, 4, 4), (75, 4, 2, 4),
        (100, 2, 5, 3), (100, 3, 3, 4), (100, 4, 2, 4),
    ]
    
    vrpnc_index = 1
    for n_customers, n_products, count, n_compartments in mirzaei_configs:
        for j in range(1, count + 1):
            name = f"vrpnc{vrpnc_index}"
            instance = generate_single_instance(
                n_customers=n_customers,
                layer='Mirzaei',
                seed=instance_counter,
                name=name,
                n_products=n_products,
                n_compartments=n_compartments
            )
            all_instances.append((name, instance))
            instance_counter += 1
            vrpnc_index += 1
    
    for name, instance in all_instances:
        filename = f"{name}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(instance, f, indent=2)
        print(f"  Generated: {filename}")
    
    print(f"\nGenerated {len(all_instances)} benchmark instances in: {output_dir}")
    return all_instances


def generate_single_instance(n_customers: int, layer: str, seed: int, 
                            name: str = None, n_products: int = 3,
                            n_compartments: int = 3) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    
    if name is None:
        name = f"{layer}_{n_customers}"
    
    customers = []
    for i in range(n_customers):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        demand = {}
        for p in range(n_products):
            demand[p] = random.uniform(5, 30)
        start = random.uniform(0, 100)
        end = start + random.uniform(20, 60)
        customers.append({
            'id': i,
            'x': x,
            'y': y,
            'demand': demand,
            'time_window': [start, end],
            'service_time': random.uniform(5, 15)
        })
    
    time_windows = {}
    for i in range(n_customers):
        start = random.uniform(0, 100)
        end = start + random.uniform(20, 60)
        time_windows[i] = [start, end]
    
    compartments = []
    compartment_positions = {}
    for i in range(n_compartments):
        pos_x = random.uniform(-2, 2) + i * 1.5
        compartment_positions[i] = pos_x
        compartments.append({
            'id': i,
            'capacity': random.uniform(30, 80),
            'position_x': pos_x,
            'position_y': 0.0,
            'load_density': 1.0
        })
    
    product_densities = {}
    for p in range(n_products):
        product_densities[p] = random.uniform(0.6, 1.2)
    
    wheelbase = random.uniform(3, 5)
    front_axle_max = random.uniform(150, 250)
    rear_axle_max = random.uniform(200, 350)
    
    vehicle = {
        'id': 0,
        'compartments': compartments,
        'compartment_positions': compartment_positions,
        'max_payload': random.uniform(200, 400),
        'front_axle_max': front_axle_max,
        'rear_axle_max': rear_axle_max,
        'front_axle_min': front_axle_max * 0.1,
        'rear_axle_min': rear_axle_max * 0.1,
        'wheelbase': wheelbase,
        'unladen_weight': random.uniform(50, 100),
        'max_volume': sum(c['capacity'] for c in compartments),
        'steering_min_ratio': random.uniform(25, 35),
        'driving_min_ratio': random.uniform(55, 65),
        'product_densities': product_densities,
        'cg_min': 0.3 * wheelbase,
        'cg_max': 0.7 * wheelbase
    }
    
    coords = [[0, 0]] + [[c['x'], c['y']] for c in customers]
    dist_matrix = [[math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
                   for j in range(n_customers + 1)] for i in range(n_customers + 1)]
    
    return {
        'name': name,
        'layer': layer,
        'customers': customers,
        'time_windows': time_windows,
        'vehicle': vehicle,
        'distance_matrix': dist_matrix,
        'n_products': n_products,
        'n_customers': n_customers,
        'n_compartments': n_compartments
    }


# ============================================================================
# LDD VERIFIER
# ============================================================================

class LDDVerifier:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.compartment_positions = {c.id: c.position_x for c in vehicle.compartments}
        self.front_axle_min = getattr(vehicle, 'front_axle_min', 0.0)
        self.rear_axle_min = getattr(vehicle, 'rear_axle_min', 0.0)
        self.cg_min = getattr(vehicle, 'cg_min', 0.3 * vehicle.wheelbase)
        self.cg_max = getattr(vehicle, 'cg_max', 0.7 * vehicle.wheelbase)
        
    def verify_load(self, compartment_loads: Dict[int, float]) -> LDDResult:
        violations = []
        compartment_loads = {k: v for k, v in compartment_loads.items() if v > 0}
        total_weight = sum(compartment_loads.values())
        
        if total_weight == 0:
            return LDDResult(True, 0, 0, 0, 100.0, [], 
                           cg_position=0, cg_min=self.cg_min, cg_max=self.cg_max,
                           cg_compliant=True, total_mass=0, longitudinal_moment=0)
        
        numerator = sum(load * self.compartment_positions.get(c_id, 0) 
                       for c_id, load in compartment_loads.items())
        cg_position = numerator / total_weight
        longitudinal_moment = numerator
        
        front_load = total_weight * (self.vehicle.wheelbase - cg_position) / self.vehicle.wheelbase
        rear_load = total_weight - front_load
        
        front_load += self.vehicle.unladen_weight * 0.5
        rear_load += self.vehicle.unladen_weight * 0.5
        
        total_vehicle_weight = total_weight + self.vehicle.unladen_weight
        
        if front_load > self.vehicle.front_axle_max:
            violations.append(f"Front axle overload: {front_load:.2f} > {self.vehicle.front_axle_max}")
        if front_load < self.front_axle_min:
            violations.append(f"Front axle underload: {front_load:.2f} < {self.front_axle_min}")
        if rear_load > self.vehicle.rear_axle_max:
            violations.append(f"Rear axle overload: {rear_load:.2f} > {self.vehicle.rear_axle_max}")
        if rear_load < self.rear_axle_min:
            violations.append(f"Rear axle underload: {rear_load:.2f} < {self.rear_axle_min}")
        if total_weight > self.vehicle.max_payload:
            violations.append(f"Payload exceeds maximum: {total_weight:.2f} > {self.vehicle.max_payload}")
        
        cg_compliant = True
        if cg_position < self.cg_min:
            violations.append(f"CG too far forward: {cg_position:.2f} < {self.cg_min}")
            cg_compliant = False
        if cg_position > self.cg_max:
            violations.append(f"CG too far rearward: {cg_position:.2f} > {self.cg_max}")
            cg_compliant = False
        
        if total_vehicle_weight > 0:
            steering_ratio = front_load / total_vehicle_weight
            if steering_ratio < self.vehicle.steering_min_ratio / 100:
                violations.append(f"Steering axle underload: {steering_ratio*100:.1f}% < {self.vehicle.steering_min_ratio}%")
            driving_ratio = rear_load / total_vehicle_weight
            if driving_ratio < self.vehicle.driving_min_ratio / 100:
                violations.append(f"Driving axle underload: {driving_ratio*100:.1f}% < {self.vehicle.driving_min_ratio}%")
        
        margins = []
        if self.vehicle.front_axle_max > 0 and front_load > 0:
            margins.append((self.vehicle.front_axle_max - front_load) / self.vehicle.front_axle_max)
        if self.vehicle.rear_axle_max > 0 and rear_load > 0:
            margins.append((self.vehicle.rear_axle_max - rear_load) / self.vehicle.rear_axle_max)
        if self.vehicle.max_payload > 0 and total_weight > 0:
            margins.append((self.vehicle.max_payload - total_weight) / self.vehicle.max_payload)
        if self.cg_max > self.cg_min:
            cg_margin_min = (cg_position - self.cg_min) / (self.cg_max - self.cg_min)
            cg_margin_max = (self.cg_max - cg_position) / (self.cg_max - self.cg_min)
            margins.extend([cg_margin_min, cg_margin_max])
        
        stability_margin = min(margins) * 100 if margins else 100.0
        
        return LDDResult(
            feasible=len(violations) == 0,
            front_load=front_load,
            rear_load=rear_load,
            center_of_gravity=cg_position,
            stability_margin=stability_margin,
            violations=violations,
            cg_position=cg_position,
            cg_min=self.cg_min,
            cg_max=self.cg_max,
            cg_compliant=cg_compliant,
            total_mass=total_weight,
            longitudinal_moment=longitudinal_moment
        )


# ============================================================================
# SACA (Stability-Aware Compartment Allocation)
# ============================================================================

class SACA:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle
        self.ldd_verifier = LDDVerifier(vehicle)
        self.cache = {}
        self.milp_calls = 0
        self.cache_hits = 0
        self.cg_history = []
        self.route_cg_positions = []
        self.route_stability_margins = []
        
    def solve_heuristic(self, demands: Dict[int, float], product_densities: Dict[int, float]) -> Tuple[bool, Dict]:
        cache_key = tuple(sorted(demands.items()))
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        if not demands or sum(demands.values()) == 0:
            result = (True, {c.id: 0 for c in self.vehicle.compartments})
            self.cache[cache_key] = result
            return result
        
        self.milp_calls += 1
        sorted_products = sorted(demands.items(), key=lambda x: x[1], reverse=True)
        allocation = {c.id: 0 for c in self.vehicle.compartments}
        assigned_products = {}
        
        for product_id, demand_volume in sorted_products:
            remaining = demand_volume
            
            while remaining > 0:
                best_comp_id = None
                best_score = float('inf')
                best_alloc = 0
                best_ldd_result = None
                
                for comp in self.vehicle.compartments:
                    if comp.id in assigned_products.values():
                        continue
                    available = comp.capacity - allocation[comp.id]
                    if available <= 0:
                        continue
                    alloc_amount = min(remaining, available)
                    if alloc_amount <= 0:
                        continue
                    
                    test_allocation = allocation.copy()
                    test_allocation[comp.id] += alloc_amount
                    test_load = {c_id: test_allocation[c_id] * product_densities.get(assigned_products.get(c_id, product_id), 0.8)
                                for c_id in test_allocation if test_allocation[c_id] > 0}
                    ldd_result = self.ldd_verifier.verify_load(test_load)
                    
                    if ldd_result.feasible:
                        fill_ratio = test_allocation[comp.id] / comp.capacity
                        cg_score = abs(ldd_result.cg_position - 0.5 * self.vehicle.wheelbase) / (0.5 * self.vehicle.wheelbase)
                        score = (1 - fill_ratio) + 0.5 * cg_score + (1 - ldd_result.stability_margin / 100) * 0.3
                        if score < best_score:
                            best_score = score
                            best_comp_id = comp.id
                            best_alloc = alloc_amount
                            best_ldd_result = ldd_result
                
                if best_comp_id is None:
                    result = (False, {})
                    self.cache[cache_key] = result
                    return result
                
                allocation[best_comp_id] += best_alloc
                assigned_products[best_comp_id] = product_id
                remaining -= best_alloc
        
        final_load = {c_id: allocation[c_id] * product_densities.get(assigned_products.get(c_id, 0), 0.8)
                     for c_id in allocation if allocation[c_id] > 0}
        ldd_result = self.ldd_verifier.verify_load(final_load)
        result = (ldd_result.feasible, allocation)
        self.cache[cache_key] = result
        
        if ldd_result.feasible:
            self.cg_history.append({
                'cg_position': ldd_result.cg_position,
                'stability_margin': ldd_result.stability_margin,
                'total_mass': ldd_result.total_mass,
                'front_load': ldd_result.front_load,
                'rear_load': ldd_result.rear_load
            })
            self.route_cg_positions.append(ldd_result.cg_position)
            self.route_stability_margins.append(ldd_result.stability_margin)
        
        return result


# ============================================================================
# OBJECTIVE CALCULATOR - WITH ALL PERFORMANCE METRICS
# ============================================================================

class ObjectiveCalculator:
    def __init__(self, instance_data: Dict, vehicle: Vehicle):
        self.instance_data = instance_data
        self.vehicle = vehicle
        self.ldd_verifier = LDDVerifier(vehicle)
        self.saca = SACA(vehicle)
        self.all_cg_positions = []
        self.all_stability_margins = []
        self.all_fill_ratios = []
        self.distance_matrix = instance_data['distance_matrix']
        self.total_saca_calls = 0
        self.cache_hits_total = 0
        
    def compute_total_distance(self, solution: Dict[int, List[int]]) -> float:
        """∑_{k∈K}∑_{(i,j)∈A} d_{ij}x_{ij}^k"""
        total = 0.0
        for route in solution.values():
            if not route:
                continue
            nodes = [0] + [c + 1 for c in route] + [0]
            total += sum(self.distance_matrix[nodes[i]][nodes[i+1]] 
                        for i in range(len(nodes) - 1))
        return total
    
    def compute_transportation_cost(self, solution: Dict[int, List[int]]) -> float:
        """∑_{k∈K} (c^{km} · D_k + c^{fixed} · y_k)"""
        total_distance = self.compute_total_distance(solution)
        fleet_size = len(solution)
        return self.vehicle.cost_per_km * total_distance + self.vehicle.fixed_cost * fleet_size
    
    def compute_fleet_size(self, solution: Dict[int, List[int]]) -> int:
        """∑_{k∈K} y_k - Number of vehicles used"""
        return len(solution)
    
    def compute_fleet_utilization(self, solution: Dict[int, List[int]]) -> float:
        """Route consolidation efficiency - percentage of vehicles with >80% utilization"""
        utilized = 0
        total_vehicles = len(solution)
        if total_vehicles == 0:
            return 0.0
        
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            route_volume = sum(demands.values())
            utilization = route_volume / self.vehicle.max_volume if self.vehicle.max_volume > 0 else 0
            if utilization >= 0.8:  # 80% threshold for "efficient" utilization
                utilized += 1
        
        return (utilized / total_vehicles) * 100
    
    def compute_volume_utilization(self, solution: Dict[int, List[int]]) -> float:
        """(1/|K|) ∑_{k∈K} (∑_{i,p,c} q_{ipck} / Cap^{truck}_k) × 100"""
        total_volume = 0.0
        total_capacity = 0.0
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            total_volume += sum(demands.values())
            total_capacity += self.vehicle.max_volume
        return (total_volume / total_capacity * 100) if total_capacity > 0 else 0.0
    
    def compute_ldd_violation_rate(self, solution: Dict[int, List[int]]) -> float:
        """(1/|K|) ∑_{k∈K} I(⋃_{j∈J} {L_{rjk} ∉ [Lmin_{rk}, Lmax_{rk}]}) × 100"""
        violations = 0
        total_routes = 0
        for route in solution.values():
            if not route:
                continue
            total_routes += 1
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            if allocation:
                load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                       for c_id in allocation if allocation[c_id] > 0}
                ldd_result = self.ldd_verifier.verify_load(load)
                if not ldd_result.feasible:
                    violations += 1
        return (violations / total_routes * 100) if total_routes > 0 else 0.0
    
    def compute_stability_margin(self, solution: Dict[int, List[int]]) -> float:
        """min_{k,j} [min{L_{rjk} - Lmin_{rk}, Lmax_{rk} - L_{rjk}} / max{Lmax_{rk} - Lmin_{rk}, 1}] × 100"""
        min_margin = float('inf')
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            if feasible and allocation:
                load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                       for c_id in allocation if allocation[c_id] > 0}
                ldd_result = self.ldd_verifier.verify_load(load)
                if ldd_result.feasible:
                    min_margin = min(min_margin, ldd_result.stability_margin)
                    self.all_stability_margins.append(ldd_result.stability_margin)
        return min_margin if min_margin != float('inf') else 100.0
    
    def compute_minimum_load_ratio(self, solution: Dict[int, List[int]]) -> float:
        """min_{k,j} (L_front,jk / (W_jk + U)) × 100 - Steering axle compliance"""
        min_ratio = float('inf')
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            if feasible and allocation:
                load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                       for c_id in allocation if allocation[c_id] > 0}
                ldd_result = self.ldd_verifier.verify_load(load)
                if ldd_result.feasible and ldd_result.total_mass > 0:
                    total_vehicle_weight = ldd_result.total_mass + self.vehicle.unladen_weight
                    ratio = ldd_result.front_load / total_vehicle_weight if total_vehicle_weight > 0 else 0
                    min_ratio = min(min_ratio, ratio * 100)
        return min_ratio if min_ratio != float('inf') else 100.0
    
    def compute_slosh_risk_score(self, solution: Dict[int, List[int]]) -> float:
        """(1/|K|) ∑_{k∈K} ∑_{c∈C_k} max(0, (ρ V_c ω_c^2 A_c) / F_critical)"""
        risk_score = 0.0
        total_compartments = 0
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            if feasible and allocation:
                for c_id, volume in allocation.items():
                    comp = next(c for c in self.vehicle.compartments if c.id == c_id)
                    fill_ratio = volume / comp.capacity if comp.capacity > 0 else 0
                    # Simplified slosh risk estimation
                    # ρ = density, V = volume, ω = fill ratio, A = slosh amplitude factor
                    if 0.35 <= fill_ratio <= 0.55:
                        risk_score += 1.0  # High slosh risk (partial fill)
                    elif 0.25 <= fill_ratio <= 0.65:
                        risk_score += 0.5  # Medium slosh risk
                    total_compartments += 1
                    self.all_fill_ratios.append(fill_ratio)
        return risk_score / max(total_compartments, 1)
    
    def compute_cg_metrics(self, solution: Dict[int, List[int]]) -> Dict:
        """Compute Center of Gravity metrics for all routes."""
        cg_positions = []
        route_cg_data = []
        
        for vid, route in solution.items():
            if not route:
                continue
            
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            
            if feasible and allocation:
                load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                       for c_id in allocation if allocation[c_id] > 0}
                ldd_result = self.ldd_verifier.verify_load(load)
                
                if ldd_result.feasible:
                    cg_positions.append(ldd_result.cg_position)
                    route_cg_data.append({
                        'route_id': vid,
                        'cg_position': ldd_result.cg_position,
                        'stability_margin': ldd_result.stability_margin,
                        'front_load': ldd_result.front_load,
                        'rear_load': ldd_result.rear_load,
                        'total_mass': ldd_result.total_mass,
                        'n_customers': len(route)
                    })
        
        if not cg_positions:
            return {
                'cg_position_mean': 0,
                'cg_position_std': 0,
                'cg_position_min': 0,
                'cg_position_max': 0,
                'cg_compliance_rate': 100.0,
                'cg_deviation_from_optimum': 0,
                'route_cg_data': []
            }
        
        cg_mean = np.mean(cg_positions)
        cg_std = np.std(cg_positions) if len(cg_positions) > 1 else 0
        cg_min = min(cg_positions)
        cg_max = max(cg_positions)
        
        optimum_cg = 0.45 * self.vehicle.wheelbase
        cg_deviation = abs(cg_mean - optimum_cg)
        
        cg_min_bound = 0.3 * self.vehicle.wheelbase
        cg_max_bound = 0.7 * self.vehicle.wheelbase
        compliant_count = sum(1 for cg in cg_positions if cg_min_bound <= cg <= cg_max_bound)
        compliance_rate = (compliant_count / len(cg_positions)) * 100 if cg_positions else 100.0
        
        return {
            'cg_position_mean': cg_mean,
            'cg_position_std': cg_std,
            'cg_position_min': cg_min,
            'cg_position_max': cg_max,
            'cg_compliance_rate': compliance_rate,
            'cg_deviation_from_optimum': cg_deviation,
            'route_cg_data': route_cg_data
        }
    
    def compute_heeling_moment_estimate(self, solution: Dict[int, List[int]], 
                                       lateral_acceleration: float = 0.4) -> float:
        """Estimate heeling moment due to lateral acceleration."""
        max_moment = 0.0
        for route in solution.values():
            if not route:
                continue
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            if feasible and allocation:
                load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                       for c_id in allocation if allocation[c_id] > 0}
                total_mass = sum(load.values())
                cg_height = 0.5 + np.mean([self.ldd_verifier.compartment_positions.get(c_id, 0) 
                                          for c_id in load.keys()])
                heeling_moment = total_mass * lateral_acceleration * cg_height
                max_moment = max(max_moment, heeling_moment)
        return max_moment
    
    def compute_lateral_acceleration_margin(self, solution: Dict[int, List[int]], 
                                           max_heeling_moment: float = 1000) -> float:
        """Compute lateral acceleration margin."""
        heeling_moment = self.compute_heeling_moment_estimate(solution)
        if heeling_moment > 0:
            return (max_heeling_moment - heeling_moment) / max_heeling_moment * 100
        return 100.0
    
    def _extract_demands(self, route: List[int]) -> Dict[int, float]:
        demands = defaultdict(float)
        for customer_id in route:
            for product_id, volume in self.instance_data['customers'][customer_id]['demand'].items():
                demands[product_id] += volume
        return dict(demands)
    
    def compute_all_metrics(self, solution: Dict[int, List[int]], 
                            iterations_to_convergence: int = 0) -> PerformanceMetrics:
        """Compute all performance metrics as defined in Table 1."""
        start_time = time.time()
        
        # Economic Efficiency Metrics
        total_distance = self.compute_total_distance(solution)
        transportation_cost = self.compute_transportation_cost(solution)
        fleet_size = self.compute_fleet_size(solution)
        fleet_utilization = self.compute_fleet_utilization(solution)
        volume_utilization = self.compute_volume_utilization(solution)
        
        # Safety Compliance Metrics
        ldd_violation_rate = self.compute_ldd_violation_rate(solution)
        stability_margin = self.compute_stability_margin(solution)
        minimum_load_ratio = self.compute_minimum_load_ratio(solution)
        slosh_risk = self.compute_slosh_risk_score(solution)
        
        # CG Metrics
        cg_metrics = self.compute_cg_metrics(solution)
        heeling_moment = self.compute_heeling_moment_estimate(solution)
        lateral_margin = self.compute_lateral_acceleration_margin(solution)
        
        # Computational Performance
        cpu_time = time.time() - start_time
        saca_calls = self.saca.milp_calls
        cache_hit_rate = (self.saca.cache_hits / max(1, self.saca.milp_calls + self.saca.cache_hits)) * 100
        
        return PerformanceMetrics(
            # Economic Efficiency
            total_distance=total_distance,
            transportation_cost=transportation_cost,
            fleet_size=fleet_size,
            fleet_utilization=fleet_utilization,
            volume_utilization=volume_utilization,
            # Safety Compliance
            ldd_violation_rate=ldd_violation_rate,
            stability_margin=stability_margin,
            minimum_load_ratio=minimum_load_ratio,
            slosh_risk_score=slosh_risk,
            # Computational Performance
            cpu_time=cpu_time,
            saca_calls=saca_calls,
            cache_hit_rate=cache_hit_rate,
            iterations_to_convergence=iterations_to_convergence,
            # CG Metrics
            cg_position_mean=cg_metrics['cg_position_mean'],
            cg_position_std=cg_metrics['cg_position_std'],
            cg_position_min=cg_metrics['cg_position_min'],
            cg_position_max=cg_metrics['cg_position_max'],
            cg_compliance_rate=cg_metrics['cg_compliance_rate'],
            cg_deviation_from_optimum=cg_metrics['cg_deviation_from_optimum'],
            heeling_moment_estimate=heeling_moment,
            lateral_acceleration_margin=lateral_margin,
            route_cg_positions=[d['cg_position'] for d in cg_metrics['route_cg_data']],
            route_stability_margins=[d['stability_margin'] for d in cg_metrics['route_cg_data']],
            route_fill_ratios=self.all_fill_ratios.copy()
        )


# ============================================================================
# SINGLE-COMPARTMENT VEHICLE CREATOR
# ============================================================================

def create_single_compartment_vehicle(instance: Dict) -> Vehicle:
    """Create a single-compartment vehicle for comparison."""
    vehicle_data = instance['vehicle']
    
    total_capacity = sum(c['capacity'] for c in vehicle_data['compartments'])
    avg_position = np.mean([c['position_x'] for c in vehicle_data['compartments']])
    
    compartments = [Compartment(
        id=0,
        capacity=total_capacity,
        position_x=avg_position,
        position_y=0.0,
        load_density=1.0
    )]
    
    product_densities = vehicle_data.get('product_densities', {})
    
    return Vehicle(
        id=0,
        compartments=compartments,
        max_payload=vehicle_data['max_payload'],
        front_axle_max=vehicle_data['front_axle_max'],
        rear_axle_max=vehicle_data['rear_axle_max'],
        wheelbase=vehicle_data['wheelbase'],
        unladen_weight=vehicle_data['unladen_weight'],
        max_volume=total_capacity,
        steering_min_ratio=vehicle_data['steering_min_ratio'],
        driving_min_ratio=vehicle_data['driving_min_ratio'],
        product_densities=product_densities,
        fixed_cost=100.0,
        cost_per_km=1.0,
        front_axle_min=vehicle_data.get('front_axle_min', vehicle_data['front_axle_max'] * 0.1),
        rear_axle_min=vehicle_data.get('rear_axle_min', vehicle_data['rear_axle_max'] * 0.1),
        cg_min=vehicle_data.get('cg_min', 0.3 * vehicle_data['wheelbase']),
        cg_max=vehicle_data.get('cg_max', 0.7 * vehicle_data['wheelbase']),
        compartment_positions={0: avg_position},
        is_multi_compartment=False
    )


# ============================================================================
# BASE ALGORITHM CLASS
# ============================================================================

class BaseAlgorithm:
    """Base class with common functionality for all algorithms."""
    
    def __init__(self, instance_data: Dict, vehicle: Vehicle):
        self.instance_data = instance_data
        self.vehicle = vehicle
        self.saca = SACA(vehicle)
        self.calc = ObjectiveCalculator(instance_data, vehicle)
        self.distance_matrix = instance_data['distance_matrix']
        self.feasibility_cache = {}
        self.milp_calls = 0
        self.cache_hits = 0
        self.n_customers = len(instance_data['customers'])
        self.convergence_history = []
        
    def _is_route_feasible(self, route: List[int]) -> bool:
        if not route:
            return True
        route_key = tuple(sorted(route))
        if route_key in self.feasibility_cache:
            self.cache_hits += 1
            return self.feasibility_cache[route_key]
        
        demands = self._extract_demands(route)
        total_volume = sum(demands.values())
        if total_volume > self.vehicle.max_volume:
            self.feasibility_cache[route_key] = False
            return False
        
        self.milp_calls += 1
        feasible, _ = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
        self.feasibility_cache[route_key] = feasible
        return feasible
    
    def _extract_demands(self, route: List[int]) -> Dict[int, float]:
        demands = defaultdict(float)
        for customer_id in route:
            for product_id, volume in self.instance_data['customers'][customer_id]['demand'].items():
                demands[product_id] += volume
        return dict(demands)
    
    def _generate_initial_solution(self) -> Dict[int, List[int]]:
        unassigned = list(range(self.n_customers))
        random.shuffle(unassigned)
        solution = {}
        
        for customer in unassigned:
            inserted = False
            best_vid = None
            best_pos = None
            best_cost = float('inf')
            
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible(test_route):
                        cost = self.calc.compute_transportation_cost({vid: test_route})
                        if cost < best_cost:
                            best_cost = cost
                            best_vid = vid
                            best_pos = pos
                            inserted = True
            
            if inserted:
                solution[best_vid].insert(best_pos, customer)
            else:
                solution[len(solution)] = [customer]
        
        return solution
    
    def _two_opt_improvement(self, solution: Dict[int, List[int]]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        for vid in list(new_solution.keys()):
            route = new_solution[vid]
            if len(route) < 3:
                continue
            
            improved = True
            while improved:
                improved = False
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if self._is_route_feasible(new_route):
                            new_cost = self.calc.compute_transportation_cost({vid: new_route})
                            old_cost = self.calc.compute_transportation_cost({vid: route})
                            if new_cost < old_cost:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
            
            new_solution[vid] = route
        
        return new_solution


# ============================================================================
# ALGORITHM 1: STANDARD ALNS
# ============================================================================

class StandardALNS(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'initial_temperature': 100.0,
            'cooling_rate': 0.99,
            'destruction_rate': 0.3,
            'max_no_improvement': 50
        }
        self.destroy_operators = ['random', 'worst', 'shaw']
        self.repair_operators = ['greedy', 'regret']
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        solution = self._generate_initial_solution()
        current_cost = self.calc.compute_transportation_cost(solution)
        best_solution = solution.copy()
        best_cost = current_cost
        best_metrics = self.calc.compute_all_metrics(solution, iterations_to_convergence=0)
        
        temperature = self.params['initial_temperature']
        iterations = 0
        no_improvement = 0
        
        while iterations < self.params['max_iterations']:
            iterations += 1
            
            destroy_op = random.choice(self.destroy_operators)
            repair_op = random.choice(self.repair_operators)
            
            destroyed, removed = self._apply_destroy(solution, destroy_op)
            repaired = self._apply_repair(destroyed, removed, repair_op)
            
            new_cost = self.calc.compute_transportation_cost(repaired)
            
            if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temperature):
                solution = repaired
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = new_cost
                    best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            
            temperature *= self.params['cooling_rate']
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iterations
        
        return best_solution, best_cost, {'metrics': best_metrics, 'convergence': self.convergence_history}
    
    def _apply_destroy(self, solution: Dict[int, List[int]], operator: str) -> Tuple[Dict[int, List[int]], List[int]]:
        if operator == 'random':
            return self._destroy_random(solution)
        elif operator == 'worst':
            return self._destroy_worst(solution)
        else:
            return self._destroy_shaw(solution)
    
    def _destroy_random(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        all_customers = [c for route in new_solution.values() for c in route]
        n_remove = max(2, int(len(all_customers) * self.params['destruction_rate']))
        
        if len(all_customers) > n_remove:
            to_remove = set(random.sample(all_customers, n_remove))
            removed = list(to_remove)
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if c not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_worst(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        marginal_costs = {}
        for vid, route in new_solution.items():
            if len(route) <= 1:
                continue
            for idx, customer in enumerate(route):
                route_without = route[:idx] + route[idx+1:]
                if route_without:
                    cost_with = self.calc.compute_transportation_cost({vid: route})
                    cost_without = self.calc.compute_transportation_cost({vid: route_without})
                    marginal_costs[(vid, customer)] = cost_with - cost_without
        
        if marginal_costs:
            n_remove = max(2, int(len(marginal_costs) * self.params['destruction_rate'] * 0.8))
            sorted_customers = sorted(marginal_costs.items(), key=lambda x: x[1], reverse=True)
            to_remove = {item[0] for item in sorted_customers[:n_remove]}
            removed = [c for _, c in to_remove]
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if (vid, c) not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_shaw(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        all_customers = [c for route in new_solution.values() for c in route]
        
        if len(all_customers) < 3:
            return self._destroy_random(solution)
        
        seed = random.choice(all_customers)
        seed_demand = set(self.instance_data['customers'][seed]['demand'].keys())
        
        similar = []
        for customer in all_customers:
            if customer == seed:
                continue
            customer_demand = set(self.instance_data['customers'][customer]['demand'].keys())
            overlap = len(seed_demand.intersection(customer_demand))
            if overlap > 0:
                similar.append((customer, overlap))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        n_remove = max(2, min(len(similar), int(len(all_customers) * self.params['destruction_rate'] * 0.7)))
        
        if similar:
            to_remove = {seed} | {c for c, _ in similar[:n_remove]}
            removed = list(to_remove)
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if c not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _apply_repair(self, solution: Dict[int, List[int]], unassigned: List[int], operator: str) -> Dict[int, List[int]]:
        if operator == 'greedy':
            return self._repair_greedy(solution, unassigned)
        else:
            return self._repair_regret(solution, unassigned)
    
    def _repair_greedy(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        random.shuffle(remaining)
        
        for customer in remaining[:]:
            best_vid = None
            best_pos = None
            best_cost = float('inf')
            
            for vid, route in new_solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible(new_route):
                        cost = self.calc.compute_transportation_cost({vid: new_route})
                        if cost < best_cost:
                            best_cost = cost
                            best_vid = vid
                            best_pos = pos
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
        
        for customer in remaining:
            if self._is_route_feasible([customer]):
                new_solution[len(new_solution)] = [customer]
            else:
                new_solution[len(new_solution)] = [customer]
        
        return new_solution
    
    def _repair_regret(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        
        while remaining:
            regrets = []
            for customer in remaining:
                costs = []
                for vid, route in new_solution.items():
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [customer] + route[pos:]
                        if self._is_route_feasible(new_route):
                            costs.append(self.calc.compute_transportation_cost({vid: new_route}))
                
                if len(costs) >= 2:
                    costs.sort()
                    regret = costs[1] - costs[0]
                elif costs:
                    regret = costs[0]
                else:
                    regret = float('inf')
                regrets.append((regret, customer))
            
            if not regrets:
                break
            
            regrets.sort(key=lambda x: x[0], reverse=True)
            if regrets[0][0] == float('inf'):
                customer = remaining.pop(0)
                new_solution[len(new_solution)] = [customer]
                continue
            
            customer = regrets[0][1]
            best_vid = None
            best_pos = None
            best_cost = float('inf')
            
            for vid, route in new_solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible(new_route):
                        cost = self.calc.compute_transportation_cost({vid: new_route})
                        if cost < best_cost:
                            best_cost = cost
                            best_vid = vid
                            best_pos = pos
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
            else:
                new_solution[len(new_solution)] = [customer]
                remaining.remove(customer)
        
        return new_solution


# ============================================================================
# ALGORITHM 2: FDAHS (FIXED)
# ============================================================================

class FDAHSAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'population_size': 20,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'com_penalty_weight': 0.5,
            'max_no_improvement': 50
        }
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        population = []
        for _ in range(self.params['population_size']):
            solution = self._generate_initial_solution()
            solution = self._two_opt_improvement(solution)
            population.append(solution)
        
        costs = [self.calc.compute_transportation_cost(sol) for sol in population]
        best_idx = np.argmin(costs)
        best_solution = population[best_idx].copy()
        best_cost = costs[best_idx]
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=0)
        
        no_improvement = 0
        iterations = 0
        
        while iterations < self.params['max_iterations']:
            iterations += 1
            
            new_population = []
            
            for _ in range(self.params['population_size']):
                if random.random() < self.params['crossover_rate']:
                    p1 = self._tournament_select(population, costs)
                    p2 = self._tournament_select(population, costs)
                    child = self._crossover(p1, p2)
                else:
                    child = random.choice(population).copy()
                
                if random.random() < self.params['mutation_rate']:
                    child = self._mutate(child)
                
                child = self._two_opt_improvement(child)
                child = self._com_aware_repair(child)
                new_population.append(child)
            
            population = new_population
            costs = [self.calc.compute_transportation_cost(sol) for sol in population]
            
            current_best_idx = np.argmin(costs)
            if costs[current_best_idx] < best_cost:
                best_solution = population[current_best_idx].copy()
                best_cost = costs[current_best_idx]
                best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
                no_improvement = 0
            else:
                no_improvement += 1
            
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iterations
        
        return best_solution, best_cost, {'metrics': best_metrics, 'convergence': self.convergence_history}
    
    def _tournament_select(self, population: List, costs: List, tournament_size: int = 3) -> Dict:
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(indices, key=lambda i: costs[i])
        return population[best_idx].copy()
    
    def _crossover(self, p1: Dict, p2: Dict) -> Dict:
        p1_customers = [c for route in p1.values() for c in route]
        p2_customers = [c for route in p2.values() for c in route]
        
        if len(p1_customers) < 2:
            return p1.copy()
        
        idx1 = random.randint(0, len(p1_customers) - 2)
        idx2 = random.randint(idx1 + 1, len(p1_customers) - 1)
        
        child_customers = [None] * len(p1_customers)
        child_customers[idx1:idx2] = p1_customers[idx1:idx2]
        
        p2_idx = 0
        for i in range(len(child_customers)):
            if child_customers[i] is None:
                while p2_customers[p2_idx] in child_customers:
                    p2_idx += 1
                child_customers[i] = p2_customers[p2_idx]
                p2_idx += 1
        
        child = {}
        for customer in child_customers:
            inserted = False
            for vid, route in child.items():
                test_route = route + [customer]
                if self._is_route_feasible(test_route):
                    child[vid].append(customer)
                    inserted = True
                    break
            if not inserted:
                child[len(child)] = [customer]
        
        return child
    
    def _mutate(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        routes = list(new_solution.keys())
        
        if len(routes) < 2:
            return new_solution
        
        vid1, vid2 = random.sample(routes, 2)
        route1 = new_solution[vid1]
        route2 = new_solution[vid2]
        
        if not route1 or not route2:
            return new_solution
        
        idx1 = random.randint(0, len(route1) - 1)
        idx2 = random.randint(0, len(route2) - 1)
        
        c1 = route1[idx1]
        c2 = route2[idx2]
        
        new_route1 = route1[:idx1] + [c2] + route1[idx1+1:]
        new_route2 = route2[:idx2] + [c1] + route2[idx2+1:]
        
        if self._is_route_feasible(new_route1) and self._is_route_feasible(new_route2):
            new_solution[vid1] = new_route1
            new_solution[vid2] = new_route2
        
        return new_solution
    
    def _com_aware_repair(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        for vid in list(new_solution.keys()):
            route = new_solution[vid]
            if not route:
                continue
            
            demands = self._extract_demands(route)
            feasible, _ = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            
            if not feasible and len(route) > 1:
                best_split = None
                best_cost = float('inf')
                
                for i in range(1, len(route)):
                    route1 = route[:i]
                    route2 = route[i:]
                    demands1 = self._extract_demands(route1)
                    demands2 = self._extract_demands(route2)
                    feasible1, _ = self.saca.solve_heuristic(demands1, self.vehicle.product_densities)
                    feasible2, _ = self.saca.solve_heuristic(demands2, self.vehicle.product_densities)
                    
                    if feasible1 and feasible2:
                        cost = self.calc.compute_transportation_cost({0: route1, 1: route2})
                        if cost < best_cost:
                            best_cost = cost
                            best_split = [route1, route2]
                
                if best_split:
                    del new_solution[vid]
                    new_solution[len(new_solution)] = best_split[0]
                    new_solution[len(new_solution)] = best_split[1]
        
        return new_solution


# ============================================================================
# ALGORITHM 3: GCOF 
# ============================================================================

class GCOFAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'population_size': 20,
            'white_shark_alpha': 0.5,
            'white_shark_beta': 0.8,
            'cgpa_threshold': 0.1,
            'max_no_improvement': 50
        }
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        population = []
        for _ in range(self.params['population_size']):
            solution = self._generate_initial_solution()
            solution = self._two_opt_improvement(solution)
            population.append(solution)
        
        costs = [self.calc.compute_transportation_cost(sol) for sol in population]
        best_idx = np.argmin(costs)
        best_solution = population[best_idx].copy()
        best_cost = costs[best_idx]
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=0)
        
        no_improvement = 0
        
        for iteration in range(self.params['max_iterations']):
            new_population = []
            
            for i, solution in enumerate(population):
                guided_solution = self._white_shark_update(solution, population, costs, i)
                guided_solution = self._cgpa_adjustment(guided_solution)
                guided_solution = self._hldcps_packing(guided_solution)
                guided_solution = self._two_opt_improvement(guided_solution)
                new_population.append(guided_solution)
            
            population = new_population
            costs = [self.calc.compute_transportation_cost(sol) for sol in population]
            
            current_best_idx = np.argmin(costs)
            if costs[current_best_idx] < best_cost:
                best_solution = population[current_best_idx].copy()
                best_cost = costs[current_best_idx]
                best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iteration + 1)
                no_improvement = 0
            else:
                no_improvement += 1
            
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iteration + 1)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iteration + 1
        
        return best_solution, best_cost, {'metrics': best_metrics, 'convergence': self.convergence_history}
    
    def _white_shark_update(self, solution: Dict, population: List, costs: List, idx: int) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        others = [j for j in range(len(population)) if j != idx]
        if len(others) >= 2:
            a, b = random.sample(others, 2)
            if costs[a] < costs[b]:
                guide_sol = population[a]
            else:
                guide_sol = population[b]
            
            if random.random() < self.params['white_shark_alpha']:
                routes_a = list(new_solution.values())
                routes_b = list(guide_sol.values())
                
                if routes_a and routes_b:
                    idx_a = random.randint(0, len(routes_a) - 1)
                    idx_b = random.randint(0, len(routes_b) - 1)
                    
                    route_a = routes_a[idx_a]
                    route_b = routes_b[idx_b]
                    
                    for c1 in route_a[:]:
                        for c2 in route_b[:]:
                            if random.random() < 0.1:
                                test_solution = self._swap_customers(new_solution, c1, c2)
                                new_solution = test_solution
        
        return new_solution
    
    def _swap_customers(self, solution: Dict, c1: int, c2: int) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        vid1 = None
        vid2 = None
        for vid, route in new_solution.items():
            if c1 in route:
                vid1 = vid
            if c2 in route:
                vid2 = vid
        
        if vid1 is None or vid2 is None or vid1 == vid2:
            return solution
        
        route1 = new_solution[vid1]
        route2 = new_solution[vid2]
        idx1 = route1.index(c1)
        idx2 = route2.index(c2)
        
        new_route1 = route1[:idx1] + [c2] + route1[idx1+1:]
        new_route2 = route2[:idx2] + [c1] + route2[idx2+1:]
        
        if self._is_route_feasible(new_route1) and self._is_route_feasible(new_route2):
            new_solution[vid1] = new_route1
            new_solution[vid2] = new_route2
        
        return new_solution
    
    def _cgpa_adjustment(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        for vid, route in new_solution.items():
            if len(route) <= 1:
                continue
            
            stability = self.calc.compute_stability_margin({vid: route})
            
            if stability < self.params['cgpa_threshold'] * 100:
                best_route = route
                best_stability = stability
                
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        test_route = route[:i] + route[j:j+1] + route[i:j] + route[j+1:]
                        if self._is_route_feasible(test_route):
                            test_stability = self.calc.compute_stability_margin({vid: test_route})
                            if test_stability > best_stability:
                                best_stability = test_stability
                                best_route = test_route
                
                new_solution[vid] = best_route
        
        return new_solution
    
    def _hldcps_packing(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        
        for vid in list(new_solution.keys()):
            route = new_solution[vid]
            if len(route) <= 1:
                continue
            
            demands = self._extract_demands(route)
            feasible, _ = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            
            if not feasible and len(route) > 1:
                best_split = None
                best_cost = float('inf')
                
                for i in range(1, len(route)):
                    route1 = route[:i]
                    route2 = route[i:]
                    demands1 = self._extract_demands(route1)
                    demands2 = self._extract_demands(route2)
                    feasible1, _ = self.saca.solve_heuristic(demands1, self.vehicle.product_densities)
                    feasible2, _ = self.saca.solve_heuristic(demands2, self.vehicle.product_densities)
                    
                    if feasible1 and feasible2:
                        cost = self.calc.compute_transportation_cost({0: route1, 1: route2})
                        if cost < best_cost:
                            best_cost = cost
                            best_split = [route1, route2]
                
                if best_split:
                    del new_solution[vid]
                    new_solution[len(new_solution)] = best_split[0]
                    new_solution[len(new_solution)] = best_split[1]
        
        return new_solution


# ============================================================================
# ALGORITHM 4: HSA 
# ============================================================================

class HSAAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'initial_temperature': 40.0,
            'cooling_rate': 0.98,
            'final_temperature': 0.001,
            'local_search_frequency': 5,
            'max_no_improvement': 50
        }
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        solution = self._generate_initial_solution()
        solution = self._two_opt_improvement(solution)
        current_cost = self.calc.compute_transportation_cost(solution)
        best_solution = solution.copy()
        best_cost = current_cost
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=0)
        
        temperature = self.params['initial_temperature']
        iterations = 0
        no_improvement = 0
        
        while temperature > self.params['final_temperature'] and iterations < self.params['max_iterations']:
            iterations += 1
            
            neighbor = self._generate_neighbor(solution)
            neighbor_cost = self.calc.compute_transportation_cost(neighbor)
            
            if no_improvement > 20:
                temp_factor = 0.95
            else:
                temp_factor = self.params['cooling_rate']
            
            if neighbor_cost < current_cost or random.random() < math.exp(-(neighbor_cost - current_cost) / temperature):
                solution = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = current_cost
                    best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            
            if iterations % self.params['local_search_frequency'] == 0:
                solution, current_cost = self._local_search(solution, current_cost)
                if current_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = current_cost
                    best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
            
            temperature *= temp_factor
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iterations
        
        return best_solution, best_cost, {'metrics': best_metrics, 'convergence': self.convergence_history}
    
    def _generate_neighbor(self, solution: Dict[int, List[int]]) -> Dict[int, List[int]]:
        operators = ['swap', 'relocate', '2opt']
        op = random.choice(operators)
        
        if op == 'swap':
            return self._swap_neighbor(solution)
        elif op == 'relocate':
            return self._relocate_neighbor(solution)
        else:
            return self._two_opt_neighbor(solution)
    
    def _swap_neighbor(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        routes = list(new_solution.keys())
        
        if len(routes) < 2:
            return new_solution
        
        vid1, vid2 = random.sample(routes, 2)
        route1 = new_solution[vid1]
        route2 = new_solution[vid2]
        
        if not route1 or not route2:
            return new_solution
        
        idx1 = random.randint(0, len(route1) - 1)
        idx2 = random.randint(0, len(route2) - 1)
        
        c1 = route1[idx1]
        c2 = route2[idx2]
        
        new_route1 = route1[:idx1] + [c2] + route1[idx1+1:]
        new_route2 = route2[:idx2] + [c1] + route2[idx2+1:]
        
        if self._is_route_feasible(new_route1) and self._is_route_feasible(new_route2):
            new_solution[vid1] = new_route1
            new_solution[vid2] = new_route2
        
        return new_solution
    
    def _relocate_neighbor(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        routes = list(new_solution.keys())
        
        if len(routes) < 2:
            return new_solution
        
        vid1, vid2 = random.sample(routes, 2)
        route1 = new_solution[vid1]
        route2 = new_solution[vid2]
        
        if not route1:
            return new_solution
        
        idx1 = random.randint(0, len(route1) - 1)
        c1 = route1[idx1]
        
        new_route1 = route1[:idx1] + route1[idx1+1:]
        if not new_route1:
            del new_solution[vid1]
        else:
            new_solution[vid1] = new_route1
        
        pos2 = random.randint(0, len(route2))
        new_route2 = route2[:pos2] + [c1] + route2[pos2:]
        
        if self._is_route_feasible(new_route2):
            new_solution[vid2] = new_route2
        else:
            new_solution[vid1] = route1
            new_solution[vid2] = route2
        
        return new_solution
    
    def _two_opt_neighbor(self, solution: Dict) -> Dict:
        new_solution = {k: v.copy() for k, v in solution.items()}
        routes = list(new_solution.keys())
        
        if not routes:
            return new_solution
        
        vid = random.choice(routes)
        route = new_solution[vid]
        
        if len(route) < 3:
            return new_solution
        
        i = random.randint(0, len(route) - 2)
        j = random.randint(i + 1, len(route) - 1)
        
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        
        if self._is_route_feasible(new_route):
            new_solution[vid] = new_route
        
        return new_solution
    
    def _local_search(self, solution: Dict, cost: float) -> Tuple[Dict, float]:
        improved = True
        current_solution = solution.copy()
        current_cost = cost
        
        while improved:
            improved = False
            
            for vid in list(current_solution.keys()):
                route = current_solution[vid]
                if len(route) < 3:
                    continue
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if self._is_route_feasible(new_route):
                            new_cost = self.calc.compute_transportation_cost({vid: new_route})
                            if new_cost < current_cost:
                                current_solution[vid] = new_route
                                current_cost = new_cost
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        
        return current_solution, current_cost


# ============================================================================
# ALGORITHM 5: ACO-2opt 
# ============================================================================

class ACO2optAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'num_ants': 20,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.9,
            'q0': 0.9,
            'max_no_improvement': 50
        }
        self.pheromone = None
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        n_customers = self.n_customers
        self.pheromone = np.ones((n_customers + 1, n_customers + 1))
        
        best_solution = None
        best_cost = float('inf')
        best_metrics = None
        no_improvement = 0
        
        for iteration in range(self.params['max_iterations']):
            solutions = []
            costs = []
            
            for ant in range(self.params['num_ants']):
                solution = self._ant_construction()
                solution = self._two_opt_improvement(solution)
                cost = self.calc.compute_transportation_cost(solution)
                solutions.append(solution)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution.copy()
                    best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iteration + 1)
                    no_improvement = 0
                else:
                    no_improvement += 1
            
            self._update_pheromone(solutions, costs)
            self.pheromone *= (1 - self.params['rho'])
            
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iteration + 1)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iteration + 1
        
        return best_solution, best_cost, {'metrics': best_metrics, 'convergence': self.convergence_history}
    
    def _ant_construction(self) -> Dict[int, List[int]]:
        unassigned = list(range(self.n_customers))
        random.shuffle(unassigned)
        solution = {}
        
        for customer in unassigned:
            best_vid = None
            best_pos = None
            best_score = -float('inf')
            
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    test_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible(test_route):
                        if pos == 0:
                            prev = 0
                            next_node = route[0] + 1 if route else 0
                        elif pos == len(route):
                            prev = route[-1] + 1
                            next_node = 0
                        else:
                            prev = route[pos-1] + 1
                            next_node = route[pos] + 1
                        
                        dist_prev = self.distance_matrix[prev][customer + 1] if prev != 0 else 0
                        dist_next = self.distance_matrix[customer + 1][next_node] if next_node != 0 else 0
                        heuristic = 1.0 / (dist_prev + dist_next + 1)
                        
                        pheromone = self.pheromone[prev][customer + 1] + self.pheromone[customer + 1][next_node]
                        score = (pheromone ** self.params['alpha']) * (heuristic ** self.params['beta'])
                        
                        if score > best_score:
                            best_score = score
                            best_vid = vid
                            best_pos = pos
            
            if best_vid is not None:
                solution[best_vid].insert(best_pos, customer)
            else:
                solution[len(solution)] = [customer]
        
        return solution
    
    def _update_pheromone(self, solutions: List[Dict], costs: List[float]):
        for solution, cost in zip(solutions, costs):
            if cost <= 0:
                continue
            delta = 1.0 / cost
            for route in solution.values():
                if not route:
                    continue
                nodes = [0] + [c + 1 for c in route] + [0]
                for i in range(len(nodes) - 1):
                    self.pheromone[nodes[i]][nodes[i+1]] += delta
                    self.pheromone[nodes[i+1]][nodes[i]] += delta


# ============================================================================
# ALGORITHM 6: IALNS (Proposed - Full)
# ============================================================================

class IALNSAlgorithm(BaseAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle)
        self.params = params or {
            'max_iterations': 150,
            'initial_temperature': 100.0,
            'cooling_rate': 0.99,
            'destruction_rate': 0.3,
            'learning_rate': 0.15,
            'stability_weight': 0.35,
            'distance_weight': 0.65,
            'tabu_tenure': 10,
            'adaptive_frequency': 10,
            'max_no_improvement': 50,
            'cg_weight': 0.2,
            'handling_cost_weight': 0.10,
            'overload_penalty_weight': 1.00,
        }
        
        self.destroy_weights = {
            'random': 1.0,
            'worst': 1.2,
            'compatibility': 1.0,
            'imbalance': 1.3,
            'shaw': 1.1
        }
        self.repair_weights = {
            'greedy': 1.0,
            'regret': 1.3,
            'stability_aware': 1.5,
            'balance': 1.1
        }
        self.tabu_list = []
        self.cg_history = []
        
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        solution = self._generate_enhanced_initial_solution()
        current_cost = self.calc.compute_transportation_cost(solution)
        best_solution = solution.copy()
        best_cost = current_cost
        best_metrics = self.calc.compute_all_metrics(solution, iterations_to_convergence=0)
        
        temperature = self.params['initial_temperature']
        iterations = 0
        no_improvement = 0
        
        destroy_performance = defaultdict(list)
        repair_performance = defaultdict(list)
        
        while iterations < self.params['max_iterations']:
            iterations += 1
            
            destroy_op = self._select_adaptive_operator(self.destroy_weights)
            repair_op = self._select_adaptive_operator(self.repair_weights)
            
            destroyed, removed = self._apply_enhanced_destroy(solution, destroy_op)
            repaired = self._apply_enhanced_repair(destroyed, removed, repair_op)
            
            new_cost, stability_score, cg_score = self._evaluate_with_stability_and_cg(repaired)
            
            improvement = (current_cost - new_cost) / current_cost if current_cost > 0 else 0
            destroy_performance[destroy_op].append(improvement)
            repair_performance[repair_op].append(improvement)
            
            cg_metrics = self.calc.compute_cg_metrics(repaired)
            self.cg_history.append({
                'iteration': iterations,
                'cg_mean': cg_metrics['cg_position_mean'],
                'cg_compliance': cg_metrics['cg_compliance_rate']
            })
            
            if self._accept_solution(new_cost, current_cost, temperature, stability_score, cg_score):
                solution = repaired
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = new_cost
                    best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
                    no_improvement = 0
                    self.tabu_list.append(best_solution)
                    if len(self.tabu_list) > self.params['tabu_tenure']:
                        self.tabu_list.pop(0)
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            
            if iterations % self.params['adaptive_frequency'] == 0:
                self._update_adaptive_weights(destroy_performance, repair_performance)
                destroy_performance.clear()
                repair_performance.clear()
            
            temperature *= self.params['cooling_rate']
            
            if no_improvement > 20:
                self.params['stability_weight'] = min(0.6, self.params['stability_weight'] * 1.05)
            
            self.convergence_history.append(best_cost)
            
            if no_improvement > self.params['max_no_improvement']:
                if self._verify_global_stability(best_solution):
                    break
        
        best_metrics = self.calc.compute_all_metrics(best_solution, iterations_to_convergence=iterations)
        best_metrics.cpu_time = time.time() - start_time
        best_metrics.milp_calls = self.milp_calls
        best_metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        best_metrics.iterations_to_convergence = iterations
        
        # Store parameter combination in metrics
        best_metrics.parameter_combination = {
            'handling_cost_weight': self.params.get('handling_cost_weight', 0.10),
            'overload_penalty_weight': self.params.get('overload_penalty_weight', 1.00),
            'stability_weight': self.params.get('stability_weight', 0.35),
        }
        
        extra = {
            'metrics': best_metrics,
            'convergence': self.convergence_history,
            'destroy_weights': self.destroy_weights,
            'repair_weights': self.repair_weights,
            'cg_history': self.cg_history
        }
        
        return best_solution, best_cost, extra
    
    def _generate_enhanced_initial_solution(self) -> Dict[int, List[int]]:
        n_customers = self.n_customers
        unassigned = list(range(n_customers))
        customer_demands = []
        for c in unassigned:
            total_demand = sum(self.instance_data['customers'][c]['demand'].values())
            customer_demands.append((c, total_demand))
        customer_demands.sort(key=lambda x: x[1], reverse=True)
        
        solution = {}
        for customer, _ in customer_demands:
            inserted = False
            best_vid = None
            best_pos = None
            best_stability_score = float('inf')
            
            for vid, route in solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible_with_stability(new_route):
                        stability_score = self._compute_route_stability(new_route)
                        cg_score = self._compute_route_cg_score(new_route)
                        combined_score = 0.7 * stability_score + 0.3 * cg_score
                        if combined_score < best_stability_score:
                            best_stability_score = combined_score
                            best_vid = vid
                            best_pos = pos
                            inserted = True
            
            if inserted:
                solution[best_vid].insert(best_pos, customer)
            else:
                solution[len(solution)] = [customer]
        
        return solution
    
    def _compute_route_cg_score(self, route: List[int]) -> float:
        demands = self._extract_demands(route)
        feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
        if feasible and allocation:
            load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                   for c_id in allocation if allocation[c_id] > 0}
            ldd_result = LDDVerifier(self.vehicle).verify_load(load)
            if ldd_result.feasible:
                optimum_cg = 0.45 * self.vehicle.wheelbase
                deviation = abs(ldd_result.cg_position - optimum_cg) / (0.2 * self.vehicle.wheelbase)
                return min(deviation, 1.0)
        return 1.0
    
    def _is_route_feasible_with_stability(self, route: List[int]) -> bool:
        if not route:
            return True
        route_key = tuple(sorted(route))
        if route_key in self.feasibility_cache:
            self.cache_hits += 1
            return self.feasibility_cache[route_key]
        
        demands = self._extract_demands(route)
        total_volume = sum(demands.values())
        if total_volume > self.vehicle.max_volume:
            self.feasibility_cache[route_key] = False
            return False
        
        self.milp_calls += 1
        feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
        
        if feasible and allocation:
            load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                   for c_id in allocation if allocation[c_id] > 0}
            ldd_result = LDDVerifier(self.vehicle).verify_load(load)
            feasible = ldd_result.feasible and ldd_result.stability_margin > 5.0
        
        self.feasibility_cache[route_key] = feasible
        return feasible
    
    def _compute_route_stability(self, route: List[int]) -> float:
        demands = self._extract_demands(route)
        feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
        if feasible and allocation:
            load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                   for c_id in allocation if allocation[c_id] > 0}
            ldd_result = LDDVerifier(self.vehicle).verify_load(load)
            return 100.0 - ldd_result.stability_margin
        return 100.0
    
    def _select_adaptive_operator(self, weights: Dict[str, float]) -> str:
        total = sum(weights.values())
        if total == 0:
            return random.choice(list(weights.keys()))
        
        if random.random() < 0.15:
            return random.choice(list(weights.keys()))
        
        r = random.random() * total
        cumsum = 0
        for op, w in weights.items():
            cumsum += w
            if r <= cumsum:
                return op
        return list(weights.keys())[-1]
    
    def _apply_enhanced_destroy(self, solution: Dict[int, List[int]], operator: str) -> Tuple[Dict[int, List[int]], List[int]]:
        if operator == 'random':
            return self._destroy_random(solution)
        elif operator == 'worst':
            return self._destroy_worst_enhanced(solution)
        elif operator == 'compatibility':
            return self._destroy_compatibility(solution)
        elif operator == 'imbalance':
            return self._destroy_imbalance_enhanced(solution)
        elif operator == 'shaw':
            return self._destroy_shaw(solution)
        return solution, []
    
    def _destroy_random(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        all_customers = [c for route in new_solution.values() for c in route]
        n_remove = max(2, int(len(all_customers) * self.params['destruction_rate']))
        
        if len(all_customers) > n_remove:
            to_remove = set(random.sample(all_customers, n_remove))
            removed = list(to_remove)
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if c not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_worst_enhanced(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        marginal_costs = {}
        for vid, route in new_solution.items():
            if len(route) <= 1:
                continue
            for idx, customer in enumerate(route):
                route_without = route[:idx] + route[idx+1:]
                if route_without:
                    cost_with = self.calc.compute_transportation_cost({vid: route})
                    cost_without = self.calc.compute_transportation_cost({vid: route_without})
                    marginal_costs[(vid, customer)] = cost_with - cost_without
        
        if marginal_costs:
            n_remove = max(2, int(len(marginal_costs) * self.params['destruction_rate'] * 0.8))
            sorted_customers = sorted(marginal_costs.items(), key=lambda x: x[1], reverse=True)
            to_remove = {item[0] for item in sorted_customers[:n_remove]}
            removed = [c for _, c in to_remove]
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if (vid, c) not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_shaw(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        all_customers = [c for route in new_solution.values() for c in route]
        
        if len(all_customers) < 3:
            return self._destroy_random(solution)
        
        seed = random.choice(all_customers)
        seed_demand = set(self.instance_data['customers'][seed]['demand'].keys())
        
        similar = []
        for customer in all_customers:
            if customer == seed:
                continue
            customer_demand = set(self.instance_data['customers'][customer]['demand'].keys())
            overlap = len(seed_demand.intersection(customer_demand))
            if overlap > 0:
                similar.append((customer, overlap))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        n_remove = max(2, min(len(similar), int(len(all_customers) * self.params['destruction_rate'] * 0.7)))
        
        if similar:
            to_remove = {seed} | {c for c, _ in similar[:n_remove]}
            removed = list(to_remove)
            for vid in list(new_solution.keys()):
                new_solution[vid] = [c for c in new_solution[vid] if c not in to_remove]
                if not new_solution[vid]:
                    del new_solution[vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_compatibility(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        incompatibilities = defaultdict(list)
        
        for vid, route in new_solution.items():
            if len(route) <= 1:
                continue
            for i, c1 in enumerate(route):
                for c2 in route[i+1:]:
                    p1 = set(self.instance_data['customers'][c1]['demand'].keys())
                    p2 = set(self.instance_data['customers'][c2]['demand'].keys())
                    if p1 != p2:
                        incompatibilities[vid].append(c1)
                        incompatibilities[vid].append(c2)
        
        if incompatibilities:
            target_vid = random.choice(list(incompatibilities.keys()))
            incompatible_customers = list(set(incompatibilities[target_vid]))
            n_remove = max(2, min(len(incompatible_customers), int(len(incompatible_customers) * 0.5)))
            to_remove = set(random.sample(incompatible_customers, n_remove))
            removed = list(to_remove)
            new_solution[target_vid] = [c for c in new_solution[target_vid] if c not in to_remove]
            if not new_solution[target_vid]:
                del new_solution[target_vid]
            return new_solution, removed
        return new_solution, []
    
    def _destroy_imbalance_enhanced(self, solution: Dict[int, List[int]]) -> Tuple[Dict[int, List[int]], List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        unstable_routes = []
        
        for vid, route in new_solution.items():
            if len(route) <= 1:
                continue
            if not self._is_route_feasible_with_stability(route):
                unstable_routes.append(vid)
        
        if unstable_routes:
            target_vid = random.choice(unstable_routes)
            route = new_solution[target_vid]
            for idx, customer in enumerate(route):
                test_route = route[:idx] + route[idx+1:]
                if self._is_route_feasible_with_stability(test_route):
                    to_remove = {customer}
                    removed = [customer]
                    new_solution[target_vid] = [c for c in route if c not in to_remove]
                    if not new_solution[target_vid]:
                        del new_solution[target_vid]
                    return new_solution, removed
        return self._destroy_random(solution)
    
    def _apply_enhanced_repair(self, solution: Dict[int, List[int]], unassigned: List[int], operator: str) -> Dict[int, List[int]]:
        if operator == 'greedy':
            return self._repair_greedy(solution, unassigned)
        elif operator == 'regret':
            return self._repair_regret_enhanced(solution, unassigned)
        elif operator == 'stability_aware':
            return self._repair_stability_aware(solution, unassigned)
        elif operator == 'balance':
            return self._repair_balance(solution, unassigned)
        return solution
    
    def _repair_greedy(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        random.shuffle(remaining)
        
        for customer in remaining[:]:
            best_vid = None
            best_pos = None
            best_cost = float('inf')
            
            for vid, route in new_solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible_with_stability(new_route):
                        cost = self.calc.compute_transportation_cost({vid: new_route})
                        if cost < best_cost:
                            best_cost = cost
                            best_vid = vid
                            best_pos = pos
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
        
        for customer in remaining:
            if self._is_route_feasible_with_stability([customer]):
                new_solution[len(new_solution)] = [customer]
            else:
                new_solution[len(new_solution)] = [customer]
        
        return new_solution
    
    def _repair_regret_enhanced(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        
        while remaining:
            regrets = []
            for customer in remaining:
                costs = []
                for vid, route in new_solution.items():
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [customer] + route[pos:]
                        if self._is_route_feasible_with_stability(new_route):
                            costs.append(self.calc.compute_transportation_cost({vid: new_route}))
                
                if len(costs) >= 2:
                    costs.sort()
                    regret = costs[1] - costs[0]
                elif costs:
                    regret = costs[0]
                else:
                    regret = float('inf')
                regrets.append((regret, customer))
            
            if not regrets:
                break
            
            regrets.sort(key=lambda x: x[0], reverse=True)
            if regrets[0][0] == float('inf'):
                customer = remaining.pop(0)
                new_solution[len(new_solution)] = [customer]
                continue
            
            customer = regrets[0][1]
            best_vid = None
            best_pos = None
            best_cost = float('inf')
            
            for vid, route in new_solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if self._is_route_feasible_with_stability(new_route):
                        cost = self.calc.compute_transportation_cost({vid: new_route})
                        if cost < best_cost:
                            best_cost = cost
                            best_vid = vid
                            best_pos = pos
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
            else:
                new_solution[len(new_solution)] = [customer]
                remaining.remove(customer)
        
        return new_solution
    
    def _repair_stability_aware(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        
        for customer in remaining[:]:
            best_vid = None
            best_pos = None
            best_score = float('inf')
            
            for vid, route in new_solution.items():
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [customer] + route[pos:]
                    if not self._is_route_feasible_with_stability(new_route):
                        continue
                    
                    demands = self._extract_demands(new_route)
                    feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
                    
                    stability_penalty = 0.0
                    cg_penalty = 0.0
                    if feasible and allocation:
                        load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                               for c_id in allocation if allocation[c_id] > 0}
                        ldd_result = LDDVerifier(self.vehicle).verify_load(load)
                        stability_penalty = 100.0 - ldd_result.stability_margin
                        optimum_cg = 0.45 * self.vehicle.wheelbase
                        cg_penalty = abs(ldd_result.cg_position - optimum_cg) / (0.2 * self.vehicle.wheelbase)
                    else:
                        stability_penalty = 100.0
                        cg_penalty = 1.0
                    
                    dist_cost = self.calc.compute_transportation_cost({vid: new_route})
                    score = (1 - self.params['stability_weight']) * dist_cost + \
                            self.params['stability_weight'] * stability_penalty + \
                            0.1 * self.params['cg_weight'] * cg_penalty * dist_cost
                    
                    if score < best_score:
                        best_score = score
                        best_vid = vid
                        best_pos = pos
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
        
        for customer in remaining:
            if self._is_route_feasible_with_stability([customer]):
                new_solution[len(new_solution)] = [customer]
        
        return new_solution
    
    def _repair_balance(self, solution: Dict[int, List[int]], unassigned: List[int]) -> Dict[int, List[int]]:
        new_solution = {k: v.copy() for k, v in solution.items()}
        remaining = unassigned[:]
        random.shuffle(remaining)
        
        for customer in remaining[:]:
            best_vid = None
            best_pos = None
            min_load = float('inf')
            
            for vid, route in new_solution.items():
                current_demand = sum(self._extract_demands(route).values())
                if current_demand < min_load:
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [customer] + route[pos:]
                        if self._is_route_feasible_with_stability(new_route):
                            min_load = current_demand
                            best_vid = vid
                            best_pos = pos
                            break
            
            if best_vid is not None:
                new_solution[best_vid].insert(best_pos, customer)
                remaining.remove(customer)
        
        for customer in remaining:
            new_solution[len(new_solution)] = [customer]
        
        return new_solution
    
    def _evaluate_with_stability_and_cg(self, solution: Dict[int, List[int]]) -> Tuple[float, float, float]:
        total_cost = self.calc.compute_transportation_cost(solution)
        stability_scores = []
        cg_scores = []
        
        for route in solution.values():
            if not route:
                continue
            stability_score = self._compute_route_stability(route)
            stability_scores.append(stability_score)
            cg_score = self._compute_route_cg_score(route)
            cg_scores.append(cg_score)
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0
        avg_cg = np.mean(cg_scores) if cg_scores else 0
        return total_cost, avg_stability, avg_cg
    
    def _accept_solution(self, new_cost: float, current_cost: float, temperature: float, 
                         stability_score: float, cg_score: float) -> bool:
        if new_cost < current_cost:
            return True
        
        if stability_score < 20:
            if new_cost < current_cost * 1.1:
                return True
        
        if cg_score < 0.3:
            if new_cost < current_cost * 1.05:
                return True
        
        if temperature > 1e-10:
            prob = math.exp(-(new_cost - current_cost) / temperature)
            return random.random() < prob
        
        return False
    
    def _update_adaptive_weights(self, destroy_perf: Dict, repair_perf: Dict):
        lr = self.params['learning_rate']
        
        for op, scores in destroy_perf.items():
            if scores:
                avg_score = np.mean(scores)
                self.destroy_weights[op] = (1 - lr) * self.destroy_weights[op] + lr * max(0.1, avg_score * 2)
        
        for op, scores in repair_perf.items():
            if scores:
                avg_score = np.mean(scores)
                self.repair_weights[op] = (1 - lr) * self.repair_weights[op] + lr * max(0.1, avg_score * 2)
        
        total_destroy = sum(self.destroy_weights.values())
        if total_destroy > 0:
            for op in self.destroy_weights:
                self.destroy_weights[op] /= total_destroy
        
        total_repair = sum(self.repair_weights.values())
        if total_repair > 0:
            for op in self.repair_weights:
                self.repair_weights[op] /= total_repair
    
    def _verify_global_stability(self, solution: Dict[int, List[int]]) -> bool:
        violation_count = 0
        total_routes = 0
        
        for route in solution.values():
            if not route:
                continue
            total_routes += 1
            if not self._is_route_feasible_with_stability(route):
                violation_count += 1
        
        violation_rate = violation_count / max(1, total_routes) * 100
        return violation_rate < 5.0


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

class IALNS_NoStability(IALNSAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle, params)
        self.params['stability_weight'] = 0.0
        
    def _is_route_feasible_with_stability(self, route: List[int]) -> bool:
        if not route:
            return True
        route_key = tuple(sorted(route))
        if route_key in self.feasibility_cache:
            self.cache_hits += 1
            return self.feasibility_cache[route_key]
        
        demands = self._extract_demands(route)
        total_volume = sum(demands.values())
        if total_volume > self.vehicle.max_volume:
            self.feasibility_cache[route_key] = False
            return False
        self.feasibility_cache[route_key] = True
        return True
    
    def _compute_route_stability(self, route: List[int]) -> float:
        return 0.0
    
    def _compute_route_cg_score(self, route: List[int]) -> float:
        return 0.0

class IALNS_NoAdaptive(IALNSAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle, params)
        
    def _select_adaptive_operator(self, weights: Dict[str, float]) -> str:
        return random.choice(list(weights.keys()))
    
    def _update_adaptive_weights(self, destroy_perf: Dict, repair_perf: Dict):
        pass

class IALNS_Sequential(IALNSAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle, params)
        
    def _is_route_feasible_with_stability(self, route: List[int]) -> bool:
        if not route:
            return True
        demands = self._extract_demands(route)
        total_volume = sum(demands.values())
        if total_volume > self.vehicle.max_volume:
            return False
        return True
    
    def solve(self) -> Tuple[Dict[int, List[int]], float, Dict]:
        start_time = time.time()
        
        solution = self._generate_enhanced_initial_solution()
        current_cost = self.calc.compute_transportation_cost(solution)
        best_solution = solution.copy()
        best_cost = current_cost
        
        temperature = self.params['initial_temperature']
        iterations = 0
        no_improvement = 0
        
        while iterations < self.params['max_iterations']:
            iterations += 1
            
            destroy_op = self._select_adaptive_operator(self.destroy_weights)
            repair_op = self._select_adaptive_operator(self.repair_weights)
            
            destroyed, removed = self._apply_enhanced_destroy(solution, destroy_op)
            repaired = self._apply_enhanced_repair(destroyed, removed, repair_op)
            
            new_cost = self.calc.compute_transportation_cost(repaired)
            
            if new_cost < current_cost or random.random() < math.exp(-(new_cost - current_cost) / temperature):
                solution = repaired
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = solution.copy()
                    best_cost = new_cost
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            
            temperature *= self.params['cooling_rate']
            
            if no_improvement > self.params['max_no_improvement']:
                break
        
        final_solution = {}
        for vid, route in best_solution.items():
            if not route:
                continue
            
            demands = self._extract_demands(route)
            feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
            
            if feasible:
                final_solution[vid] = route
            else:
                if len(route) > 1:
                    best_split = None
                    best_split_cost = float('inf')
                    for i in range(1, len(route)):
                        route1 = route[:i]
                        route2 = route[i:]
                        demands1 = self._extract_demands(route1)
                        demands2 = self._extract_demands(route2)
                        feasible1, _ = self.saca.solve_heuristic(demands1, self.vehicle.product_densities)
                        feasible2, _ = self.saca.solve_heuristic(demands2, self.vehicle.product_densities)
                        if feasible1 and feasible2:
                            cost = self.calc.compute_transportation_cost({0: route1, 1: route2})
                            if cost < best_split_cost:
                                best_split_cost = cost
                                best_split = [route1, route2]
                    
                    if best_split:
                        final_solution[len(final_solution)] = best_split[0]
                        final_solution[len(final_solution)] = best_split[1]
                    else:
                        for c in route:
                            final_solution[len(final_solution)] = [c]
                else:
                    final_solution[vid] = route
        
        metrics = self.calc.compute_all_metrics(final_solution, iterations_to_convergence=iterations)
        metrics.cpu_time = time.time() - start_time
        metrics.milp_calls = self.milp_calls
        metrics.cache_hit_rate = (self.cache_hits / max(1, self.milp_calls + self.cache_hits)) * 100
        metrics.iterations_to_convergence = iterations
        
        return final_solution, self.calc.compute_transportation_cost(final_solution), {'metrics': metrics}

class IALNS_NoCache(IALNSAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle, params)
        self.feasibility_cache = {}
        self.cache_hits = 0
        
    def _is_route_feasible_with_stability(self, route: List[int]) -> bool:
        if not route:
            return True
        demands = self._extract_demands(route)
        total_volume = sum(demands.values())
        if total_volume > self.vehicle.max_volume:
            return False
        
        self.milp_calls += 1
        feasible, allocation = self.saca.solve_heuristic(demands, self.vehicle.product_densities)
        
        if feasible and allocation:
            load = {c_id: allocation[c_id] * self.vehicle.product_densities.get(1, 0.8)
                   for c_id in allocation if allocation[c_id] > 0}
            ldd_result = LDDVerifier(self.vehicle).verify_load(load)
            feasible = ldd_result.feasible and ldd_result.stability_margin > 5.0
        
        return feasible

class IALNS_NoImbalance(IALNSAlgorithm):
    def __init__(self, instance_data: Dict, vehicle: Vehicle, params: Dict = None):
        super().__init__(instance_data, vehicle, params)
        self.destroy_weights = {
            'random': 1.0,
            'worst': 1.2,
            'compatibility': 1.0,
            'shaw': 1.1
        }
        
    def _apply_enhanced_destroy(self, solution: Dict[int, List[int]], operator: str) -> Tuple[Dict[int, List[int]], List[int]]:
        if operator == 'random':
            return self._destroy_random(solution)
        elif operator == 'worst':
            return self._destroy_worst_enhanced(solution)
        elif operator == 'compatibility':
            return self._destroy_compatibility(solution)
        elif operator == 'shaw':
            return self._destroy_shaw(solution)
        return solution, []


# ============================================================================
# SENSITIVITY ANALYSIS MODULE
# ============================================================================

class SensitivityAnalyzer:
    def __init__(self, benchmark_dir: str, output_dir: str, 
                 instance_subset: List[str] = None):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.instance_subset = instance_subset
        os.makedirs(os.path.join(output_dir, 'part3_sensitivity'), exist_ok=True)
        
        self.lambda_H_values = [0.05, 0.075, 0.10, 0.125, 0.15]
        self.lambda_W_values = [0.50, 0.75, 1.00, 1.25, 1.50]
        self.lambda_S_values = [0.25, 0.375, 0.50, 0.625, 0.75]
        
        self.baseline_H = 0.10
        self.baseline_W = 1.00
        self.baseline_S = 0.50
        
        self.results: List[SensitivityResult] = []
        
    def load_instance(self, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_vehicle(self, instance: Dict) -> Vehicle:
        vehicle_data = instance['vehicle']
        compartments = [Compartment(
            id=comp['id'],
            capacity=comp['capacity'],
            position_x=comp['position_x'],
            position_y=comp['position_y'],
            load_density=1.0
        ) for comp in vehicle_data['compartments']]
        
        compartment_positions = {comp['id']: comp['position_x'] for comp in vehicle_data['compartments']}
        
        return Vehicle(
            id=0,
            compartments=compartments,
            max_payload=vehicle_data['max_payload'],
            front_axle_max=vehicle_data['front_axle_max'],
            rear_axle_max=vehicle_data['rear_axle_max'],
            wheelbase=vehicle_data['wheelbase'],
            unladen_weight=vehicle_data['unladen_weight'],
            max_volume=sum(c.capacity for c in compartments),
            steering_min_ratio=vehicle_data['steering_min_ratio'],
            driving_min_ratio=vehicle_data['driving_min_ratio'],
            product_densities=vehicle_data.get('product_densities', {}),
            fixed_cost=100.0,
            cost_per_km=1.0,
            front_axle_min=vehicle_data.get('front_axle_min', vehicle_data['front_axle_max'] * 0.1),
            rear_axle_min=vehicle_data.get('rear_axle_min', vehicle_data['rear_axle_max'] * 0.1),
            cg_min=vehicle_data.get('cg_min', 0.3 * vehicle_data['wheelbase']),
            cg_max=vehicle_data.get('cg_max', 0.7 * vehicle_data['wheelbase']),
            compartment_positions=compartment_positions,
            is_multi_compartment=True
        )
    
    def build_instance_data(self, instance: Dict) -> Dict:
        n = len(instance['customers'])
        coords = [[0, 0]] + [[c['x'], c['y']] for c in instance['customers']]
        dist_matrix = [[math.sqrt((coords[i][0] - coords[j][0])**2 + 
                                  (coords[i][1] - coords[j][1])**2) 
                       for j in range(n + 1)] for i in range(n + 1)]
        
        return {
            'name': instance['name'],
            'customers': instance['customers'],
            'time_windows': instance['time_windows'],
            'distance_matrix': dist_matrix,
            'vehicle': instance['vehicle'],
            'n_products': instance.get('n_products', 3),
            'n_customers': len(instance['customers']),
            'n_compartments': instance.get('n_compartments', 
                                           len(instance['vehicle']['compartments']))
        }
    
    def create_ialns_with_params(self, instance_data: Dict, vehicle: Vehicle,
                                  lambda_H: float, lambda_W: float, lambda_S: float) -> IALNSAlgorithm:
        params = {
            'max_iterations': 150,
            'initial_temperature': 100.0,
            'cooling_rate': 0.99,
            'destruction_rate': 0.3,
            'learning_rate': 0.15,
            'stability_weight': lambda_S,
            'distance_weight': 1.0 - lambda_S,
            'tabu_tenure': 10,
            'adaptive_frequency': 10,
            'max_no_improvement': 50,
            'cg_weight': 0.2,
            'handling_cost_weight': lambda_H,
            'overload_penalty_weight': lambda_W,
        }
        
        return IALNSAlgorithm(instance_data, vehicle, params)
    
    def run_single_configuration(self, instance_file: str, lambda_H: float,
                                  lambda_W: float, lambda_S: float, 
                                  num_runs: int = 3) -> Optional[SensitivityResult]:
        try:
            instance = self.load_instance(instance_file)
            vehicle = self.create_vehicle(instance)
            instance_data = self.build_instance_data(instance)
            
            costs = []
            stabilities = []
            ldd_violations = []
            fleet_sizes = []
            cg_compliances = []
            cpu_times = []
            convergences = []
            
            for run in range(num_runs):
                random.seed(42 + run * 7)
                np.random.seed(42 + run * 7)
                
                algo = self.create_ialns_with_params(
                    instance_data, vehicle, lambda_H, lambda_W, lambda_S
                )
                start_time = time.time()
                solution, cost, extra = algo.solve()
                elapsed = time.time() - start_time
                
                if solution:
                    metrics = extra.get('metrics')
                    if metrics is None:
                        calc = ObjectiveCalculator(instance_data, vehicle)
                        metrics = calc.compute_all_metrics(solution)
                    
                    costs.append(cost)
                    stabilities.append(metrics.stability_margin)
                    ldd_violations.append(metrics.ldd_violation_rate)
                    fleet_sizes.append(metrics.fleet_size)
                    cg_compliances.append(metrics.cg_compliance_rate)
                    cpu_times.append(elapsed)
                    
                    if 'convergence' in extra:
                        convergences.append(len(extra['convergence']))
            
            if costs:
                param_id = f"H{lambda_H:.3f}_W{lambda_W:.3f}_S{lambda_S:.3f}"
                return SensitivityResult(
                    lambda_H=lambda_H,
                    lambda_W=lambda_W,
                    lambda_S=lambda_S,
                    avg_cost=np.mean(costs),
                    avg_stability=np.mean(stabilities),
                    avg_ldd_violation=np.mean(ldd_violations),
                    avg_fleet_size=np.mean(fleet_sizes),
                    avg_cg_compliance=np.mean(cg_compliances),
                    avg_cpu_time=np.mean(cpu_times),
                    std_cost=np.std(costs),
                    std_stability=np.std(stabilities),
                    convergence_iterations=int(np.mean(convergences)) if convergences else 0,
                    parameter_combination_id=param_id
                )
            return None
            
        except Exception as e:
            print(f"    Error in configuration H={lambda_H}, W={lambda_W}, S={lambda_S}: {e}")
            return None
    
    def run_sensitivity_analysis(self, max_instances: int = 10, 
                                  num_runs: int = 3) -> pd.DataFrame:
        print("\n" + "=" * 80)
        print("PART 3: SENSITIVITY ANALYSIS")
        print("=" * 80)
        print(f"Testing {len(self.lambda_H_values)} × {len(self.lambda_W_values)} × "
              f"{len(self.lambda_S_values)} = {len(self.lambda_H_values) * len(self.lambda_W_values) * len(self.lambda_S_values)} "
              f"parameter combinations")
        print(f"Parameters:")
        print(f"  λ_H (Handling Cost): {self.lambda_H_values}")
        print(f"  λ_W (Overload Penalty): {self.lambda_W_values}")
        print(f"  λ_S (Stability Penalty): {self.lambda_S_values}")
        print(f"  Baseline: λ_H={self.baseline_H}, λ_W={self.baseline_W}, λ_S={self.baseline_S}")
        print("-" * 60)
        
        instance_files = []
        for root, dirs, files in os.walk(self.benchmark_dir):
            for file in files:
                if file.endswith('.json'):
                    instance_files.append(os.path.join(root, file))
        
        instance_files.sort(key=lambda x: os.path.basename(x))
        
        if max_instances > 0:
            by_size = defaultdict(list)
            for f in instance_files:
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        n = len(data.get('customers', []))
                        if n <= 5:
                            by_size['small'].append(f)
                        elif n <= 10:
                            by_size['medium'].append(f)
                        else:
                            by_size['large'].append(f)
                except:
                    continue
            
            selected = []
            for size_category in ['small', 'medium', 'large']:
                if by_size[size_category]:
                    n_take = min(3, len(by_size[size_category]))
                    selected.extend(by_size[size_category][:n_take])
            
            instance_files = selected if selected else instance_files[:max_instances]
        
        print(f"Using {len(instance_files)} instances for sensitivity analysis")
        
        all_results = []
        total_combinations = len(self.lambda_H_values) * len(self.lambda_W_values) * len(self.lambda_S_values)
        current = 0
        
        for lambda_H in self.lambda_H_values:
            for lambda_W in self.lambda_W_values:
                for lambda_S in self.lambda_S_values:
                    current += 1
                    print(f"\nTesting combination {current}/{total_combinations}: "
                          f"λ_H={lambda_H:.3f}, λ_W={lambda_W:.3f}, λ_S={lambda_S:.3f}")
                    
                    combo_results = []
                    for instance_file in instance_files:
                        result = self.run_single_configuration(
                            instance_file, lambda_H, lambda_W, lambda_S, num_runs
                        )
                        if result:
                            combo_results.append(result)
                    
                    if combo_results:
                        avg_result = self._average_sensitivity_results(combo_results)
                        all_results.append(avg_result)
        
        df = self._results_to_dataframe(all_results)
        
        csv_path = os.path.join(self.output_dir, 'part3_sensitivity', 
                                'sensitivity_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSensitivity analysis results saved to: {csv_path}")
        
        self._generate_sensitivity_figures(df)
        
        summary = self.generate_sensitivity_summary(df)
        summary_df = pd.DataFrame([{
            'Metric': k,
            'Value': v if not isinstance(v, dict) else str(v)
        } for k, v in summary.items()])
        summary_df.to_csv(os.path.join(self.output_dir, 'part3_sensitivity', 
                                       'sensitivity_summary.csv'), index=False)
        
        return df
    
    def _average_sensitivity_results(self, results: List[SensitivityResult]) -> SensitivityResult:
        if not results:
            return None
        
        n = len(results)
        return SensitivityResult(
            lambda_H=results[0].lambda_H,
            lambda_W=results[0].lambda_W,
            lambda_S=results[0].lambda_S,
            avg_cost=np.mean([r.avg_cost for r in results]),
            avg_stability=np.mean([r.avg_stability for r in results]),
            avg_ldd_violation=np.mean([r.avg_ldd_violation for r in results]),
            avg_fleet_size=np.mean([r.avg_fleet_size for r in results]),
            avg_cg_compliance=np.mean([r.avg_cg_compliance for r in results]),
            avg_cpu_time=np.mean([r.avg_cpu_time for r in results]),
            std_cost=np.std([r.std_cost for r in results]),
            std_stability=np.std([r.std_stability for r in results]),
            convergence_iterations=int(np.mean([r.convergence_iterations for r in results])),
            parameter_combination_id=results[0].parameter_combination_id
        )
    
    def _results_to_dataframe(self, results: List[SensitivityResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {
                'lambda_H': r.lambda_H,
                'lambda_W': r.lambda_W,
                'lambda_S': r.lambda_S,
                'avg_cost': r.avg_cost,
                'avg_stability': r.avg_stability,
                'avg_ldd_violation': r.avg_ldd_violation,
                'avg_fleet_size': r.avg_fleet_size,
                'avg_cg_compliance': r.avg_cg_compliance,
                'avg_cpu_time': r.avg_cpu_time,
                'std_cost': r.std_cost,
                'std_stability': r.std_stability,
                'convergence_iterations': r.convergence_iterations,
                'param_id': r.parameter_combination_id,
                'is_baseline': (r.lambda_H == self.baseline_H and 
                               r.lambda_W == self.baseline_W and 
                               r.lambda_S == self.baseline_S)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_sensitivity_figures(self, df: pd.DataFrame):
        output_dir = os.path.join(self.output_dir, 'part3_sensitivity')
        
        self._plot_sensitivity_heatmap(df, output_dir)
        self._plot_parameter_impact(df, output_dir)
        self._plot_stability_cost_tradeoff(df, output_dir)
        self._plot_optimal_parameters(df, output_dir)
        self._plot_convergence_sensitivity(df, output_dir)
    
    def _plot_sensitivity_heatmap(self, df: pd.DataFrame, output_dir: str):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        lambda_S_values = sorted(df['lambda_S'].unique())
        
        for idx, lambda_S in enumerate(lambda_S_values[:3]):
            subset = df[df['lambda_S'] == lambda_S]
            if subset.empty:
                continue
            
            pivot = subset.pivot(index='lambda_W', columns='lambda_H', values='avg_cost')
            
            im = axes[idx].imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                                  vmin=df['avg_cost'].min(), vmax=df['avg_cost'].max())
            
            axes[idx].set_xticks(range(len(pivot.columns)))
            axes[idx].set_xticklabels([f'{x:.3f}' for x in pivot.columns])
            axes[idx].set_yticks(range(len(pivot.index)))
            axes[idx].set_yticklabels([f'{y:.3f}' for y in pivot.index])
            axes[idx].set_xlabel('λ_H (Handling Cost)', fontsize=11)
            axes[idx].set_ylabel('λ_W (Overload Penalty)', fontsize=11)
            axes[idx].set_title(f'λ_S = {lambda_S:.3f}', fontsize=12)
            
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    value = pivot.iloc[i, j]
                    if not pd.isna(value):
                        axes[idx].text(j, i, f'{value:.0f}', ha='center', va='center',
                                       fontsize=8, color='black' if value < df['avg_cost'].mean() else 'white')
        
        plt.suptitle('Sensitivity Analysis: Cost Heatmap', fontsize=14)
        plt.colorbar(im, ax=axes, label='Transportation Cost ($)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_S1_Sensitivity_Heatmap.pdf'), dpi=300)
        plt.close()
        print("  Generated: Figure_S1_Sensitivity_Heatmap")
    
    def _plot_parameter_impact(self, df: pd.DataFrame, output_dir: str):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        parameters = [
            ('lambda_H', 'λ_H (Handling Cost)'),
            ('lambda_W', 'λ_W (Overload Penalty)'),
            ('lambda_S', 'λ_S (Stability Penalty)')
        ]
        
        for idx, (param, label) in enumerate(parameters):
            grouped = df.groupby(param)['avg_cost'].agg(['mean', 'std']).reset_index()
            ax = axes[0, idx]
            ax.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'],
                       marker='o', capsize=4, linewidth=2, markersize=8, color='#2E86AB')
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Average Cost ($)', fontsize=11)
            ax.set_title(f'Cost vs {label}', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            grouped = df.groupby(param)['avg_stability'].agg(['mean', 'std']).reset_index()
            ax = axes[1, idx]
            ax.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'],
                       marker='s', capsize=4, linewidth=2, markersize=8, color='#E76F51')
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel('Stability Margin (%)', fontsize=11)
            ax.set_title(f'Stability vs {label}', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1)
        
        plt.suptitle('Parameter Impact Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_S2_Parameter_Impact.pdf'), dpi=300)
        plt.close()
        print("  Generated: Figure_S2_Parameter_Impact")
    
    def _plot_stability_cost_tradeoff(self, df: pd.DataFrame, output_dir: str):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(df['avg_cost'], df['avg_stability'], 
                            c=df['lambda_S'], cmap='viridis', s=100, alpha=0.7)
        
        baseline = df[df['is_baseline']]
        if not baseline.empty:
            ax.scatter(baseline['avg_cost'], baseline['avg_stability'], 
                      c='red', s=200, marker='*', label='Baseline', edgecolor='black', linewidth=2)
        
        ax.set_xlabel('Transportation Cost ($)', fontsize=12)
        ax.set_ylabel('Stability Margin (%)', fontsize=12)
        ax.set_title('Stability vs Cost Trade-off Across Parameter Configurations', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1, label='Safety Threshold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('λ_S (Stability Penalty)', fontsize=11)
        
        self._annotate_pareto_frontier(df, ax)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_S3_Stability_Cost_Tradeoff.pdf'), dpi=300)
        plt.close()
        print("  Generated: Figure_S3_Stability_Cost_Tradeoff")
    
    def _annotate_pareto_frontier(self, df: pd.DataFrame, ax):
        points = df[['avg_cost', 'avg_stability']].values
        pareto = []
        for i, p in enumerate(points):
            dominated = False
            for j, q in enumerate(points):
                if i != j:
                    if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                        dominated = True
                        break
            if not dominated:
                pareto.append(i)
        
        if pareto:
            pareto_points = points[pareto]
            pareto_points = pareto_points[pareto_points[:, 0].argsort()]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1], 'k--', linewidth=2, label='Pareto Frontier')
            
            for idx in pareto[:5]:
                row = df.iloc[idx]
                label = f"λH={row['lambda_H']:.2f}\nλW={row['lambda_W']:.2f}\nλS={row['lambda_S']:.2f}"
                ax.annotate(label, xy=(row['avg_cost'], row['avg_stability']),
                           xytext=(5, 5), textcoords='offset points', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _plot_optimal_parameters(self, df: pd.DataFrame, output_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        normalized_cost = (df['avg_cost'] - df['avg_cost'].min()) / (df['avg_cost'].max() - df['avg_cost'].min())
        normalized_stability = (df['avg_stability'] - df['avg_stability'].min()) / (df['avg_stability'].max() - df['avg_stability'].min())
        combined_score = 0.6 * (1 - normalized_cost) + 0.4 * normalized_stability
        
        df = df.copy()
        df['combined_score'] = combined_score
        
        top_configs = df.nlargest(10, 'combined_score')
        
        ax = axes[0]
        x = range(len(top_configs))
        bars = ax.bar(x, top_configs['combined_score'], color='#2E86AB')
        ax.set_xlabel('Configuration Rank', fontsize=12)
        ax.set_ylabel('Combined Score', fontsize=12)
        ax.set_title('Top 10 Parameter Configurations', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i+1}" for i in range(len(top_configs))])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        for i, (_, row) in enumerate(top_configs.iterrows()):
            label = f"λH={row['lambda_H']:.2f}\nλW={row['lambda_W']:.2f}\nλS={row['lambda_S']:.2f}"
            ax.text(i, row['combined_score'] + 0.02, label, ha='center', va='bottom', fontsize=7)
        
        baseline_idx = top_configs[top_configs['is_baseline']].index
        if not baseline_idx.empty and len(baseline_idx) > 0:
            pos = top_configs.index.get_loc(baseline_idx[0])
            bars[pos].set_color('red')
            ax.text(pos, top_configs.loc[baseline_idx[0], 'combined_score'] + 0.05, 
                   'BASELINE', ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')
        
        ax = axes[1]
        param_ranges = {
            'λ_H': top_configs['lambda_H'].values,
            'λ_W': top_configs['lambda_W'].values,
            'λ_S': top_configs['lambda_S'].values
        }
        
        ax.boxplot([param_ranges['λ_H'], param_ranges['λ_W'], param_ranges['λ_S']],
                  labels=['λ_H', 'λ_W', 'λ_S'])
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title('Parameter Distribution in Top Configurations', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        ax.axhline(y=0.10, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='λ_H Baseline')
        ax.axhline(y=1.00, color='green', linestyle='--', linewidth=1, alpha=0.5, label='λ_W Baseline')
        ax.axhline(y=0.50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='λ_S Baseline')
        ax.legend(fontsize=8)
        
        plt.suptitle('Optimal Parameter Identification', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_S4_Optimal_Parameters.pdf'), dpi=300)
        plt.close()
        print("  Generated: Figure_S4_Optimal_Parameters")
    
    def _plot_convergence_sensitivity(self, df: pd.DataFrame, output_dir: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        parameters = [
            ('lambda_H', 'λ_H (Handling Cost)'),
            ('lambda_W', 'λ_W (Overload Penalty)'),
            ('lambda_S', 'λ_S (Stability Penalty)')
        ]
        
        x_positions = []
        x_labels = []
        y_values = []
        y_errors = []
        colors = []
        
        for param, label in parameters:
            grouped = df.groupby(param)['convergence_iterations'].agg(['mean', 'std']).reset_index()
            for _, row in grouped.iterrows():
                x_positions.append(len(x_positions))
                x_labels.append(f"{row[param]:.3f}")
                y_values.append(row['mean'])
                y_errors.append(row['std'])
                if param == 'lambda_H':
                    colors.append('#2E86AB')
                elif param == 'lambda_W':
                    colors.append('#E76F51')
                else:
                    colors.append('#1A7A5A')
        
        bars = ax.bar(x_positions, y_values, yerr=y_errors, capsize=5, color=colors, alpha=0.8)
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('Convergence Iterations', fontsize=12)
        ax.set_title('Parameter Impact on Convergence Speed', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='λ_H'),
            Patch(facecolor='#E76F51', label='λ_W'),
            Patch(facecolor='#1A7A5A', label='λ_S')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Figure_S5_Convergence_Sensitivity.pdf'), dpi=300)
        plt.close()
        print("  Generated: Figure_S5_Convergence_Sensitivity")
    
    def generate_sensitivity_summary(self, df: pd.DataFrame) -> Dict:
        baseline = df[df['is_baseline']]
        
        summary = {
            'baseline_cost': baseline['avg_cost'].mean() if not baseline.empty else None,
            'baseline_stability': baseline['avg_stability'].mean() if not baseline.empty else None,
            'min_cost': df['avg_cost'].min(),
            'max_cost': df['avg_cost'].max(),
            'min_stability': df['avg_stability'].min(),
            'max_stability': df['avg_stability'].max(),
            'optimal_cost_config': {
                'lambda_H': df.loc[df['avg_cost'].idxmin()]['lambda_H'],
                'lambda_W': df.loc[df['avg_cost'].idxmin()]['lambda_W'],
                'lambda_S': df.loc[df['avg_cost'].idxmin()]['lambda_S'],
                'cost': df['avg_cost'].min(),
                'stability': df.loc[df['avg_cost'].idxmin()]['avg_stability']
            },
            'optimal_stability_config': {
                'lambda_H': df.loc[df['avg_stability'].idxmax()]['lambda_H'],
                'lambda_W': df.loc[df['avg_stability'].idxmax()]['lambda_W'],
                'lambda_S': df.loc[df['avg_stability'].idxmax()]['lambda_S'],
                'cost': df.loc[df['avg_stability'].idxmax()]['avg_cost'],
                'stability': df['avg_stability'].max()
            },
            'parameter_sensitivity': {}
        }
        
        for param in ['lambda_H', 'lambda_W', 'lambda_S']:
            grouped = df.groupby(param)['avg_cost'].agg(['mean', 'std'])
            sensitivity = (grouped['mean'].max() - grouped['mean'].min()) / grouped['mean'].mean()
            summary['parameter_sensitivity'][param] = sensitivity
        
        return summary


# ============================================================================
# EXPERIMENTAL COMPARISON - PART 1: SOTA
# ============================================================================

class SOTAComparison:
    def __init__(self, benchmark_dir: str, output_dir: str):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
        self.algorithms = {
            'IALNS': IALNSAlgorithm,
            'Standard_ALNS': StandardALNS,
            'FDAHS': FDAHSAlgorithm,
            'GCOF': GCOFAlgorithm,
            'HSA': HSAAlgorithm,
            'ACO_2opt': ACO2optAlgorithm,
        }
        
    def load_instance(self, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_vehicle(self, instance: Dict, multi_compartment: bool = True) -> Vehicle:
        vehicle_data = instance['vehicle']
        compartments = [Compartment(
            id=comp['id'],
            capacity=comp['capacity'],
            position_x=comp['position_x'],
            position_y=comp['position_y'],
            load_density=1.0
        ) for comp in vehicle_data['compartments']]
        
        compartment_positions = {comp['id']: comp['position_x'] for comp in vehicle_data['compartments']}
        
        if not multi_compartment:
            total_capacity = sum(c['capacity'] for c in compartments)
            avg_position = np.mean([c.position_x for c in compartments])
            compartments = [Compartment(id=0, capacity=total_capacity, position_x=avg_position, position_y=0.0, load_density=1.0)]
            compartment_positions = {0: avg_position}
        
        return Vehicle(
            id=0,
            compartments=compartments,
            max_payload=vehicle_data['max_payload'],
            front_axle_max=vehicle_data['front_axle_max'],
            rear_axle_max=vehicle_data['rear_axle_max'],
            wheelbase=vehicle_data['wheelbase'],
            unladen_weight=vehicle_data['unladen_weight'],
            max_volume=sum(c.capacity for c in compartments),
            steering_min_ratio=vehicle_data['steering_min_ratio'],
            driving_min_ratio=vehicle_data['driving_min_ratio'],
            product_densities=vehicle_data.get('product_densities', {}),
            fixed_cost=100.0,
            cost_per_km=1.0,
            front_axle_min=vehicle_data.get('front_axle_min', vehicle_data['front_axle_max'] * 0.1),
            rear_axle_min=vehicle_data.get('rear_axle_min', vehicle_data['rear_axle_max'] * 0.1),
            cg_min=vehicle_data.get('cg_min', 0.3 * vehicle_data['wheelbase']),
            cg_max=vehicle_data.get('cg_max', 0.7 * vehicle_data['wheelbase']),
            compartment_positions=compartment_positions,
            is_multi_compartment=multi_compartment
        )
    
    def build_instance_data(self, instance: Dict) -> Dict:
        n = len(instance['customers'])
        coords = [[0, 0]] + [[c['x'], c['y']] for c in instance['customers']]
        dist_matrix = [[math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
                       for j in range(n + 1)] for i in range(n + 1)]
        
        return {
            'name': instance['name'],
            'customers': instance['customers'],
            'time_windows': instance['time_windows'],
            'distance_matrix': dist_matrix,
            'vehicle': instance['vehicle'],
            'n_products': instance.get('n_products', 3),
            'n_customers': len(instance['customers']),
            'n_compartments': instance.get('n_compartments', len(instance['vehicle']['compartments']))
        }
    
    def run_single_instance(self, instance_file: str, num_runs: int = 3, 
                           multi_compartment: bool = True) -> Dict:
        instance = self.load_instance(instance_file)
        vehicle = self.create_vehicle(instance, multi_compartment)
        instance_data = self.build_instance_data(instance)
        
        results = {
            'instance': instance['name'],
            'n_customers': len(instance['customers']),
            'n_compartments': instance.get('n_compartments', len(instance['vehicle']['compartments'])),
            'layer': instance.get('layer', 'Unknown'),
            'n_products': instance.get('n_products', 3),
            'vehicle_type': 'Multi-Compartment' if multi_compartment else 'Single-Compartment',
            'results': {}
        }
        
        print(f"\n  Testing on {instance['name']} ({len(instance['customers'])} customers, "
              f"{instance.get('n_products', 3)} commodities, {results['n_compartments']} compartments, {results['vehicle_type']})")
        
        for algo_name, algo_class in self.algorithms.items():
            print(f"    Running {algo_name}...", end='', flush=True)
            costs = []
            times = []
            metrics_list = []
            
            for run in range(num_runs):
                try:
                    random.seed(42 + run * 7)
                    np.random.seed(42 + run * 7)
                    algo = algo_class(instance_data, vehicle)
                    start_time = time.time()
                    solution, cost, extra = algo.solve()
                    elapsed = time.time() - start_time
                    
                    if solution:
                        if 'metrics' in extra and extra['metrics'] is not None:
                            metrics = extra['metrics']
                        else:
                            calc = ObjectiveCalculator(instance_data, vehicle)
                            metrics = calc.compute_all_metrics(solution)
                        
                        costs.append(cost)
                        times.append(elapsed)
                        metrics_list.append(metrics)
                except Exception as e:
                    print(f"\n      Error in run {run}: {e}")
                    continue
            
            if costs:
                avg_metrics = self._average_metrics(metrics_list)
                results['results'][algo_name] = {
                    # Economic Efficiency
                    'avg_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'best_cost': min(costs),
                    'worst_cost': max(costs),
                    'avg_total_distance': avg_metrics.total_distance,
                    'avg_fleet_size': avg_metrics.fleet_size,
                    'avg_fleet_utilization': avg_metrics.fleet_utilization,
                    'avg_volume_utilization': avg_metrics.volume_utilization,
                    
                    # Safety Compliance
                    'avg_ldd_violation_rate': avg_metrics.ldd_violation_rate,
                    'avg_stability': avg_metrics.stability_margin,
                    'avg_minimum_load_ratio': avg_metrics.minimum_load_ratio,
                    'avg_slosh_risk': avg_metrics.slosh_risk_score,
                    
                    # Computational Performance
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'avg_cpu_time': avg_metrics.cpu_time,
                    'avg_saca_calls': avg_metrics.saca_calls,
                    'avg_cache_hit_rate': avg_metrics.cache_hit_rate,
                    'avg_iterations_to_convergence': avg_metrics.iterations_to_convergence,
                    
                    # CG Metrics
                    'avg_cg_position': avg_metrics.cg_position_mean,
                    'avg_cg_compliance_rate': avg_metrics.cg_compliance_rate,
                    'avg_cg_deviation': avg_metrics.cg_deviation_from_optimum,
                    'avg_heeling_moment': avg_metrics.heeling_moment_estimate,
                    'avg_lateral_margin': avg_metrics.lateral_acceleration_margin
                }
                print(f" done (avg cost: {np.mean(costs):.2f})", flush=True)
            else:
                print(" failed", flush=True)
        
        return results
    
    def _average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        if not metrics_list:
            return PerformanceMetrics(
                total_distance=0,
                transportation_cost=0,
                fleet_size=0,
                fleet_utilization=0,
                volume_utilization=0,
                ldd_violation_rate=0,
                stability_margin=0,
                minimum_load_ratio=0,
                slosh_risk_score=0,
                cpu_time=0,
                saca_calls=0,
                cache_hit_rate=0,
                iterations_to_convergence=0
            )
        
        avg = PerformanceMetrics(
            total_distance=0,
            transportation_cost=0,
            fleet_size=0,
            fleet_utilization=0,
            volume_utilization=0,
            ldd_violation_rate=0,
            stability_margin=0,
            minimum_load_ratio=0,
            slosh_risk_score=0,
            cpu_time=0,
            saca_calls=0,
            cache_hit_rate=0,
            iterations_to_convergence=0
        )
        
        for m in metrics_list:
            avg.total_distance += m.total_distance
            avg.transportation_cost += m.transportation_cost
            avg.fleet_size += m.fleet_size
            avg.fleet_utilization += m.fleet_utilization
            avg.volume_utilization += m.volume_utilization
            avg.ldd_violation_rate += m.ldd_violation_rate
            avg.stability_margin += m.stability_margin
            avg.minimum_load_ratio += m.minimum_load_ratio
            avg.slosh_risk_score += m.slosh_risk_score
            avg.cpu_time += m.cpu_time
            avg.saca_calls += m.saca_calls
            avg.cache_hit_rate += m.cache_hit_rate
            avg.iterations_to_convergence += m.iterations_to_convergence
            avg.cg_position_mean += m.cg_position_mean
            avg.cg_position_std += m.cg_position_std
            avg.cg_position_min += m.cg_position_min
            avg.cg_position_max += m.cg_position_max
            avg.cg_compliance_rate += m.cg_compliance_rate
            avg.cg_deviation_from_optimum += m.cg_deviation_from_optimum
            avg.heeling_moment_estimate += m.heeling_moment_estimate
            avg.lateral_acceleration_margin += m.lateral_acceleration_margin
        
        n = len(metrics_list)
        if n > 0:
            for attr in ['total_distance', 'transportation_cost', 'fleet_size', 'fleet_utilization',
                        'volume_utilization', 'ldd_violation_rate', 'stability_margin', 
                        'minimum_load_ratio', 'slosh_risk_score', 'cpu_time', 'saca_calls',
                        'cache_hit_rate', 'iterations_to_convergence', 'cg_position_mean',
                        'cg_position_std', 'cg_position_min', 'cg_position_max',
                        'cg_compliance_rate', 'cg_deviation_from_optimum', 
                        'heeling_moment_estimate', 'lateral_acceleration_margin']:
                if hasattr(avg, attr):
                    setattr(avg, attr, getattr(avg, attr) / n)
        
        return avg
    
    def run_all_instances(self, max_instances: int = None) -> pd.DataFrame:
        instance_files = []
        for root, dirs, files in os.walk(self.benchmark_dir):
            for file in files:
                if file.endswith('.json'):
                    instance_files.append(os.path.join(root, file))
        
        instance_files.sort(key=lambda x: os.path.basename(x))
        
        if max_instances:
            instance_files = instance_files[:max_instances]
        
        print(f"\nPART 1: SOTA COMPARISON (6 Algorithms)")
        print(f"Found {len(instance_files)} instance files")
        print(f"Algorithms:")
        print("  1. IALNS (Proposed) - Adaptive weights + enhanced operators")
        print("  2. Standard_ALNS - Standard baseline (no adaptive)")
        print("  3. FDAHS - Feasibility-Driven Anytime Hybrid Search (2026)")
        print("  4. GCOF - Guided Collaborative Optimization Framework (2026)")
        print("  5. HSA - Hybrid Simulated Annealing (2023)")
        print("  6. ACO_2opt - Ant Colony Optimization with 2-Opt (2015)")
        print("-" * 60)
        
        all_results = []
        for i, filepath in enumerate(instance_files):
            print(f"\nProcessing {i+1}/{len(instance_files)}: {os.path.basename(filepath)}")
            try:
                result = self.run_single_instance(filepath, num_runs=3)
                all_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        df = self._results_to_dataframe(all_results)
        if not df.empty:
            csv_path = os.path.join(self.output_dir, 'sota_comparison_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nSOTA results saved to: {csv_path}")
        return df
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        rows = []
        for result in results:
            if not result['results']:
                continue
            base_row = {
                'instance': result['instance'],
                'n_customers': result['n_customers'],
                'n_compartments': result.get('n_compartments', 0),
                'layer': result.get('layer', 'Unknown'),
                'n_products': result.get('n_products', 3),
                'vehicle_type': result.get('vehicle_type', 'Multi-Compartment')
            }
            for algo, data in result['results'].items():
                row = base_row.copy()
                row['algorithm'] = algo
                for key, value in data.items():
                    row[key] = value
                rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# ABLATION COMPARISON
# ============================================================================

class AblationComparison:
    def __init__(self, benchmark_dir: str, output_dir: str):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        
        self.algorithms = {
            'IALNS': IALNSAlgorithm,
            'IALNS_NS': IALNS_NoStability,
            'IALNS_NA': IALNS_NoAdaptive,
            'IALNS_SEQ': IALNS_Sequential,
            'IALNS_NC': IALNS_NoCache,
            'IALNS_IMB': IALNS_NoImbalance,
        }
        
    def load_instance(self, filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_vehicle(self, instance: Dict) -> Vehicle:
        vehicle_data = instance['vehicle']
        compartments = [Compartment(
            id=comp['id'],
            capacity=comp['capacity'],
            position_x=comp['position_x'],
            position_y=comp['position_y'],
            load_density=1.0
        ) for comp in vehicle_data['compartments']]
        
        compartment_positions = {comp['id']: comp['position_x'] for comp in vehicle_data['compartments']}
        
        return Vehicle(
            id=0,
            compartments=compartments,
            max_payload=vehicle_data['max_payload'],
            front_axle_max=vehicle_data['front_axle_max'],
            rear_axle_max=vehicle_data['rear_axle_max'],
            wheelbase=vehicle_data['wheelbase'],
            unladen_weight=vehicle_data['unladen_weight'],
            max_volume=sum(c.capacity for c in compartments),
            steering_min_ratio=vehicle_data['steering_min_ratio'],
            driving_min_ratio=vehicle_data['driving_min_ratio'],
            product_densities=vehicle_data.get('product_densities', {}),
            fixed_cost=100.0,
            cost_per_km=1.0,
            front_axle_min=vehicle_data.get('front_axle_min', vehicle_data['front_axle_max'] * 0.1),
            rear_axle_min=vehicle_data.get('rear_axle_min', vehicle_data['rear_axle_max'] * 0.1),
            cg_min=vehicle_data.get('cg_min', 0.3 * vehicle_data['wheelbase']),
            cg_max=vehicle_data.get('cg_max', 0.7 * vehicle_data['wheelbase']),
            compartment_positions=compartment_positions,
            is_multi_compartment=True
        )
    
    def build_instance_data(self, instance: Dict) -> Dict:
        n = len(instance['customers'])
        coords = [[0, 0]] + [[c['x'], c['y']] for c in instance['customers']]
        dist_matrix = [[math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
                       for j in range(n + 1)] for i in range(n + 1)]
        
        return {
            'name': instance['name'],
            'customers': instance['customers'],
            'time_windows': instance['time_windows'],
            'distance_matrix': dist_matrix,
            'vehicle': instance['vehicle'],
            'n_products': instance.get('n_products', 3),
            'n_customers': len(instance['customers']),
            'n_compartments': instance.get('n_compartments', len(instance['vehicle']['compartments']))
        }
    
    def run_single_instance(self, instance_file: str, num_runs: int = 3) -> Dict:
        instance = self.load_instance(instance_file)
        vehicle = self.create_vehicle(instance)
        instance_data = self.build_instance_data(instance)
        
        results = {
            'instance': instance['name'],
            'n_customers': len(instance['customers']),
            'n_compartments': instance.get('n_compartments', len(instance['vehicle']['compartments'])),
            'layer': instance.get('layer', 'Unknown'),
            'n_products': instance.get('n_products', 3),
            'results': {}
        }
        
        print(f"\n  Testing on {instance['name']} ({len(instance['customers'])} customers, "
              f"{instance.get('n_products', 3)} commodities, {results['n_compartments']} compartments)")
        
        for algo_name, algo_class in self.algorithms.items():
            print(f"    Running {algo_name}...", end='', flush=True)
            costs = []
            times = []
            metrics_list = []
            
            for run in range(num_runs):
                try:
                    random.seed(42 + run * 13)
                    np.random.seed(42 + run * 13)
                    algo = algo_class(instance_data, vehicle)
                    start_time = time.time()
                    solution, cost, extra = algo.solve()
                    elapsed = time.time() - start_time
                    
                    if solution:
                        if 'metrics' in extra and extra['metrics'] is not None:
                            metrics = extra['metrics']
                        else:
                            calc = ObjectiveCalculator(instance_data, vehicle)
                            metrics = calc.compute_all_metrics(solution)
                        
                        costs.append(cost)
                        times.append(elapsed)
                        metrics_list.append(metrics)
                except Exception as e:
                    print(f"\n      Error in run {run}: {e}")
                    continue
            
            if costs:
                avg_metrics = self._average_metrics(metrics_list)
                results['results'][algo_name] = {
                    # Economic Efficiency
                    'avg_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'best_cost': min(costs),
                    'worst_cost': max(costs),
                    'avg_total_distance': avg_metrics.total_distance,
                    'avg_fleet_size': avg_metrics.fleet_size,
                    'avg_fleet_utilization': avg_metrics.fleet_utilization,
                    'avg_volume_utilization': avg_metrics.volume_utilization,
                    
                    # Safety Compliance
                    'avg_ldd_violation_rate': avg_metrics.ldd_violation_rate,
                    'avg_stability': avg_metrics.stability_margin,
                    'avg_minimum_load_ratio': avg_metrics.minimum_load_ratio,
                    'avg_slosh_risk': avg_metrics.slosh_risk_score,
                    
                    # Computational Performance
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'avg_cpu_time': avg_metrics.cpu_time,
                    'avg_saca_calls': avg_metrics.saca_calls,
                    'avg_cache_hit_rate': avg_metrics.cache_hit_rate,
                    'avg_iterations_to_convergence': avg_metrics.iterations_to_convergence,
                    
                    # CG Metrics
                    'avg_cg_position': avg_metrics.cg_position_mean,
                    'avg_cg_compliance_rate': avg_metrics.cg_compliance_rate,
                    'avg_cg_deviation': avg_metrics.cg_deviation_from_optimum,
                    'avg_heeling_moment': avg_metrics.heeling_moment_estimate,
                    'avg_lateral_margin': avg_metrics.lateral_acceleration_margin
                }
                print(f" done (avg cost: {np.mean(costs):.2f})", flush=True)
            else:
                print(" failed", flush=True)
        
        return results
    
    def _average_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        if not metrics_list:
            return PerformanceMetrics(
                total_distance=0,
                transportation_cost=0,
                fleet_size=0,
                fleet_utilization=0,
                volume_utilization=0,
                ldd_violation_rate=0,
                stability_margin=0,
                minimum_load_ratio=0,
                slosh_risk_score=0,
                cpu_time=0,
                saca_calls=0,
                cache_hit_rate=0,
                iterations_to_convergence=0
            )
        
        avg = PerformanceMetrics(
            total_distance=0,
            transportation_cost=0,
            fleet_size=0,
            fleet_utilization=0,
            volume_utilization=0,
            ldd_violation_rate=0,
            stability_margin=0,
            minimum_load_ratio=0,
            slosh_risk_score=0,
            cpu_time=0,
            saca_calls=0,
            cache_hit_rate=0,
            iterations_to_convergence=0
        )
        
        for m in metrics_list:
            avg.total_distance += m.total_distance
            avg.transportation_cost += m.transportation_cost
            avg.fleet_size += m.fleet_size
            avg.fleet_utilization += m.fleet_utilization
            avg.volume_utilization += m.volume_utilization
            avg.ldd_violation_rate += m.ldd_violation_rate
            avg.stability_margin += m.stability_margin
            avg.minimum_load_ratio += m.minimum_load_ratio
            avg.slosh_risk_score += m.slosh_risk_score
            avg.cpu_time += m.cpu_time
            avg.saca_calls += m.saca_calls
            avg.cache_hit_rate += m.cache_hit_rate
            avg.iterations_to_convergence += m.iterations_to_convergence
            avg.cg_position_mean += m.cg_position_mean
            avg.cg_position_std += m.cg_position_std
            avg.cg_position_min += m.cg_position_min
            avg.cg_position_max += m.cg_position_max
            avg.cg_compliance_rate += m.cg_compliance_rate
            avg.cg_deviation_from_optimum += m.cg_deviation_from_optimum
            avg.heeling_moment_estimate += m.heeling_moment_estimate
            avg.lateral_acceleration_margin += m.lateral_acceleration_margin
        
        n = len(metrics_list)
        if n > 0:
            for attr in ['total_distance', 'transportation_cost', 'fleet_size', 'fleet_utilization',
                        'volume_utilization', 'ldd_violation_rate', 'stability_margin', 
                        'minimum_load_ratio', 'slosh_risk_score', 'cpu_time', 'saca_calls',
                        'cache_hit_rate', 'iterations_to_convergence', 'cg_position_mean',
                        'cg_position_std', 'cg_position_min', 'cg_position_max',
                        'cg_compliance_rate', 'cg_deviation_from_optimum', 
                        'heeling_moment_estimate', 'lateral_acceleration_margin']:
                if hasattr(avg, attr):
                    setattr(avg, attr, getattr(avg, attr) / n)
        
        return avg
    
    def run_all_instances(self, max_instances: int = None) -> pd.DataFrame:
        instance_files = []
        for root, dirs, files in os.walk(self.benchmark_dir):
            for file in files:
                if file.endswith('.json'):
                    instance_files.append(os.path.join(root, file))
        
        instance_files.sort(key=lambda x: os.path.basename(x))
        
        if max_instances:
            instance_files = instance_files[:max_instances]
        
        print(f"\nPART 2: ABLATION STUDY (6 Variants)")
        print(f"Found {len(instance_files)} instance files")
        print(f"Ablation Variants:")
        print("  1. IALNS (Full)")
        print("  2. IALNS_NS (No Stability verification)")
        print("  3. IALNS_NA (No Adaptive operator selection)")
        print("  4. IALNS_SEQ (Sequential optimization)")
        print("  5. IALNS_NC (No Cache)")
        print("  6. IALNS_IMB (No Imbalance destroy operator)")
        print("-" * 60)
        
        all_results = []
        for i, filepath in enumerate(instance_files):
            print(f"\nProcessing {i+1}/{len(instance_files)}: {os.path.basename(filepath)}")
            try:
                result = self.run_single_instance(filepath, num_runs=3)
                all_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        df = self._results_to_dataframe(all_results)
        if not df.empty:
            csv_path = os.path.join(self.output_dir, 'ablation_study_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nAblation results saved to: {csv_path}")
        return df
    
    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        rows = []
        for result in results:
            if not result['results']:
                continue
            base_row = {
                'instance': result['instance'],
                'n_customers': result['n_customers'],
                'n_compartments': result.get('n_compartments', 0),
                'layer': result.get('layer', 'Unknown'),
                'n_products': result.get('n_products', 3)
            }
            for algo, data in result['results'].items():
                row = base_row.copy()
                row['algorithm'] = algo
                for key, value in data.items():
                    row[key] = value
                rows.append(row)
        return pd.DataFrame(rows)


# ============================================================================
# MULTI-COMPARTMENT COMPARISON
# ============================================================================

def run_multicomp_comparison(sota_df: pd.DataFrame, benchmark_dir: str, output_dir: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("MULTI-COMPARTMENT vs SINGLE-COMPARTMENT COMPARISON")
    print("=" * 80)
    
    # Check if DataFrame is empty or missing required columns
    if sota_df.empty:
        print("SOTA DataFrame is empty. Skipping multi-compartment comparison.")
        return pd.DataFrame()
    
    if 'algorithm' not in sota_df.columns:
        print("'algorithm' column not found in SOTA DataFrame. Skipping multi-compartment comparison.")
        return pd.DataFrame()
    
    ialns_df = sota_df[sota_df['algorithm'] == 'IALNS']
    
    if ialns_df.empty:
        print("No IALNS data found for comparison")
        return pd.DataFrame()
    
    results = []
    compartment_stats = defaultdict(lambda: {'count': 0, 'cost_imp': [], 'fleet_red': [], 'util_imp': []})
    
    for idx, row in ialns_df.iterrows():
        instance_name = row['instance']
        n_customers = row['n_customers']
        layer = row.get('layer', 'Unknown')
        n_compartments = row.get('n_compartments', 3)
        
        instance_file = None
        for root, dirs, files in os.walk(benchmark_dir):
            if f"{instance_name}.json" in files:
                instance_file = os.path.join(root, f"{instance_name}.json")
                break
        
        if instance_file is None:
            continue
        
        try:
            with open(instance_file, 'r') as f:
                instance = json.load(f)
            
            actual_n_compartments = instance.get('n_compartments', len(instance['vehicle']['compartments']))
            
            vehicle = create_single_compartment_vehicle(instance)
            instance_data = build_instance_data(instance)
            
            random.seed(42)
            np.random.seed(42)
            algo = IALNSAlgorithm(instance_data, vehicle)
            solution, cost, extra = algo.solve()
            metrics = extra['metrics'] if 'metrics' in extra else None
            
            if metrics is None:
                calc = ObjectiveCalculator(instance_data, vehicle)
                metrics = calc.compute_all_metrics(solution)
            
            cost_improvement = (metrics.transportation_cost - row['avg_cost']) / metrics.transportation_cost * 100 if metrics.transportation_cost > 0 else 0
            fleet_reduction = (metrics.fleet_size - row.get('avg_fleet_size', 0)) / max(1, metrics.fleet_size) * 100
            util_improvement = (row['avg_volume_utilization'] - metrics.volume_utilization) / max(1, row['avg_volume_utilization']) * 100
            
            results.append({
                'instance': instance_name,
                'n_customers': n_customers,
                'layer': layer,
                'n_compartments': actual_n_compartments,
                'n_products': instance.get('n_products', 3),
                'multi_cost': row['avg_cost'],
                'single_cost': metrics.transportation_cost,
                'cost_improvement': cost_improvement,
                'multi_fleet': row.get('avg_fleet_size', 0),
                'single_fleet': metrics.fleet_size,
                'fleet_reduction': fleet_reduction,
                'multi_util': row['avg_volume_utilization'],
                'single_util': metrics.volume_utilization,
                'util_improvement': util_improvement,
                'multi_violation': row.get('avg_ldd_violation_rate', 0),
                'single_violation': metrics.ldd_violation_rate,
                'multi_cg_compliance': row.get('avg_cg_compliance_rate', 100),
                'single_cg_compliance': metrics.cg_compliance_rate
            })
            
            compartment_stats[actual_n_compartments]['count'] += 1
            compartment_stats[actual_n_compartments]['cost_imp'].append(cost_improvement)
            compartment_stats[actual_n_compartments]['fleet_red'].append(fleet_reduction)
            compartment_stats[actual_n_compartments]['util_imp'].append(util_improvement)
            
            print(f"  Processed {instance_name}: {actual_n_compartments} compartments, Cost improvement {cost_improvement:.1f}%")
        except Exception as e:
            print(f"  Error processing {instance_name}: {e}")
            continue
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        csv_path = os.path.join(output_dir, 'part4_statistics', 'multicomp_benefits.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nMulti-compartment comparison results saved to: {csv_path}")
        
        print("\n" + "=" * 60)
        print("COMPARTMENT-LEVEL ANALYSIS")
        print("=" * 60)
        
        for n_comp, stats in sorted(compartment_stats.items()):
            if stats['count'] > 0:
                print(f"\n{n_comp} Compartments ({stats['count']} instances):")
                print(f"  Avg Cost Improvement: {np.mean(stats['cost_imp']):.1f}% (±{np.std(stats['cost_imp']):.1f})")
                print(f"  Avg Fleet Reduction: {np.mean(stats['fleet_red']):.1f}% (±{np.std(stats['fleet_red']):.1f})")
                print(f"  Avg Utilization Improvement: {np.mean(stats['util_imp']):.1f}% (±{np.std(stats['util_imp']):.1f})")
        
        comp_summary = []
        for n_comp, stats in sorted(compartment_stats.items()):
            if stats['count'] > 0:
                comp_summary.append({
                    'n_compartments': n_comp,
                    'count': stats['count'],
                    'avg_cost_improvement': np.mean(stats['cost_imp']),
                    'std_cost_improvement': np.std(stats['cost_imp']),
                    'avg_fleet_reduction': np.mean(stats['fleet_red']),
                    'std_fleet_reduction': np.std(stats['fleet_red']),
                    'avg_util_improvement': np.mean(stats['util_imp']),
                    'std_util_improvement': np.std(stats['util_imp'])
                })
        
        comp_df = pd.DataFrame(comp_summary)
        if not comp_df.empty:
            comp_csv_path = os.path.join(output_dir, 'part4_statistics', 'multicomp_benefits_by_compartments.csv')
            comp_df.to_csv(comp_csv_path, index=False)
            print(f"\nCompartment-level summary saved to: {comp_csv_path}")
    
    return df

def build_instance_data(instance: Dict) -> Dict:
    n = len(instance['customers'])
    coords = [[0, 0]] + [[c['x'], c['y']] for c in instance['customers']]
    dist_matrix = [[math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) 
                   for j in range(n + 1)] for i in range(n + 1)]
    
    return {
        'name': instance['name'],
        'customers': instance['customers'],
        'time_windows': instance['time_windows'],
        'distance_matrix': dist_matrix,
        'vehicle': instance['vehicle'],
        'n_products': instance.get('n_products', 3),
        'n_customers': len(instance['customers']),
        'n_compartments': instance.get('n_compartments', len(instance['vehicle']['compartments']))
    }


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_all_figures(sota_df: pd.DataFrame, ablation_df: pd.DataFrame, 
                        multicomp_df: pd.DataFrame, output_dir: str):
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    
    # SOTA Figures
    if not sota_df.empty:
        fig1_sota_cost(sota_df, output_dir)
        fig2_sota_stability(sota_df, output_dir)
        fig3_sota_radar(sota_df, output_dir)
        fig4_sota_scalability(sota_df, output_dir)
    else:
        print("  Skipping SOTA figures: No SOTA data")
    
    # Ablation Figures
    if not ablation_df.empty:
        fig5_ablation_cost(ablation_df, output_dir)
        fig6_ablation_stability(ablation_df, output_dir)
        fig7_ablation_bar(ablation_df, output_dir)
    else:
        print("  Skipping Ablation figures: No ablation data")
    
    # Statistics Figures
    if not sota_df.empty:
        fig8_convergence_analysis(sota_df, output_dir)
        fig9_correlation(sota_df, output_dir)
        fig10_layer_analysis(sota_df, ablation_df, output_dir)
        fig11_statistical_significance(sota_df, output_dir)
        fig12_cg_validation(sota_df, output_dir)
        fig13_cg_compliance(sota_df, output_dir)
    else:
        print("  Skipping Statistics figures: No SOTA data")
    
    # Multi-compartment comparison figure
    if not multicomp_df.empty:
        fig14_multicomp_comparison(multicomp_df, output_dir)
        fig15_compartment_analysis(multicomp_df, output_dir)
    else:
        print("  Skipping Multi-compartment figures: No comparison data")
    
    print("\nAll figures generated successfully!")


def fig1_sota_cost(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    algo_order = ['IALNS', 'Standard_ALNS', 'FDAHS', 'GCOF', 'HSA', 'ACO_2opt']
    
    for algo in algo_order:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue
        grouped = algo_df.groupby('n_customers')['avg_cost'].agg(['mean', 'std']).reset_index()
        linewidth = 3 if algo == 'IALNS' else 2
        markersize = 10 if algo == 'IALNS' else 7
        ax.errorbar(grouped['n_customers'], grouped['mean'], yerr=grouped['std'],
                   marker=MARKERS.get(algo, 'o'), color=COLORS_SOTA.get(algo, '#333333'),
                   label=SOTA_DISPLAY_NAMES.get(algo, algo), markersize=markersize,
                   capsize=4, elinewidth=1.5, linewidth=linewidth)
    
    ax.set_xlabel('Number of Customers', fontsize=12)
    ax.set_ylabel('Average Transportation Cost ($)', fontsize=12)
    ax.set_title('SOTA Comparison: Cost vs Instance Size', fontsize=14)
    ax.legend(loc='upper left', frameon=True, fancybox=False, framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part1_sota', 'Figure_1_SOTA_Cost_Comparison.pdf'), dpi=300)
    plt.close()
    print("  Generated: Figure_1_SOTA_Cost_Comparison")


def fig2_sota_stability(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    algo_order = ['IALNS', 'Standard_ALNS', 'FDAHS', 'GCOF', 'HSA', 'ACO_2opt']
    
    for algo in algo_order:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue
        grouped = algo_df.groupby('n_customers')['avg_stability'].agg(['mean', 'std']).reset_index()
        linewidth = 3 if algo == 'IALNS' else 2
        markersize = 10 if algo == 'IALNS' else 7
        ax.errorbar(grouped['n_customers'], grouped['mean'], yerr=grouped['std'],
                   marker=MARKERS.get(algo, 'o'), color=COLORS_SOTA.get(algo, '#333333'),
                   label=SOTA_DISPLAY_NAMES.get(algo, algo), markersize=markersize,
                   capsize=4, elinewidth=1.5, linewidth=linewidth)
    
    ax.set_xlabel('Number of Customers', fontsize=12)
    ax.set_ylabel('Stability Margin (%)', fontsize=12)
    ax.set_title('SOTA Comparison: Stability vs Instance Size', fontsize=14)
    ax.legend(loc='upper right', frameon=True, fancybox=False, framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=1, label='Safety Threshold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part1_sota', 'Figure_2_SOTA_Stability_Comparison.pdf'), dpi=300)
    plt.close()
    print("  Generated: Figure_2_SOTA_Stability_Comparison")


def fig3_sota_radar(df: pd.DataFrame, output_dir: str):
    from math import pi
    algo_order = ['IALNS', 'Standard_ALNS', 'FDAHS', 'GCOF', 'HSA', 'ACO_2opt']
    metrics = ['avg_cost', 'avg_ldd_violation_rate', 'avg_stability', 'avg_volume_utilization', 'avg_cg_compliance_rate']
    metric_labels = ['Cost (↓)', 'LDD Violation (↓)', 'Stability (↑)', 'Utilization (↑)', 'CG Compliance (↑)']
    
    algo_data = {}
    for algo in algo_order:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue
        avg_metrics = {}
        for m in metrics:
            if m in algo_df.columns:
                avg_metrics[m] = algo_df[m].mean()
            else:
                avg_metrics[m] = 0
        algo_data[algo] = avg_metrics
    
    if not algo_data:
        return
    
    all_values = {m: [algo_data[a][m] for a in algo_data if m in algo_data[a]] for m in metrics}
    for m in metrics:
        min_val = min(all_values[m]) if all_values[m] else 1
        max_val = max(all_values[m]) if all_values[m] else 1
        range_val = max_val - min_val or 1
        for a in algo_data:
            if m == 'avg_cost' or m == 'avg_ldd_violation_rate':
                algo_data[a][m] = 1 - (algo_data[a][m] - min_val) / range_val
            else:
                algo_data[a][m] = (algo_data[a][m] - min_val) / range_val
    
    n_metrics = len(metrics)
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for algo in algo_order:
        if algo not in algo_data:
            continue
        values = [algo_data[algo][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='-',
                color=COLORS_SOTA.get(algo, '#333333'), label=SOTA_DISPLAY_NAMES.get(algo, algo))
        ax.fill(angles, values, alpha=0.1, color=COLORS_SOTA.get(algo, '#333333'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, fancybox=False, fontsize=8)
    ax.set_title('SOTA Performance Radar Chart', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part1_sota', 'Figure_3_SOTA_Radar.pdf'), dpi=300)
    plt.close()
    print("  Generated: Figure_3_SOTA_Radar")


def fig4_sota_scalability(df: pd.DataFrame, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    algo_order = ['IALNS', 'Standard_ALNS', 'FDAHS', 'GCOF', 'HSA', 'ACO_2opt']
    
    for algo in algo_order:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue
        grouped = algo_df.groupby('n_customers')['avg_time'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped['n_customers'], grouped['mean'], yerr=grouped['std'],
                   marker=MARKERS.get(algo, 'o'), color=COLORS_SOTA.get(algo, '#333333'),
                   label=SOTA_DISPLAY_NAMES.get(algo, algo), markersize=6,
                   capsize=3, elinewidth=1, linewidth=1.5)
    
    ax1.set_xlabel('Number of Customers', fontsize=12)
    ax1.set_ylabel('CPU Time (seconds)', fontsize=12)
    ax1.set_title('Runtime Scaling', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fancybox=False, framealpha=0.9, fontsize=7)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    data_to_plot = []
    labels = []
    for algo in algo_order:
        algo_df = df[df['algorithm'] == algo]
        if algo_df.empty:
            continue
        data_to_plot.append(algo_df['avg_cost'].values)
        labels.append(SOTA_DISPLAY_NAMES.get(algo, algo))
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [COLORS_SOTA.get(a, '#333333') for a in algo_order if a in df['algorithm'].unique()]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xlabel('Algorithm', fontsize=12)
        ax2.set_ylabel('Transportation Cost ($)', fontsize=12)
        ax2.set_title('Cost Distribution by Algorithm', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'part1_sota', 'Figure_4_SOTA_Scalability.pdf'), dpi=300)
    plt.close()
    print("  Generated: Figure_4_SOTA_Scalability")


# [The remaining figure functions fig5-fig15 are included in the full script]
# Due to length constraints, they are available in the previous complete version.


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_statistical_tests(sota_df: pd.DataFrame, output_dir: str):
    print("\n" + "=" * 80)
    print("RUNNING STATISTICAL TESTS")
    print("=" * 80)
    
    results = []
    ialns_df = sota_df[sota_df['algorithm'] == 'IALNS']
    if ialns_df.empty:
        print("No IALNS data found")
        return
    
    ialns_costs = ialns_df['avg_cost'].values
    ialns_cg_compliance = ialns_df['avg_cg_compliance_rate'].values if 'avg_cg_compliance_rate' in ialns_df.columns else None
    
    for algo in ['Standard_ALNS', 'FDAHS', 'GCOF', 'HSA', 'ACO_2opt']:
        algo_df = sota_df[sota_df['algorithm'] == algo]
        if algo_df.empty:
            continue
        
        algo_costs = algo_df['avg_cost'].values
        algo_cg_compliance = algo_df['avg_cg_compliance_rate'].values if 'avg_cg_compliance_rate' in algo_df.columns else None
        
        if len(ialns_costs) > 0 and len(algo_costs) > 0:
            try:
                _, wilcoxon_p = wilcoxon(ialns_costs, algo_costs)
            except:
                wilcoxon_p = 1.0
            try:
                _, mannwhitney_p = mannwhitneyu(ialns_costs, algo_costs)
            except:
                mannwhitney_p = 1.0
            
            mean1, mean2 = np.mean(ialns_costs), np.mean(algo_costs)
            std1, std2 = np.std(ialns_costs, ddof=1), np.std(algo_costs, ddof=1)
            pooled_std = np.sqrt(((len(ialns_costs) - 1) * std1**2 + (len(algo_costs) - 1) * std2**2) / 
                                (len(ialns_costs) + len(algo_costs) - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            improvement = (mean2 - mean1) / mean2 * 100 if mean2 > 0 else 0
            
            cg_improvement = 0
            if ialns_cg_compliance is not None and algo_cg_compliance is not None:
                cg_improvement = np.mean(ialns_cg_compliance) - np.mean(algo_cg_compliance)
            
            results.append({
                'Algorithm': SOTA_DISPLAY_NAMES.get(algo, algo),
                'IALNS Mean': mean1,
                'Competitor Mean': mean2,
                'Improvement (%)': improvement,
                'CG Compliance Improvement (%)': cg_improvement,
                'Wilcoxon p-value': wilcoxon_p,
                'Mann-Whitney p-value': mannwhitney_p,
                'Cohen_d': cohens_d,
                'Significant (p < 0.05)': mannwhitney_p < 0.05
            })
    
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'part4_statistics', 'statistical_tests_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Statistical tests results saved to: {csv_path}")
    
    print("\nStatistical Test Results:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"\n{row['Algorithm']}:")
        print(f"  Cost Improvement vs IALNS: {row['Improvement (%)']:.1f}%")
        print(f"  CG Compliance Improvement: {row['CG Compliance Improvement (%)']:.1f}%")
        print(f"  Wilcoxon p-value: {row['Wilcoxon p-value']:.4f}")
        print(f"  Mann-Whitney p-value: {row['Mann-Whitney p-value']:.4f}")
        print(f"  Cohen's d: {row['Cohen_d']:.3f}")
        print(f"  Statistically significant: {row['Significant (p < 0.05)']}")
    
    return results_df


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def generate_summary_statistics(sota_df: pd.DataFrame, ablation_df: pd.DataFrame, 
                               multicomp_df: pd.DataFrame, output_dir: str):
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_dir = os.path.join(output_dir, 'part4_statistics')
    os.makedirs(summary_dir, exist_ok=True)
    
    if not sota_df.empty:
        sota_metrics = ['avg_cost', 'avg_ldd_violation_rate', 'avg_stability', 'avg_time',
                       'avg_fleet_utilization', 'avg_volume_utilization', 'avg_minimum_load_ratio',
                       'avg_slosh_risk', 'avg_cg_compliance_rate']
        sota_summary = sota_df.groupby('algorithm')[sota_metrics].agg(['mean', 'std']).round(2)
        sota_summary.to_csv(os.path.join(summary_dir, 'sota_summary.csv'))
        print("  Generated: sota_summary.csv")
        
        if 'avg_cg_position' in sota_df.columns:
            cg_summary = sota_df.groupby('algorithm')[['avg_cg_position', 'avg_cg_compliance_rate', 'avg_cg_deviation']].mean().round(2)
            cg_summary.to_csv(os.path.join(summary_dir, 'cg_summary.csv'))
            print("  Generated: cg_summary.csv")
        
        if 'layer' in sota_df.columns:
            layer_summary = sota_df.groupby(['algorithm', 'layer']).agg({
                'avg_cost': 'mean', 'avg_ldd_violation_rate': 'mean',
                'avg_stability': 'mean', 'avg_cg_compliance_rate': 'mean'
            }).round(2)
            layer_summary.to_csv(os.path.join(summary_dir, 'sota_layer_summary.csv'))
            print("  Generated: sota_layer_summary.csv")
    
    if not ablation_df.empty:
        ablation_metrics = ['avg_cost', 'avg_ldd_violation_rate', 'avg_stability', 'avg_time',
                          'avg_fleet_utilization', 'avg_volume_utilization']
        ablation_summary = ablation_df.groupby('algorithm')[ablation_metrics].agg(['mean', 'std']).round(2)
        ablation_summary.to_csv(os.path.join(summary_dir, 'ablation_summary.csv'))
        print("  Generated: ablation_summary.csv")
    
    if not multicomp_df.empty:
        multicomp_summary = multicomp_df[['multi_cost', 'single_cost', 'cost_improvement',
                                         'multi_fleet', 'single_fleet', 'fleet_reduction',
                                         'multi_util', 'single_util', 'util_improvement']].mean().round(2)
        multicomp_summary.to_csv(os.path.join(summary_dir, 'multicomp_summary.csv'))
        print("  Generated: multicomp_summary.csv")
        
        if 'n_compartments' in multicomp_df.columns:
            comp_summary = multicomp_df.groupby('n_compartments').agg({
                'cost_improvement': ['mean', 'std', 'count'],
                'fleet_reduction': ['mean', 'std'],
                'util_improvement': ['mean', 'std']
            }).round(2)
            comp_summary.to_csv(os.path.join(summary_dir, 'multicomp_compartment_summary.csv'))
            print("  Generated: multicomp_compartment_summary.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("DSC-CTRP EXPERIMENTAL COMPARISON - FINAL VERSION WITH ALL METRICS")
    print("=" * 80)
    print("\nALGORITHMS COMPARED:")
    print("  1. IALNS (Proposed) - Adaptive weights + enhanced operators")
    print("  2. Standard_ALNS - Standard ALNS baseline (no adaptive)")
    print("  3. FDAHS - Feasibility-Driven Anytime Hybrid Search (2026)")
    print("  4. GCOF - Guided Collaborative Optimization Framework (2026)")
    print("  5. HSA - Hybrid Simulated Annealing (2023)")
    print("  6. ACO_2opt - Ant Colony Optimization with 2-Opt (2015)")
    print("\nABLATION VARIANTS (6 total):")
    print("  1. IALNS (Full)")
    print("  2. IALNS_NS (No Stability verification)")
    print("  3. IALNS_NA (No Adaptive operator selection)")
    print("  4. IALNS_SEQ (Sequential optimization)")
    print("  5. IALNS_NC (No Cache)")
    print("  6. IALNS_IMB (No Imbalance destroy operator)")
    print("\nPERFORMANCE METRICS:")
    print("  Economic Efficiency: Total Distance, Transportation Cost, Fleet Size,")
    print("                      Fleet Utilization, Volume Utilization")
    print("  Safety Compliance: LDD Violation Rate, Stability Margin,")
    print("                     Minimum Load Ratio, Slosh Risk Score, CG Compliance")
    print("  Computational: CPU Time, SACA Calls, Cache Hit Rate, Iterations to Convergence")
    print("\nSENSITIVITY ANALYSIS:")
    print("  λ_H (Handling Cost): 0.05, 0.075, 0.10, 0.125, 0.15 (Baseline: 0.10)")
    print("  λ_W (Overload Penalty): 0.50, 0.75, 1.00, 1.25, 1.50 (Baseline: 1.00)")
    print("  λ_S (Stability Penalty): 0.25, 0.375, 0.50, 0.625, 0.75 (Baseline: 0.50)")
    print("\nBENCHMARK INSTANCES (281 total):")
    print("  - Paixão Layer (40 instances): S_1--S_10 (3 customers, 1 comp),")
    print("    M_1--M_10 (5 customers, 2 comp), L_1--L_10 (7 customers, 3 comp),")
    print("    EL_1--EL_10 (9 customers, 4 comp)")
    print("  - Mirzaei/Muyldermans Layer (241 instances): vrpnc1--vrpnc241 (10--100 customers)")
    print("=" * 80)
    
    print(f"\nBenchmark Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not os.path.exists(BASE_DIR):
        print(f"\nWARNING: Benchmark directory not found: {BASE_DIR}")
        print("Creating benchmark directory and generating all 281 instances...")
        os.makedirs(BASE_DIR, exist_ok=True)
        generate_benchmark_instances(BASE_DIR, num_instances=281)
    else:
        json_files = []
        for root, dirs, files in os.walk(BASE_DIR):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        if len(json_files) == 0:
            print(f"\nWARNING: No JSON instance files found in: {BASE_DIR}")
            print("Generating all 281 benchmark instances...")
            generate_benchmark_instances(BASE_DIR, num_instances=281)
        else:
            print(f"\nFound {len(json_files)} JSON instance files")
            if len(json_files) < 281:
                print(f"Only {len(json_files)} instances found. Generating remaining instances...")
                generate_benchmark_instances(BASE_DIR, num_instances=281)
    
    # PART 1: SOTA
    print("\n" + "=" * 80)
    print("PART 1: STATE-OF-THE-ART COMPARISON (6 Algorithms)")
    print("=" * 80)
    sota_exp = SOTAComparison(BASE_DIR, OUTPUT_DIR)
    sota_df = sota_exp.run_all_instances(max_instances=50)
    
    # PART 2: Ablation
    print("\n" + "=" * 80)
    print("PART 2: ABLATION STUDY (6 Variants)")
    print("=" * 80)
    ablation_exp = AblationComparison(BASE_DIR, OUTPUT_DIR)
    ablation_df = ablation_exp.run_all_instances(max_instances=50)
    
    # PART 3: Sensitivity Analysis
    print("\n" + "=" * 80)
    print("PART 3: SENSITIVITY ANALYSIS")
    print("=" * 80)
    sensitivity_analyzer = SensitivityAnalyzer(BASE_DIR, OUTPUT_DIR)
    sensitivity_df = sensitivity_analyzer.run_sensitivity_analysis(max_instances=10, num_runs=3)
    
    # PART 4: Multi-compartment comparison
    multicomp_df = run_multicomp_comparison(sota_df, BASE_DIR, OUTPUT_DIR)
    
    # PART 5: Statistical Tests
    if not sota_df.empty:
        stats_results = run_statistical_tests(sota_df, OUTPUT_DIR)
    
    # PART 6: Generate All Figures
    generate_all_figures(sota_df, ablation_df, multicomp_df, OUTPUT_DIR)
    
    # PART 7: Summary Statistics
    generate_summary_statistics(sota_df, ablation_df, multicomp_df, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
