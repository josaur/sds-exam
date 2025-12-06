#!/usr/bin/env python3
"""
Quick test script to verify the notebook functions work correctly.
Run this before executing the full notebook.
"""

import numpy as np
from scipy.optimize import minimize, basinhopping

print("Testing key functions...")
print("=" * 60)

# Test 1: SCM weight computation
print("\n1. Testing SCM weight computation...")
def compute_scm_weights(X0, X1, V=None):
    if V is None:
        V = np.eye(len(X1))

    n_controls = X0.shape[1]

    def objective(w):
        diff = X1 - X0 @ w
        return diff @ V @ diff

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_controls)]
    w0 = np.ones(n_controls) / n_controls

    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    return result.x

# Simple test case
X0_test = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]).T
X1_test = np.array([2, 3, 4])
w_test = compute_scm_weights(X0_test, X1_test)

print(f"   Weights: {w_test}")
print(f"   Sum of weights: {w_test.sum():.6f} (should be 1.0)")
print(f"   All non-negative: {np.all(w_test >= 0)}")
assert np.abs(w_test.sum() - 1.0) < 1e-6, "Weights should sum to 1"
assert np.all(w_test >= 0), "Weights should be non-negative"
print("   ✓ PASSED")

# Test 2: Basin-hopping optimization
print("\n2. Testing basin-hopping optimization...")
try:
    def simple_quadratic(x):
        return np.sum((x - 2)**2)

    result = basinhopping(
        simple_quadratic,
        np.array([0, 0]),
        minimizer_kwargs={"method": "L-BFGS-B"},
        niter=5,
        seed=12345
    )

    print(f"   Optimum found at: {result.x}")
    print(f"   Function value: {result.fun:.6f} (should be near 0)")
    assert result.fun < 0.1, "Should find minimum near 0"
    print("   ✓ PASSED")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    raise

# Test 3: Z-standardization
print("\n3. Testing z-standardization...")
def zstd(x, gelman=False):
    divisor = 2 * np.nanstd(x) if gelman else np.nanstd(x)
    return (x - np.nanmean(x)) / divisor

x_test = np.array([1, 2, 3, 4, 5])
z_test = zstd(x_test)
print(f"   Original: {x_test}")
print(f"   Standardized: {z_test}")
print(f"   Mean: {np.mean(z_test):.6f} (should be ~0)")
print(f"   Std: {np.std(z_test):.6f} (should be ~1)")
assert np.abs(np.mean(z_test)) < 1e-10, "Mean should be 0"
assert np.abs(np.std(z_test) - 1.0) < 1e-6, "Std should be 1"
print("   ✓ PASSED")

print("\n" + "=" * 60)
print("All tests passed! The notebook should run without errors.")
print("=" * 60)
