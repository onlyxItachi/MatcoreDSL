"""End-to-end Family C fusion tests: softmax(Q @ K.T) @ V on GPU."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matcore as mc


@mc.fused
def _attention(Q, K, V):
    return mc.softmax(Q @ K.T) @ V


@mc.fused
def _attention_swapped_qk(K, Q, V):
    return mc.softmax(Q @ K.T) @ V


@mc.fused
def _attention_score_cached_dtile64(Q, K, V):
    return mc.softmax(Q @ K.T) @ V


def _softmax_ref(scores):
    scores = scores.astype(np.float64)
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True)


def _attention_ref(Q, K, V, score_scale=1.0):
    scores = (Q.astype(np.float64) @ K.astype(np.float64).T) * score_scale
    return (_softmax_ref(scores) @ V.astype(np.float64)).astype(np.float32)


def _make_inputs(M, N, qk_width, v_width, seed):
    rng = np.random.default_rng(seed)
    Q = rng.normal(0.0, 0.65, size=(M, qk_width)).astype(np.float32)
    K = rng.normal(0.0, 0.65, size=(N, qk_width)).astype(np.float32)
    V = rng.normal(0.0, 0.65, size=(N, v_width)).astype(np.float32)
    return Q, K, V


def _check_close(name, result, expected, tol=2e-3):
    result = np.asarray(result)
    assert result.shape == expected.shape, (
        f"{name} shape mismatch: expected {expected.shape}, got {result.shape}"
    )
    err = float(np.max(np.abs(result.astype(np.float64) - expected.astype(np.float64))))
    print(f"  {name}: shape={result.shape} max_err={err:.6f}")
    assert err < tol, f"{name} max_err={err:.6f} >= tol={tol}"


def _assert_family_c_optimized_route(expected_dtile=None):
    """Assert the native stats expose the optimized one-launch Family C path."""
    try:
        native = importlib.import_module("matcore._matcore_native")
    except ImportError:
        print("  route stats unavailable")
        return

    get_stats = getattr(native, "get_compilation_stats", None)
    if not callable(get_stats):
        print("  route stats unavailable")
        return

    info = get_stats()
    if not info:
        print("  route stats unavailable")
        return

    route = str(info.get("route", ""))
    route_lower = route.lower()
    assert "fused" in route_lower, f"expected fused Family C route, got {route!r}"
    assert "fallback" not in route_lower, f"unexpected fallback route: {route!r}"

    launch_count = info.get("fusion_launch_count")
    if launch_count is not None:
        assert int(launch_count) == 1, f"expected one launch, got {launch_count}"

    strategy = info.get("family_c_strategy")
    if strategy:
        assert strategy in {
            "score_cached_dtile64",
            "score_cached_block_coop_dtile64",
        }, strategy

    dtile = info.get("family_c_dtile")
    if expected_dtile is not None and dtile is not None:
        assert int(dtile) == expected_dtile, f"expected Dtile {expected_dtile}, got {dtile}"

    print(
        "  route: "
        f"{route} strategy={strategy or 'unknown'} launches={launch_count or 'unknown'} "
        f"dtile={dtile or 'unknown'} [PASS]"
    )


def test_attention_uses_unscaled_softmax_scores():
    """mc.softmax(Q @ K.T) means unscaled softmax, not scaled-dot-product attention."""
    M, N, D = 16, 16, 8
    Q, K, V = _make_inputs(M, N, D, D, seed=1001)

    expected = _attention_ref(Q, K, V)
    scaled = _attention_ref(Q, K, V, score_scale=1.0 / np.sqrt(D))
    scaled_delta = float(np.max(np.abs(expected - scaled)))
    assert scaled_delta > 5e-2, "test inputs must distinguish scaled vs unscaled softmax"

    result = _attention(Q, K, V)
    _check_close(f"attention unscaled ({M}x{N}x{D})", result, expected)


def test_attention_allows_m_not_equal_n():
    """Family C must not assume square score matrices."""
    M, N, D = 9, 14, 6
    Q, K, V = _make_inputs(M, N, D, D, seed=1002)

    result = _attention(Q, K, V)
    expected = _attention_ref(Q, K, V)
    _check_close(f"attention rectangular scores ({M}x{N}x{D})", result, expected)


def test_attention_allows_value_width_different_from_qk_width():
    """The output width comes from V.shape[1], not Q/K's feature width."""
    M, N, qk_width, v_width = 10, 12, 7, 5
    Q, K, V = _make_inputs(M, N, qk_width, v_width, seed=1003)

    result = _attention(Q, K, V)
    expected = _attention_ref(Q, K, V)
    _check_close(
        f"attention value width ({M}x{N}x{qk_width}->{v_width})",
        result,
        expected,
    )


def test_attention_allows_swapped_qk_argument_order():
    """Family C should bind tensors by graph values, not hard-coded Q,K,V arg order."""
    M, N, qk_width, v_width = 12, 12, 6, 4
    Q, K, V = _make_inputs(M, N, qk_width, v_width, seed=1004)

    result = _attention_swapped_qk(K, Q, V)
    expected = _attention_ref(Q, K, V)
    _check_close("attention swapped Q/K args", result, expected)


def test_attention_uses_score_cached_dtile64_route():
    """The optimized Family C route should widen V tiling to 64 columns."""
    M, N, D = 8, 12, 64
    Q, K, V = _make_inputs(M, N, D, D, seed=1006)

    result = _attention_score_cached_dtile64(Q, K, V)
    expected = _attention_ref(Q, K, V)
    _check_close(f"attention optimized route ({M}x{N}x{D})", result, expected)
    _assert_family_c_optimized_route(expected_dtile=64)


def test_attention_small():
    """Compatibility entry point for the original small Family C smoke test."""
    M, N, D = 16, 16, 8
    Q, K, V = _make_inputs(M, N, D, D, seed=1005)

    result = _attention(Q, K, V)
    expected = _attention_ref(Q, K, V)
    _check_close(f"attention small ({M}x{N}x{D})", result, expected)


if __name__ == "__main__":
    print("=== Family C End-to-End GPU Tests ===")
    test_attention_uses_unscaled_softmax_scores()
    test_attention_allows_m_not_equal_n()
    test_attention_allows_value_width_different_from_qk_width()
    test_attention_allows_swapped_qk_argument_order()
    test_attention_uses_score_cached_dtile64_route()
    test_attention_small()
    print("\nAll Family C GPU tests passed! ✓")
