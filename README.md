
# FluxFormer v3

This version adds:
- **Exact reversible coupling** segment (closed-form inverse).
- **Per-head exact gauge attention** (already in v2).
- **2D separable Q-walk** (already in v2).
- **Top-K reversible MoE** with capacity control (already in v2).
- **Whitepaper** (WHITEPAPER.md).

## Run NumPy tests (no torch required)
```bash
python tests/test_numpy_qwalk_1d.py
python tests/test_numpy_qwalk_2d.py
python tests/test_numpy_perhead_bias_demo.py
python tests/test_numpy_rev_coupling.py
python tests/test_schedules.py
```

## Training (PyTorch, optional)
See v2 scripts in `examples/` (LM and images). Replace the FFN path with the **RevCouplingSegment** where you want exact reversibility.
