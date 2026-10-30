# TopKSelect Ascend backend

The `ascend` target implements `TopKSelectFwdOp` using the Ascend backend in the
main TileLang repository. It does not use the separate tilelang-ascend repository,
Triton, PyTorch TopK, or an external Ascend C TopK implementation.

## Usage

The package entry point registers the backend on installed distributions. A source
checkout can register it explicitly:

```python
import torch
import torch_npu
import tileops.ascend
from tileops.attention import TopKSelectFwdOp

scores = torch.randn(1, 2, 4096, 1, device="npu", dtype=torch.float32)
starts = torch.zeros(1, 2, device="npu", dtype=torch.int32)
ends = torch.full_like(starts, 4096)
indices = TopKSelectFwdOp(1024, target="ascend")(scores, starts, ends)
```

The experimental environment uses mainline TileLang 0.1.15, PyTorch/torch_npu 2.13
and the private CANN 9.3 weekly installation. The repository's general dependency
pin remains TileLang 0.1.12; dependency packaging for the Ascend environment is not
yet resolved. Do not reinstall dependencies from the general pin into this environment.

## Semantics and implementation

- Scores: contiguous FP32 `[B, S, L, G]` after framework staging.
- Windows: int32 `[B, S]`, selecting `[start, end)` separately for each group.
- Output: newly allocated contiguous int32 `[B, S, G, K]`, absolute indices,
  with `L` for unused slots. Slot order is unspecified.
- NaNs rank above positive infinity; all NaN payloads tie; signed zeros tie;
  equal scores select lower absolute indices first.
- Four exact 8-bit Radix passes, 2048-entry double-buffered boundary candidate
  cache, and a full-window rescan on cache overflow. K is not limited by cache size.
- Each worker uses 128 SIMT threads and reuses its UB while processing rows.
  The initial implementation caps scheduling at 32 workers; this is a conservative
  scheduling choice, not a statement about physical Vector core count.
- Explicit UB allocations total 34,344 bytes before alignment, compiler temporaries
  and runtime reservation: histogram 1,028; candidates 32,768; tie prefix 512;
  state 28; key prefix 8. There is no full-score transpose or global candidate scratch.

No performance threshold has been demonstrated. Single-row parallelism, the serial
256-bucket scan and the serial tie-prefix scan are known optimization opportunities.
The implementation currently targets Ascend 950 devices.

## Validation

```bash
python -m pytest tests/ops/test_topk_select_ascend.py --tileops-device=npu:0 -v
```

The tests compare index sets to a CPU reference with exact FP32 bit keys, cover
short/reversed windows, multiple groups, K greater than 2048, special values,
non-contiguous inputs and repeated calls. The compile-boundary test uses
`torch.compile(backend="eager", fullgraph=True)`; it does not establish support
for the default Inductor backend or NPU graph replay.
