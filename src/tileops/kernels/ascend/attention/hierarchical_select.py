"""Exact partition-local TopK followed by TopK over the candidate union."""

import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target="ascend", out_idx=[])
def pack_shards(R: int, N: int, G: int, P: int, L: int, W: int):
    @T.prim_func
    def main(x: T.Tensor((R // G, N * G), "uint32"),
             starts: T.Tensor((R // G,), "int32"), ends: T.Tensor((R // G,), "int32"),
             packed: T.Tensor((R * P, L), "uint32"),
             lo: T.Tensor((R * P,), "int32"), hi: T.Tensor((R * P,), "int32")):
        with T.Kernel(W) as worker:
            data = T.alloc_shared((L,), "uint32")
            for step in T.serial(T.ceildiv(R * P, W)):
                task = worker + step * W
                if task < R * P:
                    row = task // P
                    base = (task % P) * L
                    count = T.max(0, T.min(L, N - base))
                    with T.SimtVF(threads=128):
                        for j in T.Parallel(L):
                            data[j] = T.uint32(0)
                    if G == 1:
                        if count > 0:
                            T.copy(x[row, base:base + count], data[:count])
                    else:
                        with T.SimtVF(threads=128):
                            for j in T.Parallel(L):
                                if j < count:
                                    data[j] = x[row // G, (base + j) * G + row % G]
                    T.copy(data, packed[task, :])
                    lo[task] = T.min(count, T.max(0, starts[row // G] - base))
                    hi[task] = T.min(count, T.max(0, ends[row // G] - base))
    return main


@tilelang.jit(target="ascend", out_idx=[])
def pack_candidates(R: int, N: int, G: int, P: int, L: int, J: int, W: int):
    C = P * J

    @T.prim_func
    def main(x: T.Tensor((R // G, N * G), "uint32"),
             lo: T.Tensor((R * P,), "int32"), hi: T.Tensor((R * P,), "int32"),
             ids: T.Tensor((R * P, J), "uint32"),
             scores: T.Tensor((R, C), "uint32"), mapping: T.Tensor((R, C), "uint32"),
             begin: T.Tensor((R,), "int32"), end: T.Tensor((R,), "int32")):
        with T.Kernel(W) as worker:
            selected = T.alloc_shared((J,), "uint32")
            for step in T.serial(T.ceildiv(R * P, W)):
                task = worker + step * W
                if task < R * P:
                    row = task // P
                    shard = task % P
                    offset = T.alloc_var("int32", init=0)
                    for earlier in T.serial(P):
                        if earlier < shard:
                            offset += T.min(J, hi[row * P + earlier] - lo[row * P + earlier])
                    count = T.min(J, hi[task] - lo[task])
                    T.copy(ids[task, :], selected)
                    with T.SimtVF(threads=128):
                        for j in T.Parallel(J):
                            if j < count:
                                original = shard * L + T.cast(selected[j], "int32")
                                scores[row, offset + j] = x[row // G, original * G + row % G]
                                mapping[row, offset + j] = T.cast(original, "uint32")
                    if shard == P - 1:
                        begin[row] = 0
                        end[row] = offset + count
    return main


@tilelang.jit(target="ascend", out_idx=[])
def remap_candidates(R: int, N: int, K: int, C: int, W: int):
    @T.prim_func
    def main(ids: T.Tensor((R, K), "uint32"), mapping: T.Tensor((R, C), "uint32"),
             out: T.Tensor((R, K), "uint32")):
        with T.Kernel(W) as worker:
            for step in T.serial(T.ceildiv(R, W)):
                row = worker + step * W
                if row < R:
                    with T.SimtVF(threads=128):
                        for j in T.Parallel(K):
                            index = ids[row, j]
                            if index < C:
                                out[row, j] = mapping[row, index]
                            else:
                                out[row, j] = T.cast(N, "uint32")
    return main
