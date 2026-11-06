"""Shape-parameterized resident SIMD radix selection; experimental dispatch candidate."""
import tilelang
import tilelang.ascend.language as T
import tilelang.ascend.language.simd


def simd_workspace_bytes(n, k, groups):
    np = (n + 63) // 64 * 64
    kp = (k + 63) // 64 * 64
    return 4 * (np + kp + (np * groups if groups > 1 else 0)) + 4096


@tilelang.jit(target="ascend", out_idx=[])
def topk_select_simd(R: int, N: int, K: int, G: int, W: int):
    assert 0 < K <= N < 65536 and G > 0 and 0 < W <= R
    assert simd_workspace_bytes(N, K, G) <= 96 * 1024
    NP = (N + 63) // 64 * 64
    KP = (K + 63) // 64 * 64

    @T.prim_func
    def main(bits: T.Tensor((R // G, N * G), "uint32"),
             starts: T.Tensor((R // G,), "int32"), ends: T.Tensor((R // G,), "int32"),
             ramp: T.Tensor((64,), "uint32"), out: T.Tensor((R, K), "uint32")):
        with T.Kernel(W) as worker:
            for iteration in T.serial(T.ceildiv(R, W)):
                row = worker + iteration * W
                if row < R:
                    keys = T.alloc_shared((NP,), "uint32")
                    result = T.alloc_shared((KP,), "uint32")
                    lanes = T.alloc_shared((64,), "uint32")
                    hist = T.alloc_shared((2, 128), "uint16")
                    flags = T.alloc_shared((2, 64), "uint32")
                    p = T.alloc_shared((2, 2, 64), "uint32")
                    counts = T.alloc_shared((64,), "uint32")
                    prefix = T.alloc_var("uint32", init=0)
                    prefix_mask = T.alloc_var("uint32", init=0)
                    remaining = T.alloc_var("int32", init=K)
                    selected_bin = T.alloc_var("int32", init=0)
                    above_offset = T.alloc_var("int32", init=0)
                    equal_offset = T.alloc_var("int32", init=0)
                    begin = T.alloc_var("int32", init=starts[row // G])
                    end = T.alloc_var("int32", init=ends[row // G])
                    T.copy(ramp, lanes)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for tile in T.serial(KP // 64):
                            T.simd.vsts(result[tile * 64], T.simd.vdup(T.uint32(N), "uint32", m), m)
                    if end - begin <= K:
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            for tile in T.serial(KP // 64):
                                ids = T.simd.vadd(T.simd.vld(lanes[0]), T.simd.vdup(T.cast(begin + tile * 64, "uint32"), "uint32", m), m)
                                valid = T.simd.vcmps(ids, T.cast(T.min(end, begin + K), "uint32"), m, op="lt")
                                T.simd.vsts(result[tile * 64], T.simd.vsel(ids, T.simd.vdup(T.uint32(N), "uint32", m), valid), m)
                    else:
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            for tile in T.serial(NP // 64):
                                T.simd.vsts(keys[tile * 64], T.simd.vdup(T.uint32(0), "uint32", m), m)
                        if G == 1:
                            T.copy(bits[row, :N], keys[:N])
                        else:
                            packed = T.alloc_shared((NP * G,), "uint32")
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                for tile in T.serial(NP * G // 64):
                                    T.simd.vsts(packed[tile * 64], T.simd.vdup(T.uint32(0), "uint32", m), m)
                            T.copy(bits[row // G, :N * G], packed[:N * G])
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                for tile in T.serial(NP // 64):
                                    ids = T.simd.vadd(T.simd.vld(lanes[0]), T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m), m)
                                    offsets = T.simd.vadd(T.simd.vmul(ids, T.simd.vdup(T.uint32(G), "uint32", m), m), T.simd.vdup(T.cast(row % G, "uint32"), "uint32", m), m)
                                    values = T.simd.vgather2(T.access_ptr(packed[0], "r", extent=NP * G), offsets, m)
                                    T.simd.vsts(keys[tile * 64], values, m)
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            sign = T.simd.vdup(T.uint32(0x80000000), "uint32", m)
                            magnitude_mask = T.simd.vdup(T.uint32(0x7fffffff), "uint32", m)
                            ones = T.simd.vdup(T.uint32(0xffffffff), "uint32", m)
                            for tile in T.serial(NP // 64):
                                b = T.simd.vld(keys[tile * 64])
                                magnitude = T.simd.vand(b, magnitude_mask, m)
                                neg = T.simd.vcmps(b, T.uint32(0x80000000), m, op="ge")
                                k = T.simd.vsel(T.simd.vxor(b, ones, m), T.simd.vxor(b, sign, m), neg)
                                k = T.simd.vsel(sign, k, T.simd.vcmps(magnitude, T.uint32(0), m, op="eq"))
                                k = T.simd.vsel(ones, k, T.simd.vcmps(magnitude, T.uint32(0x7f800000), m, op="gt"))
                                T.simd.vsts(keys[tile * 64], k, m)
                        for round in T.serial(4):
                            with T.SimdVF():
                                m32 = T.simd.pset(32)
                                m16 = T.simd.pset(16)
                                m8 = T.simd.pset(8)
                                f0 = T.simd.alloc_var("uint16")
                                f1 = T.simd.alloc_var("uint16")
                                f0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                f1 = T.simd.vdup(T.uint16(0), "uint16", m16)
                                ones32 = T.simd.vdup(T.uint32(1), "uint32", m32)
                                zero32 = T.simd.vdup(T.uint32(0), "uint32", m32)
                                zero8 = T.simd.vdup(T.uint8(0), "uint8", m8)
                                byte_mask = T.simd.vdup(T.uint32(255), "uint32", m32)
                                high_mask = T.simd.vdup(prefix_mask, "uint32", m32)
                                for tile in T.serial(NP // 64):
                                    k = T.simd.vld(keys[tile * 64])
                                    ids = T.simd.vadd(T.simd.vld(lanes[0]), T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m32), m32)
                                    visible = T.simd.vcmps(ids, T.cast(begin, "uint32"), m32, op="ge")
                                    visible = T.simd.pand(visible, T.simd.vcmps(ids, T.cast(end, "uint32"), m32, op="lt"), m32)
                                    matched = T.simd.vcmps(T.simd.vand(k, high_mask, m32), prefix, m32, op="eq")
                                    active32 = T.simd.pand(visible, matched, m32)
                                    digit32 = T.simd.vand(T.simd.vshrs(k, 24 - round * 8, m32), byte_mask, m32)
                                    active_values = T.simd.vsel(ones32, zero32, active32)
                                    # Same narrowing layout for digit and 0/1 validity. Only
                                    # nonzero validity lanes participate, regardless of part placement.
                                    digits = T.simd.vcvt(digit32, "uint8", m8, part=0)
                                    tags = T.simd.vcvt(active_values, "uint8", m8, part=0)
                                    active8 = T.simd.vcmp(tags, zero8, m8, op="ne")
                                    T.simd.dhistv2(f0, digits, active8, bin=0)
                                    T.simd.dhistv2(f1, digits, active8, bin=1)
                                T.simd.vsts(hist[0, 0], f0, m16, dist="NORM_B16")
                                T.simd.vsts(hist[1, 0], f1, m16, dist="NORM_B16")
                            selected_bin = -1
                            for bucket in T.serial(256):
                                if selected_bin < 0:
                                    if remaining > T.cast(hist[(255 - bucket) // 128, (255 - bucket) % 128], "int32"):
                                        remaining = remaining - T.cast(hist[(255 - bucket) // 128, (255 - bucket) % 128], "int32")
                                    else:
                                        selected_bin = 255 - bucket
                            prefix = prefix | (T.cast(selected_bin, "uint32") << (24 - round * 8))
                            prefix_mask = prefix_mask | (T.uint32(255) << (24 - round * 8))
                        for tile in T.serial(NP // 64):
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                one_lane = T.simd.pset(32, "PAT_VL1")
                                one = T.simd.vdup(T.uint32(1), "uint32", m)
                                zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                k = T.simd.vld(keys[tile * 64])
                                ids = T.simd.vadd(T.simd.vld(lanes[0]), T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m), m)
                                visible = T.simd.vcmps(ids, T.cast(begin, "uint32"), m, op="ge")
                                visible = T.simd.pand(visible, T.simd.vcmps(ids, T.cast(end, "uint32"), m, op="lt"), m)
                                gt = T.simd.pand(visible, T.simd.vcmps(k, prefix, m, op="gt"), m)
                                eq = T.simd.pand(visible, T.simd.vcmps(k, prefix, m, op="eq"), m)
                                g = T.simd.vsel(one, zero, gt)
                                e = T.simd.vsel(one, zero, eq)
                                T.simd.vsts(flags[0, 0], g, m)
                                T.simd.vsts(flags[1, 0], e, m)
                                T.simd.vsts(p[0, 0, 0], g, m)
                                T.simd.vsts(p[1, 0, 0], e, m)
                                gc = T.simd.vcadd(g, m)
                                ec = T.simd.vcadd(e, m)
                                T.simd.vsts(counts[0], gc, one_lane, dist="ONEPT_B32", extent=1)
                                T.simd.vsts(counts[1], ec, one_lane, dist="ONEPT_B32", extent=1)
                            for stage in T.serial(6):
                                with T.SimdVF():
                                    m = T.simd.pset(32)
                                    ids = T.simd.vld(lanes[0])
                                    distance = T.simd.vdup(T.cast(1 << stage, "uint32"), "uint32", m)
                                    valid = T.simd.vcmps(ids, T.cast(1 << stage, "uint32"), m, op="ge")
                                    prev = T.simd.vsub(ids, distance, m)
                                    zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                    safe = T.simd.vsel(prev, zero, valid)
                                    for kind in T.serial(2):
                                        shifted = T.simd.vgather2(p[kind, stage % 2, 0], safe, m)
                                        shifted = T.simd.vsel(shifted, zero, valid)
                                        summed = T.simd.vadd(T.simd.vld(p[kind, stage % 2, 0]), shifted, m)
                                        T.simd.vsts(p[kind, (stage + 1) % 2, 0], summed, m)
                            with T.SimdVF():
                                m = T.simd.pset(32)
                                one = T.simd.vdup(T.uint32(1), "uint32", m)
                                zero = T.simd.vdup(T.uint32(0), "uint32", m)
                                ids = T.simd.vadd(T.simd.vld(lanes[0]), T.simd.vdup(T.cast(tile * 64, "uint32"), "uint32", m), m)
                                grank = T.simd.vsub(T.simd.vld(p[0, 0, 0]), one, m)
                                erank = T.simd.vsub(T.simd.vld(p[1, 0, 0]), one, m)
                                gm = T.simd.vcmps(T.simd.vld(flags[0, 0]), T.uint32(0), m, op="ne")
                                em = T.simd.vcmps(T.simd.vld(flags[1, 0]), T.uint32(0), m, op="ne")
                                quota = T.simd.vcmps(erank, T.cast(T.max(remaining - equal_offset, 0), "uint32"), m, op="lt")
                                em = T.simd.pand(em, quota, m)
                                gi = T.simd.vadd(grank, T.simd.vdup(T.cast(above_offset, "uint32"), "uint32", m), m)
                                ei = T.simd.vadd(erank, T.simd.vdup(T.cast(K - remaining + equal_offset, "uint32"), "uint32", m), m)
                                gi = T.simd.vsel(gi, zero, gm)
                                ei = T.simd.vsel(ei, zero, em)
                                T.simd.vscatter(ids, T.access_ptr(result[0], "w", extent=K), gi, gm)
                                T.simd.vscatter(ids, T.access_ptr(result[0], "w", extent=K), ei, em)
                            above_offset = above_offset + T.cast(counts[0], "int32")
                            equal_offset = equal_offset + T.cast(counts[1], "int32")
                    T.copy(result[:K], out[row, :])
    return main
