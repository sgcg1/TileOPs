"""Experimental Decode histogram/merge cost probe, not a TopK implementation."""
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target="ascend", out_idx=[])
def decode_histogram(N: int, P: int):
    assert N == 16384 and P in (1, 2, 4)
    def coarse_byte(value):
        bits = T.reinterpret(T.Cast(T.float16, value), T.uint16)
        magnitude = bits & T.Cast(T.uint16, 32767)
        ordered = T.if_then_else(bits & T.Cast(T.uint16, 32768) != 0, ~bits, bits | T.Cast(T.uint16, 32768))
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint16, 32768), ordered)
        canonical = T.if_then_else(magnitude > T.Cast(T.uint16, 31744), T.Cast(T.uint16, 65535), zeroed)
        return T.Cast(T.int32, canonical >> 8 & T.Cast(T.uint16, 255))

    @T.prim_func
    def main(x: T.Tensor((N,), T.float32),
             starts: T.Tensor((1,), T.int32), ends: T.Tensor((1,), T.int32),
             partial: T.Tensor((P, 256), T.int32)):
        with T.Kernel(P) as worker:
            hist = T.alloc_shared((256,), T.int32)
            with T.SimtVF(threads=128):
                for bucket in T.Parallel(256):
                    hist[bucket] = 0
            with T.SimtVF(threads=128):
                for tile in T.serial(N // P // 128):
                    for lane in T.Parallel(128):
                        i = worker * (N // P) + tile * 128 + lane
                        if i >= starts[0] and i < ends[0]:
                            digit = T.alloc_var(T.int32)
                            digit = coarse_byte(x[i])
                            T.assume(digit >= 0)
                            T.assume(digit < 256)
                            T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1)
            with T.SimtVF(threads=128):
                for bucket in T.Parallel(256):
                    partial[worker, bucket] = hist[bucket]
    return main


@tilelang.jit(target="ascend", out_idx=[])
def decode_merge(P: int, K: int = 2048):
    assert P in (1, 2, 4) and K == 2048

    @T.prim_func
    def main(partial: T.Tensor((P, 256), T.int32),
             higher: T.Tensor((256,), T.int32), meta: T.Tensor((4,), T.int32)):
        with T.Kernel(1):
            hist = T.alloc_shared((256,), T.int32)
            local_prefix = T.alloc_shared((256,), T.int32)
            groups = T.alloc_shared((16,), T.int32)
            with T.SimtVF(threads=128):
                for bucket in T.Parallel(256):
                    total = T.alloc_var(T.int32)
                    total = 0
                    for worker in T.serial(P):
                        total = total + partial[worker, bucket]
                    hist[bucket] = total
            with T.SimtVF(threads=128):
                for group in T.Parallel(16):
                    total = T.alloc_var(T.int32)
                    total = 0
                    for j in T.serial(16):
                        bucket = 255 - group * 16 - j
                        local_prefix[bucket] = total
                        total = total + hist[bucket]
                    groups[group] = total
            with T.SimtVF(threads=128):
                for thread in T.Parallel(128):
                    if thread == 0:
                        total = T.alloc_var(T.int32)
                        count = T.alloc_var(T.int32)
                        total = 0
                        for group in T.serial(16):
                            count = groups[group]
                            groups[group] = total
                            total = total + count
                        meta[0] = -1
                        meta[1] = total
                        meta[2] = 0
                        meta[3] = total
            with T.SimtVF(threads=128):
                for bucket in T.Parallel(256):
                    before = T.alloc_var(T.int32)
                    before = local_prefix[bucket] + groups[(255 - bucket) // 16]
                    higher[bucket] = before
                    if before <= K and before + hist[bucket] > K:
                        meta[0] = bucket
                        meta[1] = before
                        meta[2] = hist[bucket]
    return main
