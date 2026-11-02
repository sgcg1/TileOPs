"""Experimental four-kernel Decode selector; never used by the default host."""
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target='ascend', out_idx=[])
def decode_prepare(P: int, K: int=2048):
    assert P in (1, 2, 4) and K == 2048

    @T.prim_func
    def main(partial: T.Tensor((P, 256), T.int32), higher: T.Tensor((256,), T.int32), meta: T.Tensor((4,), T.int32), out: T.Tensor((2048,), T.int32), offsets: T.Tensor((P, 2), T.int32)):
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
            with T.SimtVF(threads=128):
                for i in T.Parallel(2048):
                    out[i] = 16384
                for worker in T.Parallel(P):
                    accepted = T.alloc_var(T.int32)
                    boundary = T.alloc_var(T.int32)
                    accepted = 0
                    boundary = 0
                    for previous in T.serial(P):
                        if previous < worker:
                            for bucket in T.serial(256):
                                if bucket > meta[0]:
                                    accepted = accepted + partial[previous, bucket]
                            if meta[0] >= 0:
                                boundary = boundary + partial[previous, T.max(0, T.min(meta[0], 255))]
                    offsets[worker, 0] = accepted
                    offsets[worker, 1] = boundary
    return main


@tilelang.jit(target="ascend", out_idx=[])
def decode_compact(P: int):
    assert P in (2, 4)
    def coarse_byte(value):
        bits = T.reinterpret(T.Cast(T.float16, value), T.uint16)
        magnitude = bits & T.Cast(T.uint16, 32767)
        ordered = T.if_then_else(bits & T.Cast(T.uint16, 32768) != 0, ~bits, bits | T.Cast(T.uint16, 32768))
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint16, 32768), ordered)
        canonical = T.if_then_else(magnitude > T.Cast(T.uint16, 31744), T.Cast(T.uint16, 65535), zeroed)
        return T.Cast(T.int32, canonical >> 8 & T.Cast(T.uint16, 255))

    @T.prim_func
    def main(x: T.Tensor((16384,), T.float32),
             starts: T.Tensor((1,), T.int32), ends: T.Tensor((1,), T.int32),
             meta: T.Tensor((4,), T.int32), offsets: T.Tensor((P, 2), T.int32),
             candidate_scores: T.Tensor((2048,), T.float32),
             candidate_indices: T.Tensor((2048,), T.int32), out: T.Tensor((2048,), T.int32)):
        with T.Kernel(P) as worker:
            cursors = T.alloc_shared((2,), T.int32)
            with T.SimtVF(threads=128):
                for i in T.Parallel(2):
                    cursors[i] = 0
            with T.SimtVF(threads=128):
                for tile in T.serial(16384 // P // 128):
                    for lane in T.Parallel(128):
                        i = worker * (16384 // P) + tile * 128 + lane
                        if i >= starts[0] and i < ends[0]:
                            digit = T.alloc_var(T.int32)
                            digit = coarse_byte(x[i])
                            if digit > meta[0]:
                                pos = T.atomic_add(cursors[0], 1, return_prev=True) + offsets[worker, 0]
                                if pos >= 0 and pos < 2048:
                                    out[pos] = i
                            elif digit == meta[0]:
                                slot = T.atomic_add(cursors[1], 1, return_prev=True) + offsets[worker, 1]
                                if slot >= 0 and slot < 2048:
                                    candidate_scores[slot] = x[i]
                                    candidate_indices[slot] = i
    return main


@tilelang.jit(target="ascend", out_idx=[])
def decode_finish(STABLE_TIE: bool = False):
    def score_key(value):
        bits = T.reinterpret(value, T.uint32)
        magnitude = bits & T.Cast(T.uint32, 2147483647)
        ordered = T.if_then_else(bits & T.Cast(T.uint32, 2147483648) != 0, ~bits, bits | T.Cast(T.uint32, 2147483648))
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint32, 2147483648), ordered)
        return T.if_then_else(magnitude > T.Cast(T.uint32, 2139095040), T.Cast(T.uint32, 4294967295), zeroed)

    def coarse_byte(value):
        bits = T.reinterpret(T.Cast(T.float16, value), T.uint16)
        magnitude = bits & T.Cast(T.uint16, 32767)
        ordered = T.if_then_else(bits & T.Cast(T.uint16, 32768) != 0, ~bits, bits | T.Cast(T.uint16, 32768))
        zeroed = T.if_then_else(magnitude == 0, T.Cast(T.uint16, 32768), ordered)
        canonical = T.if_then_else(magnitude > T.Cast(T.uint16, 31744), T.Cast(T.uint16, 65535), zeroed)
        return T.Cast(T.int32, canonical >> 8 & T.Cast(T.uint16, 255))

    @T.prim_func
    def main(x: T.Tensor((16384,), T.float32),
             starts: T.Tensor((1,), T.int32), ends: T.Tensor((1,), T.int32),
             meta: T.Tensor((4,), T.int32),
             candidate_scores: T.Tensor((2048,), T.float32),
             candidate_indices: T.Tensor((2048,), T.int32),
             out: T.Tensor((2048,), T.int32), final_state: T.Tensor((2,), T.int32)):
        with T.Kernel(1):
            if STABLE_TIE:
                tie_prefix = T.alloc_shared((128,), T.int32)
            keys = T.alloc_shared((2048,), T.uint32)
            indices = T.alloc_shared((2048,), T.int32)
            hist = T.alloc_shared((256,), T.int32)
            local_prefix = T.alloc_shared((256,), T.int32)
            groups = T.alloc_shared((16,), T.int32)
            # quota, committed output count, threshold digit, accepted this round
            state = T.alloc_shared((4,), T.int32)
            # score value/mask, complemented absolute index value/mask
            prefix = T.alloc_shared((4,), T.uint32)
            with T.SimtVF(threads=128):
                for i in T.Parallel(4):
                    prefix[i] = 0
                    state[i] = 0
                for i in T.Parallel(2048):
                    if meta[2] <= 2048 and i < meta[2]:
                        keys[i] = score_key(candidate_scores[i])
                        indices[i] = candidate_indices[i]
            with T.SimtVF(threads=128):
                for thread in T.Parallel(128):
                    if thread == 0:
                        state[0] = T.if_then_else(meta[3] > 2048, 2048 - meta[1], 0)
                        state[1] = meta[1]
            for round in T.serial(4 if STABLE_TIE else 8):
                with T.SimtVF(threads=128):
                    for bucket in T.Parallel(256):
                        hist[bucket] = 0
                    for thread in T.Parallel(128):
                        if thread == 0:
                            state[2] = -1
                            state[3] = 0
                with T.SimtVF(threads=128):
                    count = T.alloc_var(T.int32)
                    count = T.if_then_else(meta[2] <= 2048, meta[2], 16384)
                    for tile in T.serial(T.ceildiv(count, 128)):
                        for lane in T.Parallel(128):
                            j = tile * 128 + lane
                            if state[0] > 0 and j < count:
                                value_key = T.alloc_var(T.uint32)
                                absolute = T.alloc_var(T.int32)
                                valid = T.alloc_var(T.int32)
                                valid = 1
                                if meta[2] <= 2048:
                                    value_key = keys[T.max(0, T.min(j, 2047))]
                                    absolute = indices[T.max(0, T.min(j, 2047))]
                                else:
                                    absolute = j
                                    valid = 0
                                    if j >= starts[0] and j < ends[0]:
                                        if coarse_byte(x[j]) == meta[0]:
                                            value_key = score_key(x[j])
                                            valid = 1
                                if valid > 0:
                                    index_key = ~T.Cast(T.uint32, absolute)
                                    if (value_key & prefix[1]) == prefix[0] and (True if STABLE_TIE else (index_key & prefix[3]) == prefix[2]):
                                        key = value_key if STABLE_TIE else T.if_then_else(round < 4, value_key, index_key)
                                        shift = 24 - round * 8 if STABLE_TIE else 24 - (round % 4) * 8
                                        digit = T.alloc_var(T.int32)
                                        digit = T.Cast(T.int32, (key >> shift) & T.Cast(T.uint32, 255))
                                        T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1)
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
                            state[3] = total
                with T.SimtVF(threads=128):
                    for bucket in T.Parallel(256):
                        before = T.alloc_var(T.int32)
                        through = T.alloc_var(T.int32)
                        before = local_prefix[bucket] + groups[(255 - bucket) // 16]
                        through = before + hist[bucket]
                        hist[bucket] = before
                        if before <= state[0] and through > state[0]:
                            state[2] = bucket
                            state[3] = before
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        if thread == 0:
                            state[0] = state[0] - state[3]
                with T.SimtVF(threads=128):
                    count = T.alloc_var(T.int32)
                    count = T.if_then_else(meta[2] <= 2048, meta[2], 16384)
                    for tile in T.serial(T.ceildiv(count, 128)):
                        for lane in T.Parallel(128):
                            j = tile * 128 + lane
                            if state[3] > 0 and j < count:
                                value_key = T.alloc_var(T.uint32)
                                absolute = T.alloc_var(T.int32)
                                valid = T.alloc_var(T.int32)
                                valid = 1
                                if meta[2] <= 2048:
                                    value_key = keys[T.max(0, T.min(j, 2047))]
                                    absolute = indices[T.max(0, T.min(j, 2047))]
                                else:
                                    absolute = j
                                    valid = 0
                                    if j >= starts[0] and j < ends[0]:
                                        if coarse_byte(x[j]) == meta[0]:
                                            value_key = score_key(x[j])
                                            valid = 1
                                if valid > 0:
                                    index_key = ~T.Cast(T.uint32, absolute)
                                    if (value_key & prefix[1]) == prefix[0] and (True if STABLE_TIE else (index_key & prefix[3]) == prefix[2]):
                                        key = value_key if STABLE_TIE else T.if_then_else(round < 4, value_key, index_key)
                                        shift = 24 - round * 8 if STABLE_TIE else 24 - (round % 4) * 8
                                        digit = T.alloc_var(T.int32)
                                        digit = T.Cast(T.int32, (key >> shift) & T.Cast(T.uint32, 255))
                                        if digit > state[2]:
                                            pos = T.atomic_add(hist[T.max(0, T.min(digit, 255))], 1, return_prev=True) + state[1]
                                            if pos >= 0 and pos < 2048:
                                                out[pos] = absolute
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        if thread == 0:
                            state[1] = state[1] + state[3]
                            if state[2] >= 0:
                                if round < 4:
                                    prefix[0] = prefix[0] | (T.Cast(T.uint32, state[2]) << (24 - round * 8))
                                    prefix[1] = prefix[1] | (T.Cast(T.uint32, 255) << (24 - round * 8))
                                else:
                                    prefix[2] = prefix[2] | (T.Cast(T.uint32, state[2]) << (24 - (round - 4) * 8))
                                    prefix[3] = prefix[3] | (T.Cast(T.uint32, 255) << (24 - (round - 4) * 8))
            if STABLE_TIE:
                # Absolute-index segments avoid depending on compact's slot order.
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        count = T.alloc_var(T.int32)
                        count = 0
                        for offset in T.serial(T.ceildiv(16384, 128)):
                            i = thread * T.ceildiv(16384, 128) + offset
                            if state[0] > 0 and i >= starts[0] and i < ends[0]:
                                if score_key(x[i]) == prefix[0]:
                                    count = count + 1
                        tie_prefix[thread] = count
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        if thread == 0:
                            total = T.alloc_var(T.int32)
                            count = T.alloc_var(T.int32)
                            total = 0
                            for i in T.serial(128):
                                count = tie_prefix[i]
                                tie_prefix[i] = total
                                total = total + count
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        position = T.alloc_var(T.int32)
                        position = state[1] + tie_prefix[thread]
                        for offset in T.serial(T.ceildiv(16384, 128)):
                            i = thread * T.ceildiv(16384, 128) + offset
                            if state[0] > 0 and i >= starts[0] and i < ends[0]:
                                if score_key(x[i]) == prefix[0]:
                                    if position < state[1] + state[0]:
                                        out[position] = i
                                    position = position + 1
                with T.SimtVF(threads=128):
                    for thread in T.Parallel(128):
                        if thread == 0:
                            state[1] = state[1] + state[0]
                            state[0] = 0
            with T.SimtVF(threads=128):
                for i in T.Parallel(2):
                    final_state[i] = state[i]
    return main
