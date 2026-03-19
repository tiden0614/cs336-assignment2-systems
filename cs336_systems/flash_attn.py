import torch
from torch import Tensor
import triton
import triton.language as tl
import einx
import math
from jaxtyping import Float, Bool, Int

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointers
    output_ptr, # Output pointer
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim, # Likely 1
    output_stride_row, # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr # Tile shape must be known at compile time
):
    row_tile_idx = tl.program_id(axis=0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D, ),
        strides=(weight_stride_dim, ),
        offsets=(0, ),
        block_shape=(D_TILE_SIZE, ),
        order=(0, ),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS, ),
        strides=(output_stride_row, ),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0, ),
    )

    output = tl.zeros((ROWS_TILE_SIZE, ), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROW_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0, ), padding_option="zero") # (D_TILE_SIZE, )
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))
    
    tl.store(output_block_ptr, output, boundary_check=(0,))


def weighted_sum_backward(
    x_ptr, weight_ptr, # Input
    grad_L_to_f_ptr, # Grad input
    grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwr, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D, ),
        strides=(stride_wd, ),
        offsets=(0, ),
        block_shape=(D_TILE_SIZE, ),
        order=(0, ),
    )

    # Input grad
    grad_L_to_f_block_ptr = tl.make_block_ptr(
        grad_L_to_f_ptr,
        shape=(NUM_ROWS, ),
        strides=(stride_gr, ),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0, ),
    )
    # We can directly load this grad because
    # there's no logic for it to loop over D_TILE_SIZE
    grad_L_to_f = tl.load(grad_L_to_f_block_ptr, boundary_check=(0, ), padding_option="zero")

    # Output grad: X
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    # Output grad: weight
    grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwr, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        weight = tl.load(weight_block_ptr, boundary_check=(0, ), padding_option="zero")

        # directly write to grad_x because there's no reduction at the column dimension at all
        grad_x_row = weight[None, :] * grad_L_to_f[:, None]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(x * grad_L_to_f[:, None], axis=0, keep_dims=True)
        tl.store(grad_weight_block_ptr, grad_weight_row, boundary_check=(1, ))

        x_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr.advance((D_TILE_SIZE, ))
        grad_weight_block_ptr.advance((0, D_TILE_SIZE, ))



class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = einx.rearrange("... d -> (...) d", x)

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous, "kernel assumes continuous tensors"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.ROW_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)

        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE), )](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty(tl.cdiv(n_rows, ROWS_TILE_SIZE), D, device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(tl.cdiv(n_rows, ROWS_TILE_SIZE), )](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stirde(1),
            NUM_ROWS = n_rows,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0, keepdim=False)
        return grad_x, grad_weight


# Memory layout of non-contiguous (head, seq, d_head) after rearrange from (seq, head, d_head):
#
# Physical memory is contiguous as (seq, head, d_head).
# After rearrange to (head, seq, d_head), strides become (d_head, head*d_head, 1).
#
# Example: d_head=2, heads=3, seq=4. Strides = (2, 6, 1).
# Shown as a 2D grid: rows=seq, cols=head*d_head (physical memory order).
# ██ = head 1's data, ░░ = other heads' data.
#
#            head=0      head=1      head=2
#          d0    d1    d0    d1    d0    d1
#        +-----+-----+-----+-----+-----+-----+
# seq=0  | ░░  | ░░  | ██  | ██  | ░░  | ░░  |
#        +-----+-----+-----+-----+-----+-----+
# seq=1  | ░░  | ░░  | ██  | ██  | ░░  | ░░  |
#        +-----+-----+-----+-----+-----+-----+
# seq=2  | ░░  | ░░  | ██  | ██  | ░░  | ░░  |
#        +-----+-----+-----+-----+-----+-----+
# seq=3  | ░░  | ░░  | ██  | ██  | ░░  | ░░  |
#        +-----+-----+-----+-----+-----+-----+
#
# Head 1's (seq, d_head) view occupies a non-contiguous vertical stripe.
# Within each row, d_head elements are contiguous (stride_d=1).
# Across rows, stride_seq = head*d_head = 6 (skips over other heads).
# The kernel handles this via make_block_ptr with strides=(stride_seq, stride_d).
#
# For the grid of this kernel, it's launched as 3D-parallelism: (batch, head, seq)
# axis=0: batch
# axis=1: head
# axis=2: seq
@triton.jit
def flash_attn2_forward_triton(
    q_ptr, k_ptr, v_ptr, # Input pointers
    o_ptr, L_ptr, # Output pointer

    # strides
    q_stride_b, q_stride_h, q_stride_seq, q_stride_d,
    k_stride_b, k_stride_h, k_stride_seq, k_stride_d,
    v_stride_b, v_stride_h, v_stride_seq, v_stride_d,
    o_stride_b, o_stride_h, o_stride_seq, o_stride_d,
    L_stride_b, L_stride_h, L_stride_seq, 

    SEQ_LEN, D, D_sqrt,
    # Tile shape must be known at compile time
    # Br: used by Q, O and L
    # Bc: used by K, V
    Br: tl.constexpr, Bc: tl.constexpr 
):
    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    #
    # $$ i \in [0, T_r), T_r = \lceil \frac {N}{B_r} \rceil $$
    #
    i = tl.program_id(axis=2)

    q_block_ptr = tl.make_block_ptr(
        q_ptr + q_stride_b * batch_idx + q_stride_h * head_idx,
        shape=(SEQ_LEN, D),
        strides=(q_stride_seq, q_stride_d),
        offsets=(i * Br, 0),
        block_shape=(Br, D),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + k_stride_b * batch_idx + k_stride_h * head_idx,
        shape=(SEQ_LEN, D),
        strides=(k_stride_seq, k_stride_d),
        offsets=(0, 0),
        block_shape=(Bc, D),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + v_stride_b * batch_idx + v_stride_h * head_idx,
        shape=(SEQ_LEN, D),
        strides=(v_stride_seq, v_stride_d),
        offsets=(0, 0),
        block_shape=(Bc, D),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        o_ptr + o_stride_b * batch_idx + o_stride_h * head_idx,
        shape=(SEQ_LEN, D),
        strides=(o_stride_seq, o_stride_d),
        offsets=(i * Br, 0),
        block_shape=(Br, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + L_stride_b * batch_idx + L_stride_h * head_idx,
        shape=(SEQ_LEN, ),
        strides=(L_stride_seq, ),
        offsets=(i * Br, ),
        block_shape=(Br, ),
        order=(0, ),
    )

    Q_i = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_i = tl.full((Br, ), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Br, ), dtype=tl.float32)
    O = tl.zeros((Br, D), dtype=tl.float32)

    T_c = tl.cdiv(SEQ_LEN, Bc)
    for j in range(T_c):
        K_j = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S_ij = Q_i @ tl.trans(K_j) / D_sqrt
        m_i_new = tl.max(m_i, tl.max(S_ij, axis=1, keepdim=False))
        P_i = tl.exp(S_ij - m_i_new[:, None])
        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(P_i, axis=1, keepdim=False)

        # Rescale O: the accumulated O contains terms of the form e^{S - m_old} @ V.
        # To convert to the new max m_new, we multiply by e^{m_old - m_new}:
        #
        #   $$e^{S - m_{new}} = e^{S - m_{old}} * e^{m_{old} - m_{new}}$$
        #
        # So: $$ O_{new} = O_{old} * e^{m_{old} - m_{new}} + P_iV_j $$
        #
        # Note: FA2 paper (Algorithm 1, line 10) writes this as
        #   $$diag(e^{m^{j-1} - m^{j}})^{-1} O^{j-1}$$
        # which inverts the correction factor -- this appears to be an error
        # in the paper. The correct operation is multiplication by
        # $$e^{m_{old} - m_{new}}$$, not its inverse.
        O = tl.exp(m_i - m_i_new)[:, None] * O + P_i @ V_j
        m_i = m_i_new

        k_block_ptr.advance((Bc, 0))
        v_block_ptr.advance((Bc, 0))
        
    tl.store(o_block_ptr, O / l_i[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(1, ))


class FlashAttention2Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        q: Float[Tensor, "batch seq head d_head"],
        k: Float[Tensor, "batch seq head d_head"],
        v: Float[Tensor, "batch seq head d_head"]
    ) -> Float[Tensor, "batch seq head d_head"]:

        for x in (q, k, v):
            assert x.is_contiguous()

        BATCH, SEQ_LEN, HEAD, D_HEAD = q.shape
        D_sqrt = math.sqrt(D_HEAD)

        # TODO make Br and Bc more dynamic relative to total SRAM
        ctx.Br = 16
        ctx.Bc = 16

        o: Float[Tensor, "batch seq head d_head"] = torch.empty(q.shape, device=q.device)
        L: Float[Tensor, "batch seq head"] = torch.empty(BATCH, SEQ_LEN, HEAD, device=q.device)

        def _reshape_4d(x: Float[Tensor, "batch seq head d_head"]):
            x1: Tensor = einx.rearrange("batch seq head d_head -> batch head seq d_head", x)
            b, h, s, d = x1.stride()
            return (x1, b, h, s, d)

        def _reshape_3d(x: Float[Tensor, "batch seq head"]):
            x1: Tensor = einx.rearrange("batch seq head -> batch head seq", x)
            b, h, s = x1.stride()
            return (x1, b, h, s)

        q_ptr, q_stride_b, q_stride_h, q_stride_seq, q_stride_d = _reshape_4d(q)
        k_ptr, k_stride_b, k_stride_h, k_stride_seq, k_stride_d = _reshape_4d(k)
        v_ptr, v_stride_b, v_stride_h, v_stride_seq, v_stride_d = _reshape_4d(v)
        o_ptr, o_stride_b, o_stride_h, o_stride_seq, o_stride_d = _reshape_4d(o)
        L_ptr, l_stride_b, l_stride_h, l_stride_seq = _reshape_3d(L)

        # axis=0: batch
        # axis=1: head
        # axis=2: seq
        Tr = triton.cdiv(SEQ_LEN, ctx.Br)
        flash_attn2_forward_triton[(BATCH, HEAD, Tr)](
            q_ptr, k_ptr, v_ptr, # Input pointers
            o_ptr, L_ptr, # Output pointers

            # strides
            q_stride_b, q_stride_h, q_stride_seq, q_stride_d,
            k_stride_b, k_stride_h, k_stride_seq, k_stride_d,
            v_stride_b, v_stride_h, v_stride_seq, v_stride_d,
            o_stride_b, o_stride_h, o_stride_seq, o_stride_d,
            l_stride_b, l_stride_h, l_stride_seq,

            SEQ_LEN=SEQ_LEN, D=D_HEAD, D_sqrt=D_sqrt,
            Br=ctx.Br, Bc=ctx.Bc
        )

        ctx.save_for_backward(q, k, v, o, L)
        return o

