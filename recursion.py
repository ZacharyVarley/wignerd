"""

Algorithms based on:

"Numerical computation of spherical harmonics of arbitrary degree and order by
extending exponent of floating point numbers"

https://doi.org/10.1007/s00190-011-0519-2

"Numerical computation of Wigner's d-function of arbitrary high degree and
orders by extending exponent of floating point numbers"

http://dx.doi.org/10.13140/RG.2.2.31922.20160

https://www.researchgate.net/publication/309652602

Both by Toshio Fukushima.

Instead of using Toshio's x-numbers (fp64 mantissas with int32 exponents),
totaling 96 bits, we instead do everything in logspace with fp32 or fp64. As far
as I can tell this is a completely new approach to avoiding numerical
instability. I do not think I've ever seen fp32 Wigner d recursions. The
drawback comes with taking a bunch of logs and exps as the recursion is
additive and we need to use the logsumexp "trick".

"""

from typing import Tuple
import torch
from torch import Tensor
import math


@torch.jit.script
def signedlogsumexp(
    ln_a: Tensor,
    ln_b: Tensor,
    sign_a: Tensor,
    sign_b: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the log of the sum of the exponentials of the inputs and the sign of the result.

    Args:

    ln_a: Tensor
        The log of the first input.
    ln_b: Tensor
        The log of the second input.
    sign_a: Tensor
        The sign of the first input.
    sign_b: Tensor
        The sign of the second input.

    Returns:

    logsum: Tensor
        The log of the magnitude of the sum of the exponentials of the inputs.

    sign: Tensor
        The sign of the sum of the exponentials of the inputs.

    """

    # Get the maximum and minimum of the inputs
    max_ln = torch.max(ln_a, ln_b)

    # only subtract where both are not -inf
    ln_a_shifted = torch.where(~torch.isinf(ln_a), ln_a - max_ln, ln_a)
    ln_b_shifted = torch.where(~torch.isinf(ln_b), ln_b - max_ln, ln_b)

    # exponentiate the shifted inputs and sum them with the correct signs
    exp_a = torch.exp(ln_a_shifted)
    exp_b = torch.exp(ln_b_shifted)
    exp_sum = exp_a * (2 * sign_a.to(exp_a.dtype) - 1) + exp_b * (
        2 * sign_b.to(exp_b.dtype) - 1
    )
    sign_out = exp_sum >= 0
    log_exp_sum_shifted = torch.log(torch.abs(exp_sum))

    # add the maximum back to the result
    logsum = log_exp_sum_shifted + max_ln

    return logsum, sign_out


@torch.jit.script
def log_powers_of_half(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Returns the powers of 0.5 up to n.

    Args:

    n: int
        The number of powers to compute.

    device: torch.device

    Returns:

    powers of sin/cos of pi/4: Tensor

    """

    # Initialize the powers of sin and cos of pi/4.
    powers = torch.arange(0, n, device=device, dtype=dtype)

    # Compute the powers
    return math.log(0.5) * powers


@torch.jit.script
def get_index_into_array(
    coords: Tensor,
) -> Tensor:
    """

    Returns the index into the Wigner-d 1D array for a given l, m, and n.

    The 1D array is arranged in the following order:
    1) increasing l
    2) increasing m
    3) increasing n

    Only positive integers with l >= m >= n are considered.

    First few entries (in order):
    l = 0, m = 0, n = 0 *  0

    l = 1, m = 0, n = 0 ** 1
    l = 1, m = 1, n = 0 *  2
    l = 1, m = 1, n = 1 *  3

    l = 2, m = 0, n = 0 -  4
    l = 2, m = 1, n = 0 ** 5
    l = 2, m = 1, n = 1 ** 6
    l = 2, m = 2, n = 0 *  7
    l = 2, m = 2, n = 1 *  8
    l = 2, m = 2, n = 2 *  9

    l = 3, m = 0, n = 0 -- 10
    l = 3, m = 1, n = 0 -  11
    l = 3, m = 1, n = 1 -  12
    l = 3, m = 2, n = 0 ** 13
    l = 3, m = 2, n = 1 ** 14
    l = 3, m = 2, n = 2 ** 15
    l = 3, m = 3, n = 0 *  16
    l = 3, m = 3, n = 1 *  17
    l = 3, m = 3, n = 2 *  18
    l = 3, m = 3, n = 3 *  19

    l = 4, m = 0, n = 0 ---20
    ...

    Args:
        coords: Tensor
            The coordinates (..., 3) l, m, n) to get the index for.
        order_max: int
            The maximum order of the Wigner-d functions.

    """
    l, m, n = coords.unbind(-1)
    return l * (l + 1) * (l + 2) // 6 + m * (m + 1) // 2 + n


@torch.jit.script
def wigner_half_pi_stable(
    order_max: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the seed array for the Wigner D functions for the case
    where the angle is pi/2 and l >= m >= n.

    Args:
        order_max: int
            The maximum order of the Wigner D functions.
        device: torch.device
            The device to put the seed array on.

    Returns:
        wignerd: Tensor
            Log of the magnitude of Wigner d matrix entries
        wignerd_sign: Tensor
            Sign of Wigner d matrix entries

    """
    # initialize the output array
    wigner_d_logmag = torch.full(
        size=(
            order_max * (order_max + 1) * (order_max + 2) // 6
            + order_max * (order_max + 1) // 2
            + order_max
            + 1,
        ),
        fill_value=torch.nan,
        dtype=dtype,
        device=device,
    )

    wigner_d_sign = torch.ones_like(wigner_d_logmag, dtype=torch.bool, device=device)

    # initialize the powers of sin and cos of pi/4
    powers_of_half_logscale = log_powers_of_half(order_max + 1, dtype, device)

    # get all coordinates of the form (m, m, n) for m, n in [0, order_max] and m >= n
    m = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    n = torch.arange(order_max + 1, device=device, dtype=torch.int32)
    mm, nn = torch.meshgrid(m, n, indexing="ij")
    jj = mm
    coords = torch.stack((jj, mm, nn), dim=-1)
    coords = coords.view(-1, 3)
    mask = coords[:, 1] >= coords[:, 2]
    mmn_coords = coords[mask]
    first_seed_indices = get_index_into_array(mmn_coords)

    # make the coordinates the proper datatype
    mmn_coords_fp = mmn_coords.to(dtype)

    # d_m_mn = (0.5 ** m) * e_mn
    # for m >= n, e_mn = sqrt((2m)! / ((m+n)! (m-n)!))
    # in logspace thats 0.5 * (lngamma(2m + 1) - lngamma(m + n + 1) - lngamma(m - n + 1))
    wigner_d_logmag[first_seed_indices] = (
        0.5
        * (
            torch.lgamma(2 * mmn_coords_fp[:, 0] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] + mmn_coords_fp[:, 2] + 1)
            - torch.lgamma(mmn_coords_fp[:, 0] - mmn_coords_fp[:, 2] + 1)
        )
        + powers_of_half_logscale[mmn_coords[:, 0]]
    )

    coords = torch.stack((mm, mm - 1, nn - 1), dim=-1)
    coords = coords.view(-1, 3)
    mask = (coords[:, 1] >= coords[:, 2]) & (coords[:, 1] >= 0) & (coords[:, 2] >= 0)
    mp1_mn_coords = coords[mask]
    mp1_mn_coords_fp = mp1_mn_coords.to(dtype)
    second_seed_indices = get_index_into_array(mp1_mn_coords)

    # d_m+1_mn = a_mn * d_m_mn
    # a_mn = sqrt((2 * (2*m + 1)) / ((2m + 2n + 2) * (2m - 2n + 2))) * (-2n)
    # magnitude in logspace is 0.5 * (ln(2 * (2 * m + 1)) - ln(2 * m + 2 * n + 2) - ln(2 * m - 2 * n + 2)) + ln(2n)
    # sign is negative
    first_seed_indices_trunc = first_seed_indices[: len(second_seed_indices)]
    # mmn_coords_trunc = mmn_coords[: len(second_seed_indices)]

    wigner_d_logmag[second_seed_indices] = (
        wigner_d_logmag[first_seed_indices_trunc]
        + 0.5
        * (
            torch.log(2 * mp1_mn_coords_fp[:, 1] + 1)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] + mp1_mn_coords_fp[:, 2]) + 2)
            - torch.log(2 * (mp1_mn_coords_fp[:, 1] - mp1_mn_coords_fp[:, 2]) + 2)
        )
        + torch.log(2 * mp1_mn_coords_fp[:, 2])
    )
    wigner_d_sign[second_seed_indices] = False

    # can be tricky to index into flattened arrays for previous values

    # get the starting recursion coords and indices
    curr_coords = mmn_coords[: -(2 * order_max + 1)]
    curr_coords_fp = curr_coords.to(dtype)
    curr_coords[:, 0] += 2
    curr_coords_fp[:, 0] += 2.0
    curr_indices = first_seed_indices[: -(2 * order_max + 1)] + (curr_coords[:, 0]) ** 2

    for step in range(0, order_max - 1):
        # define w in logspace
        w_log = -1.0 * torch.log(2 * curr_coords_fp[:, 0] - 2) - 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1])
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2])
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2])
        )
        # v in logspace
        v_log = torch.log(2 * curr_coords_fp[:, 0]) + 0.5 * (
            torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 1] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] + 2 * curr_coords_fp[:, 2] - 2)
            + torch.log(2 * curr_coords_fp[:, 0] - 2 * curr_coords_fp[:, 2] - 2)
        )
        # b is w * v
        b_log = w_log + v_log

        # u in logspace (it is negative): -(2m)*(2n)
        u_log = (
            math.log(4)
            + torch.log(curr_coords_fp[:, 1])
            + torch.log(curr_coords_fp[:, 2])
        )

        # a in logspace
        a_log = torch.log(4 * curr_coords_fp[:, 0] - 2) + u_log + w_log

        # get a * d_l-1_mn
        term1_logmag = (
            a_log
            + wigner_d_logmag[
                curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
            ]
        )
        # for signs, a is always negative so term1_sign flips
        term1_sign = ~wigner_d_sign[
            curr_indices - ((curr_coords[:, 0]) * (curr_coords[:, 0] + 1) // 2)
        ]

        # get - (b * d_l-2_mn)
        term2_logmag = (
            b_log
            + wigner_d_logmag[curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])]
        )
        # for signs, b is always positive so term2_sign is negated
        # because we are subtracting with a call to logSUMexp
        term2_sign = ~wigner_d_sign[
            curr_indices - (curr_coords[:, 0] * curr_coords[:, 0])
        ]

        # do logsumexp trick to find a * d_l-1_mn - b * d_l-2_mn
        wigner_d_logmag[curr_indices], wigner_d_sign[curr_indices] = signedlogsumexp(
            term1_logmag, term2_logmag, term1_sign, term2_sign
        )

        # update the indices for the next iteration
        curr_coords = curr_coords[: -(order_max - 1 - step), :]
        curr_coords_fp = curr_coords_fp[: -(order_max - 1 - step), :]
        curr_indices = curr_indices[: -(order_max - 1 - step)]
        curr_coords[:, 0] += 1
        curr_coords_fp[:, 0] += 1.0
        curr_indices = curr_indices + (
            (curr_coords[:, 0] * (curr_coords[:, 0] + 1)) // 2
        )
    return wigner_d_logmag, wigner_d_sign
