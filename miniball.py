#
# Copyright (c) 2019-2023 Alexandre Devert
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy


__author__ = "Alexandre Devert <marmakoide@hotmail.fr>"
__version__ = "1.2.1"

from chromatic_tda.utils.timing import TimingUtils


def get_circumsphere(S):
    """
    Computes the circumsphere of a set of points

    Parameters
    ----------
    S : (M, N) ndarray, where 1 <= M <= N + 1
            The input points

    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere
    """
    TimingUtils().start("Miniball :: Get Circumsphere")

    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A")
    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A :: U")
    U = S[1:] - S[0]
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A :: U")
    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A :: B")
    B = numpy.sqrt(numpy.square(U).sum(axis=1))
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A :: B")
    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A :: U2")
    U = U / B[:, None]
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A :: U2")
    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A :: B2")
    B = B / 2
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A :: B2")
    TimingUtils().start("Miniball :: Get Circumsphere :: Define U, B, A :: A")
    A = numpy.inner(U, U)
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A :: A")
    TimingUtils().stop("Miniball :: Get Circumsphere :: Define U, B, A")

    TimingUtils().start("Miniball :: Get Circumsphere :: LstSq")
    x, *_ = numpy.linalg.lstsq(A, B, rcond=None)
    TimingUtils().stop("Miniball :: Get Circumsphere :: LstSq")

    TimingUtils().start("Miniball :: Get Circumsphere :: Check Correctness")
    if not numpy.allclose(A @ x, B):
        raise numpy.linalg.LinAlgError('Linear equation has no solution.')
    TimingUtils().stop("Miniball :: Get Circumsphere :: Check Correctness")

    TimingUtils().start("Miniball :: Get Circumsphere :: Compute C, r2")
    TimingUtils().start("Miniball :: Get Circumsphere :: Compute C, r2 :: C")
    C = numpy.dot(x, U)
    TimingUtils().stop("Miniball :: Get Circumsphere :: Compute C, r2 :: C")
    TimingUtils().start("Miniball :: Get Circumsphere :: Compute C, r2 :: r2")
    r2 = numpy.square(C).sum()
    TimingUtils().stop("Miniball :: Get Circumsphere :: Compute C, r2 :: r2")
    TimingUtils().start("Miniball :: Get Circumsphere :: Compute C, r2 :: C+")
    C = C + S[0]
    TimingUtils().stop("Miniball :: Get Circumsphere :: Compute C, r2 :: C+")
    TimingUtils().stop("Miniball :: Get Circumsphere :: Compute C, r2")

    TimingUtils().stop("Miniball :: Get Circumsphere")
    return C, r2


def get_bounding_ball(S, epsilon=1e-7, rng=numpy.random.default_rng()):
    """
    Computes the smallest bounding ball of a set of points

    Parameters
    ----------
    S : (M, N) ndarray, where 1 <= M <= N + 1
            The input points

    epsilon : float
            Tolerance used when testing if a set of point belongs to the same
            sphere. Default is 1e-7

    rng : numpy.random.Generator
        Pseudo-random number generator used internally. Default is the default
        one provided by numpy.

    Returns
    -------
    C, r2 : ((2) ndarray, float)
            The center and the squared radius of the circumsphere
    """

    # Iterative implementation of Welzl's algorithm, see
    # "Smallest enclosing disks (balls and ellipsoids)" Emo Welzl 1991

    def circle_contains(D, p):
        TimingUtils().start("Miniball :: Circle Contains")
        c, r2 = D
        contains = numpy.square(p - c).sum() <= r2
        TimingUtils().stop("Miniball :: Circle Contains")
        return contains

    def get_boundary(R):
        if len(R) == 0:
            return numpy.zeros(S.shape[1]), 0.0

        if len(R) <= S.shape[1] + 1:
            return get_circumsphere(S[R])

        c, r2 = get_circumsphere(S[R[: S.shape[1] + 1]])
        if numpy.all(
            numpy.fabs(numpy.square(S[R] - c).sum(axis=1) - r2) < epsilon
        ):
            return c, r2

    class Node(object):
        def __init__(self, P, R):
            self.P = P
            self.R = R
            self.D = None
            self.pivot = None
            self.left = None
            self.right = None

    def traverse(node):
        TimingUtils().start("Miniball :: Traverse")

        stack = [node]
        while len(stack) > 0:
            node = stack.pop()

            if len(node.P) == 0 or len(node.R) >= S.shape[1] + 1:
                TimingUtils().start("Miniball :: Traverse :: Case 1")
                node.D = get_boundary(node.R)
                TimingUtils().stop("Miniball :: Traverse :: Case 1")
            elif node.left is None:
                TimingUtils().start("Miniball :: Traverse :: Case 2")
                pivot_index = rng.integers(len(node.P))
                node.pivot = node.P[pivot_index]
                node.left = Node(
                    node.P[:pivot_index] + node.P[pivot_index + 1:],
                    node.R
                )
                stack.extend((node, node.left))
                TimingUtils().stop("Miniball :: Traverse :: Case 2")
            elif node.right is None:
                TimingUtils().start("Miniball :: Traverse :: Case 3")
                if circle_contains(node.left.D, S[node.pivot]):
                    node.D = node.left.D
                else:
                    node.right = Node(node.left.P, node.R + [node.pivot])
                    stack.extend((node, node.right))
                TimingUtils().stop("Miniball :: Traverse :: Case 3")
            else:
                TimingUtils().start("Miniball :: Traverse :: Case 4")
                node.D = node.right.D
                node.left, node.right = None, None
                TimingUtils().stop("Miniball :: Traverse :: Case 4")

        TimingUtils().stop("Miniball :: Traverse")

    S = S.astype(float, copy=False)
    root = Node(list(range(S.shape[0])), [])
    traverse(root)
    return root.D
