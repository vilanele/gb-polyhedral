from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Parent
from sage.rings.polynomial.polydict import ETuple
from sage.structure.category_object import normalize_names
from sage.categories.algebras import Algebras
from sage.rings.infinity import Infinity
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polydict import PolyDict
from sage.rings.infinity import Infinity
from sage.rings.infinity import Infinity
from sage.structure.element import MonoidElement
from sage.rings.polynomial.polydict import ETuple
from sage.monoids.monoid import Monoid_class
from sage.rings.polytopal.PolyCircleTerm import PolyCircleTerm
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector
from sage.rings.polynomial.term_order import TermOrder
from sage.matrix.constructor import matrix
from sage.structure.sage_object import SageObject


class GeneralizedOrder(SageObject):
    def __init__(self, n, group_order="lex", score_function="min"):
        r"""
        Create a generalized monomial order in ``n`` varaibles with optional group order and score function.

        INPUT:

        - ``n`` -- the number of variables
        - ``group_order`` (default: ``lex``) -- the name of a group order on `\ZZ^n`, choices are: "lex"
        - ``score_function`` (default: ``min``) -- the name of a score function, choices are: "min", "degmin"

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: GeneralizedOrder(2)
            Generalized monomial order in 2 variables using (lex, min)

            sage: GeneralizedOrder(8, group_order="lex")
            Generalized monomial order in 8 variables using (lex, min)

            sage: GeneralizedOrder(3, score_function="min")
            Generalized monomial order in 3 variables using (lex, min)

            sage: GeneralizedOrder(3, group_order="lex", score_function="degmin")
            Generalized monomial order in 3 variables using (lex, degmin)
        """
        self._n = n
        self._n_cones = self._n + 1
        # Build cones
        self._cones = [matrix.identity(ZZ, n)]
        for i in range(0, n):
            mat = matrix.identity(ZZ, n)
            mat.set_column(i, vector(ZZ, [-1] * n))
            self._cones.append(mat)
        # Set group order. Add more here.
        if group_order in ["lex"]:
            self._group_order = TermOrder(name=group_order)
        else:
            raise ValueError("Available group order are: 'lex'")
        # Set score function. Add more here.
        if score_function == "min":
            self._score_function = min_score_function
        elif score_function == "degmin":
            self._score_function = degmin_score_function
        else:
            raise ValueError("Available score function are: 'min', 'degmin'")
        # Store names
        self._group_order_name = group_order
        self._score_function_name = score_function

    def _repr_(self):
        r"""
        TESTS::
            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: GeneralizedOrder(2)._repr_()
            'Generalized monomial order in 2 variables using (lex, min)'
        """
        group = self._group_order_name
        function = self._score_function_name
        n = str(self._n)
        return "Generalized monomial order in %s variables using (%s, %s)" % (
            n,
            group,
            function,
        )

    def __hash__(self):
        r"""
        Return the hash of self. It depends on the number of variables, the group_order and the score function.
        """
        return hash(self._group_order_name + self._score_function_name + str(self._n))

    def n_cones(self):
        r"""
        Return the number of cones (which is the number of variables plus one).

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.n_cones()
            3
        """
        return self._n_cones

    def cones(self):
        r"""
        Return the list of matrices containing the generators of the cones.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.cones()
            [
            [1 0]  [-1  0]  [ 1 -1]
            [0 1], [-1  1], [ 0 -1]
            ]
        """
        return self._cones

    def cone(self, i):
        r"""
        Return the matrix whose columns are the generators of the ``i``-th cone.

        INPUT:

        - `i` -- an integer, a cone index

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.cone(1)
            [-1 0]
            [-1 1]
        """
        if i < 0 or i > self._n:
            raise IndexError("cone index out of range")
        return self._cones[i]

    def group_order(self):
        r"""
        Return the underlying group order.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: G = GeneralizedOrder(2)
            sage: G.group_order()
            Lexicographic term order
        """
        return self._group_order

    def group_order_name(self):
        r"""
        Return the name of the underlying group order.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2,group_order="lex")
            sage: order.group_order_name()
            'lex'
        """
        return self._group_order_name

    def score_function_name(self):
        r"""
        Return the name of the underlying score function.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2,score_function="degmin")
            sage: order.score_function_name()
            'degmin'
        """
        return self._score_function_name

    def score(self, t):
        r"""
        Compute the score of a tuple ``t``.

        INPUT:

        - `t` -- a tuple of integers

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.score((-2,3))
            2
        """
        return self._score_function(t)

    def compare(self, a, b):
        r"""
        Return 1, 0 or -1 whether tuple ``a`` is greater than, equal or less than to tuple ``b`` respectively.

        INPUT:

        - `a` and `b` -- two tuples of integers

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(3)
            sage: order.compare((1,2,-2), (3,-4,5))
            -1
            sage: order.compare((3,-4,5), (1,2,-2))
            1
            sage: order.compare((3,-4,5), (3,-4,5))
            0
        """
        if a == b:
            return 0
        diff = self._score_function(a) - self._score_function(b)
        if diff != 0:
            return 1 if diff > 0 else -1
        else:
            return 1 if self._group_order.greater_tuple(a, b) == a else -1

    def greatest_tuple(self, *L):
        r"""
        Return the greatest tuple in ``L``.

        INPUT:

        - *L -- integer tuples

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(3)
            sage: order.greatest_tuple((0,0,1),(2,3,-2))
            (2, 3, -2)
            sage: L = [(1,2,-1),(3,-3,0),(4,-5,-6)]
            sage: order.greatest_tuple(*L)
            (4, -5, -6)
        """
        n = len(L)
        if n == 0:
            raise ValueError("empty list of tuples")

        if n == 1:
            return L[0]
        else:
            a = L[0]
            b = self.greatest_tuple(*L[1:])
            return a if self.compare(a, b) == 1 else b

    def translate_to_cone(self, i, L):
        r"""
        Return a tuple ``t`` such that `t + L` is contained in the ``i``-th cone.

        INPUT:

        - ``i`` -- an integer, a cone index
        - ``L`` -- a list of integer tuples

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.translate_to_cone(0, [(1,2),(-2,-3),(1,-4)])
            (2, 4)
        """
        # print(L)
        cone_matrix = self._cones[i]
        T = matrix(ZZ, [cone_matrix * vector(v) for v in L])
        return tuple(cone_matrix * vector([-min(0, *c) for c in T.columns()]))

    def greatest_tuple_for_cone(self, i, *L):
        r"""
        Return the greatest tuple of the list of tuple `L` with respect to the `i`-th cone.

        This is the unique tuple `t` in `L`  such that each time the greatest tuple of `s + L`
        for a tuple `s` is contained in the `i`-th cone, it is equal to `s + t`.

        INPUT:

        - ``i`` -- an integer, a cone index
        - ``L`` -- a list of integer tuples

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: L = [(1,2), (-2,-2), (-4,5), (5,-6)]
            sage: t = order.greatest_tuple_for_cone(1, *L);t
            (-4, 5)

        We can check the result::

            sage: s = order.translate_to_cone(1, L)
            sage: sL = [tuple(vector(s) + vector(l)) for l in L]
            sage: tuple(vector(s) + vector(t)) == order.greatest_tuple(*sL)
            True

        """
        t = vector(self.translate_to_cone(i, L))
        L = [vector(l) for l in L]
        return tuple(vector(self.greatest_tuple(*[tuple(t + l) for l in L])) - t)

    def is_in_cone(self, i, t):
        r"""
        Test whether the tuple ``t`` is contained in the ``i``-th cone or not.

        INPUT:

        - ``i`` -- an integer, a cone index
        - ``t`` -- a tuple of integers

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: order.is_in_cone(0, (1,2))
            True
            sage: order.is_in_cone(0,(-2,3))
            False
        """
        return all(c >= 0 for c in self.cone(i) * vector(t))

    def generator(self, i, L):
        r"""
        Return the generator of the module over the ``i``-th cone for ``L``.

        This is the monoïd of elements `t \in \ZZ^n` such that the greatest
        tuple of `t + L` is contained in the ``i``-th cone.

        INPUT:

        - ``i`` -- an integer, a cone index
        - ``L`` -- a list of iinteger tuples

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(3)
            sage: L = [(1,2,-2), (3,4,-5), (-6,2,-7), (3,-6,1)]
            sage: order.generator(0,L)
            (6, 6, 7)

            sage: order.generator(2,L)
            (5, 5, 6)
        """

        cone_matrix = self._cones[i]
        t = vector(self.translate_to_cone(i, L))
        L = [vector(l) for l in L]
        for c in cone_matrix.columns():
            while self.is_in_cone(i, self.greatest_tuple(*[tuple(t + l) for l in L])):
                t = t - c
            t = t + c
        return tuple(t)

    def generator_for_pair(self, i, L1, L2):
        r"""
        Return the generator of the module over the ``i``-th cone for ``L1`` and ``L2``.

        INPUT:

        - ``i`` -- an integer, a cone index
        - ``L1`` -- a list of integer tuples
        - ``L2`` -- a list of integer tuples

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrder
            sage: order = GeneralizedOrder(2)
            sage: L1 = [(2,3),(-4,2),(-1,-2)]
            sage: L2 = [(1,-6),(5,-2),(3,4)]
            sage: order.generator_for_pair(1,L1,L2)
            (1, 5)

        """
        cone_matrix = self._cones[i]
        lm1 = vector(self.greatest_tuple_for_cone(i, *L1))
        lm2 = vector(self.greatest_tuple_for_cone(i, *L2))
        g1 = vector(self.generator(i, L1))
        g2 = vector(self.generator(i, L2))
        m = vector([max(a, b) for a, b in zip(lm1 + g1, lm2 + g2)])
        return tuple(cone_matrix * m)


# Score functions. Add more here and update the __init__ method.
def min_score_function(t):
    return -min(0, *t)


def degmin_score_function(t):
    return sum(t) - (len(t) + 1) * min(0, *t)


class PolyCircleAlgebraElement(CommutativeAlgebraElement):
    def __init__(self, parent, x=None, prec=None):
        CommutativeAlgebraElement.__init__(self, parent)
        if isinstance(x, PolyCircleAlgebraElement):
            self._poly = x._poly
            if prec is None:
                self._prec = x._prec
            else:
                self._prec = prec
        elif isinstance(x, PolyCircleTerm):
            self._poly = parent._polynomial_ring(
                PolyDict({x.exponent(): parent._field(x.coefficient())})
            )
            self._prec = Infinity
        else:
            try:
                self._poly = parent._polynomial_ring(x)
                if prec is None:
                    self._prec = Infinity
                else:
                    self._prec = prec
            except TypeError:
                raise

    def terms(self):
        r"""
        Return the terms of ``self`` in a list, sorted by decreasing order.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y + x*y + y^-1 + 1*x^-3
            sage: f.terms()
            [(1 + O(2^20))*x^-3,
            (1 + O(2^20))*y^-1,
            (1 + O(2^20))*y,
            (1 + O(2^20))*x*y,
            (2 + O(2^21))*x]
        """
        terms = [
            PolyCircleTerm(self.parent()._monoid_of_terms, coeff=c, exponent=e)
            for e, c in self._poly.dict().items()
        ]
        return sorted(terms, reverse=True)

    def leading_term(self, i=None):
        r"""
        Return the leading term of this series if ``i`` is None, otherwise the
        leading term for the ``i``-th cone.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.leading_term()
            (1 + O(2^20))*x
            sage: f.leading_term(1)
            (1 + O(2^20))*y

        """
        if i is None:
            return self.terms()[0]

        ini = self.initial()._poly
        le = self.parent()._order.greatest_tuple_for_cone(i, *ini.exponents())
        return PolyCircleTerm(self.parent()._monoid_of_terms, ini[le], le)

    def leading_monomial(self, i=None):
        r"""
        Return the leading monomial of this series if ``i`` is None, otherwise the
        leading monomial for the ``i``-th cone.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.leading_monomial()
            (1 + O(2^20))*x
            sage: f.leading_monomial(1)
            (1 + O(2^20))*y
        """
        return self.leading_term(i).monomial()

    def leading_coefficient(self, i=None):
        r"""
        Return the leading coefficient of this series if ``i`` is None, otherwise the
        leading coefficient for the ``i``-th cone.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.leading_coefficient()
            1 + O(2^20)
            sage: f.leading_coefficient(1)
            1 + O(2^20)
        """
        return self.leading_term(i).coefficient()

    def leadings(self, i=None):
        r"""
        Return a list containing the leading coefficient, monomial and term
        of this series if ``i`` is None, or th same for the ``i``-th cone otehrwise.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.leadings()
            (1 + O(2^20), (1 + O(2^20))*x, (1 + O(2^20))*x)
        """
        lt = self.leading_term(i)
        return (lt.coefficient(), lt.monomial(), lt)

    def initial(self):
        r"""
        Return the initial part of this series.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.initial()
            (1 + O(2^20))*x + (1 + O(2^20))*y
        """
        terms = self.terms()
        v = min([t.valuation() for t in terms])
        return sum(
            [self.__class__(self.parent(), t) for t in terms if t.valuation() == v]
        )

    def valuation(self):
        r"""
        Return the valuation of this series.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.valuation()
            1
        """
        if self.is_zero():
            return Infinity
        return self.leading_term().valuation()

    def _normalize(self):
        r"""
        Normalize this series.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.add_bigoh(2) # Indirect doctest
            (1 + O(2))*x + (1 + O(2))*y + OO(2)
        """
        if self._prec == Infinity:
            return
        terms = []
        for t in self.terms():
            exponent = t.exponent()
            coeff = t.coefficient()
            v = self.parent().log_radii().dotprod(exponent)
            if coeff.precision_absolute() > self._prec + v:
                coeff = coeff.add_bigoh(self._prec + v)
            if coeff.valuation() < self._prec + v:
                t._coeff = coeff
                terms.append(t)

        self._poly = sum([self.__class__(self.parent(), t)._poly for t in terms])

    def add_bigoh(self, prec):
        r"""
        Return this series truncated to precision ``prec``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.add_bigoh(2) # Indirect doctest
            (1 + O(2))*x + (1 + O(2))*y + OO(2)
        """
        return self.parent()(self, prec=prec)

    def is_zero(self):
        r"""
        Test if this series is zero.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: f.is_zero()
            False
            sage: (f-f).is_zero()
            True
        """
        # self._normalize()
        return self._poly == 0

    def __eq__(self, other):
        r"""
        Test equality of this series and ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: g = 3*x*y
            sage: f == g
            False
            sage: f == f
            True
        """
        diff = self - other
        return diff.is_zero()

    def _add_(self, other):
        r"""
        Return the addition of this series and ``other`.
        The precision is adjusted to the min of the inputs precisions.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: g = 3*x*y
            sage: f + g
            (1 + O(2^20))*x + (1 + O(2^20))*y + (2^2 + O(2^22))*x*y^-1 + (1 + 2 + O(2^20))*x*y
        """
        prec = min(self._prec, other._prec)
        ans = self.__class__(self.parent(), self._poly + other._poly, prec=prec)
        ans._normalize()
        return ans

    def _sub_(self, other):
        r"""
        Return the substraction of this series and ``other`.
        The precision is adjusted to the min of the inputs precisions.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: g = -3*x*y
            sage: f - g
            (1 + O(2^20))*x + (1 + O(2^20))*y + (2^2 + O(2^22))*x*y^-1 + (1 + 2 + O(2^20))*x*y
        """
        prec = min(self._prec, other._prec)
        ans = self.__class__(self.parent(), self._poly - other._poly, prec=prec)
        ans._normalize()
        return ans

    def _mul_(self, other):
        r"""
        Return the multiplication of this series and ``other`.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: g = 3*x*y
            sage: f*g
            (1 + 2 + O(2^20))*x^2*y + (1 + 2 + O(2^20))*x*y^2 + (2^2 + 2^3 + O(2^22))*x^2
        """
        a = self.valuation() + other._prec
        b = other.valuation() + self._prec
        prec = min(a, b)
        ans = self.__class__(self.parent(), self._poly * other._poly, prec=prec)
        ans._normalize()
        return ans

    def _div_(self, other):
        r"""
        Return the division of this series by ``other`.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x
            sage: g = x*y
            sage: f/g
            (1 + O(2^20))*y^-1
        """
        prec = self._prec - other.valuation()
        ans = self.__class__(self.parent(), self._poly / other._poly, prec=prec)
        ans._normalize()
        return ans

    def _neg_(self):
        r"""
        Return the negation of this series.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = -x
            sage: -f
            (1 + O(2^20))*x
        """
        ans = self.__class__(self.parent(), -self._poly, prec=self._prec)
        ans._normalize()
        return ans

    def _repr_(self):
        r"""
        Return a string representation of this series.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: x
            (1 + O(2^20))*x
        """
        self._normalize()
        terms = self.terms()
        r = ""
        for t in terms:
            if t.valuation() < self._prec:
                s = repr(t)
                if r == "":
                    r += s
                elif s[0] == "-":
                    r += " - " + s[1:]
                else:
                    r += " + " + s

        if not self._prec == Infinity:
            if r:
                r += " + "
            r += f"OO({self._prec})"

        return r

    def critical_pairs(self, other, i=None):
        r"""
        Return all S-pairs of this series and ``other`` for the ``i``-th cone,
        or all S-pairs for all cones if ``i`` is None.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = x + y + 4*x*y^-1
            sage: g = 3*x*y
            sage: f.critical_pairs(g)
            [(1 + 2 + O(2^20))*y^2 + (2^2 + 2^3 + O(2^22))*x,
            (1 + 2 + O(2^20))*x*y + (2^2 + 2^3 + O(2^22))*x,
            (1 + 2 + O(2^20))*x^-1*y + (2^2 + 2^3 + O(2^22))*y^-1]
        """
        if i is None:
            return [
                self.critical_pairs(other, j)
                for j in range(self.parent()._order.n_cones())
            ]

        lcm = self.parent()._order.generator_for_pair(
            i, self._poly.exponents(), other._poly.exponents()
        )
        lcm = PolyCircleTerm(self.parent()._monoid_of_terms, coeff=1, exponent=lcm)
        lcf, lmf, _ = self.leadings(i)
        lcg, lmg, _ = other.leadings(i)
        return lcg * lcm._divides(lmf) * self - lcf * lcm._divides(lmg) * other


class PolyCircleAlgebra(Parent, UniqueRepresentation):
    r"""
    Parent class for series with finite precision in a polyhedral algebra with series converging on a poly-circle.

    EXAMPLES::

        sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
        sage: x,y = A.gens()
    """

    Element = PolyCircleAlgebraElement

    def __init__(
        self, field, log_radii, names, score_function="min", group_order="lex", prec=10
    ):
        self._field = field
        self.element_class = PolyCircleAlgebraElement
        self._prec = prec
        self._log_radii = ETuple(log_radii)
        self._names = normalize_names(-1, names)
        self._polynomial_ring = LaurentPolynomialRing(field, names=names)
        self._gens = [
            self.element_class(self, g, prec=Infinity)
            for g in self._polynomial_ring.gens()
        ]
        self._ngens = len(self._gens)
        self._order = GeneralizedOrder(
            self._ngens,
            group_order=group_order,
            score_function=score_function,
        )
        self._monoid_of_terms = PolyCircleTermMonoid(self)
        Parent.__init__(self, names=names, category=Algebras(self._field).Commutative())

    def generalized_order(self):
        r"""
        Return the generalized order used to break ties.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.generalized_order()
            Generalized monomial order in 2 variables using (lex, min)
        """
        return self._order

    def log_radii(self):
        r"""
        Return the log-radii of this algebra.
        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.log_radii()
            (-1, -1)

        """
        return self._log_radii

    def variable_names(self):
        r"""
        Return the variables names of this algebra.
        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.variable_names()
            ('x', 'y')

        """
        return self._names

    def _coerce_map_from_(self, R):
        r"""
        Currently, there are no inter-algebras coercions, we only coerce from
        the coefficient field or the corresponding term monoid.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: 2*x #indirect doctest
            (2 + O(2^21))*x
        """
        if self._field.has_coerce_map_from(R):
            return True
        elif isinstance(R, PolyCircleTermMonoid):
            return True
        else:
            return False

    def field(self):
        r"""
        Return the coefficient field of this algebra (currently restricted to Qp).

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.field()
            2-adic Field with capped relative precision 20
        """
        return self._field

    def prime(self):
        r"""
        Return the prime number of this algebra. This
        is the prime of the coefficient field.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.prime()
            2
        """
        return self._field.prime()

    def gen(self, n=0):
        r"""
        Return the ``n``-th generator of this algebra.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.gen(1)
            (1 + O(2^20))*y
        """
        if n < 0 or n >= self._ngens:
            raise IndexError("Generator index out of range")
        return self._gens[n]

    def gens(self):
        r"""
        Return the list of generators of this algebra.
        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.gens()
            [(1 + O(2^20))*x, (1 + O(2^20))*y]
        """
        return self._gens

    def ngens(self):
        r"""
        Return the number of generators of this algebra.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.ngens()
            2

        """
        return len(self._gens)

    def zero(self):
        r"""
        Return the element 0 of this algebra.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.zero()

        """
        return self(0)

    def monoid_of_terms(self):
        r"""
        Return the monoid of terms for this algebra.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: A.monoid_of_terms()
            Monoid of terms for poly-circle algebra

        """
        return self._monoid_of_terms


class PolyCircleTerm(MonoidElement):
    def __init__(self, parent, coeff, exponent=None):
        MonoidElement.__init__(self, parent)
        field = parent._field
        if isinstance(coeff, PolyCircleTerm):
            if coeff.parent().variable_names() != self.parent().variable_names():
                raise ValueError("the variable names do not match")
            self._coeff = field(coeff._coeff)
            self._exponent = coeff._exponent
        else:
            self._coeff = field(coeff)
            if self._coeff.is_zero():
                raise TypeError("a term cannot be zero")
            self._exponent = ETuple([0] * parent.ngens())
        if exponent is not None:
            if not isinstance(exponent, ETuple):
                exponent = ETuple(exponent)
            self._exponent = self._exponent.eadd(exponent)
        if len(self._exponent) != parent.ngens():
            raise ValueError(
                "the length of the exponent does not match the number of variables"
            )

    def coefficient(self):
        r"""
        Return the coefficient of this term.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().coefficient()
            1 + O(2^20)

        """
        return self._coeff

    def exponent(self):
        r"""
        Return the exponent of this term.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().exponent()
            (0, -1)

        """
        return self._exponent

    def valuation(self):
        r"""
        Return the valuation of this term.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().valuation()
            -1
        """
        if self._coeff == 0:
            return Infinity
        return self._coeff.valuation() - self._exponent.dotprod(
            self.parent()._log_radii
        )

    def monomial(self):
        r"""
        Return this term divided by its coefficient.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().monomial()
            (1 + O(2^20))*y^-1
        """
        coeff = self.parent()._field(1)
        exponent = self._exponent
        return self.__class__(self.parent(), coeff, exponent)

    def _mul_(self, other):
        r"""
        Return the multiplication of this term by ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y
            sage: f.leading_term()*g.leading_term()
            (1 + O(2^20))*x*y^-1
        """
        coeff = self._coeff * other._coeff
        exponent = self._exponent.eadd(other._exponent)
        return self.__class__(self.parent(), coeff, exponent)

    def _to_poly(self):
        r"""
        Return this term as a polynomial.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: type(f.leading_term()._to_poly())
            <class 'sage.rings.polynomial.laurent_polynomial_mpair.LaurentPolynomial_mpair'>
        """
        R = self.parent()._parent_algebra._polynomial_ring
        return self._coeff * R.monomial(self._exponent)

    def _divides(self, t):
        r"""
        Return this term divided by ``t``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y
            sage: a = f.leading_term()
            sage: b = g.leading_term()
            sage: a._divides(b)
            (1 + O(2^20))*x^-1*y^-1
        """
        coeff = self._coeff / t._coeff
        exponent = self._exponent.esub(t._exponent)
        return self.__class__(self.parent(), coeff=coeff, exponent=exponent)

    def _cmp_(self, other, i=None):
        r"""
        Compare this term with ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: a._cmp_(a)
            0
        """
        ov = other.valuation()
        sv = self.valuation()
        c = ov - sv
        if c == 0 and self._exponent != other._exponent:
            if (
                self.parent()._order.greatest_tuple(self._exponent, other._exponent)
                == self._exponent
            ):
                c = 1
            else:
                c = -1
        return c

    def same(self, t):
        r"""
        Test wether this term is the same as ``t``, meaning
        it has the same coefficient and exponent.
        This is not the same as equality.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: a.same(a)
            True
        """
        return self._coeff == t._coeff and self._exponent == t._exponent

    def __eq__(self, other):
        r"""
        Test quality of this term with ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: a == a
            True
            sage: a == x.leading_term()
            False
        """
        return self._cmp_(other) == 0

    def __ne__(self, other):
        r"""
        Test inquality of this term with ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: a != a
            False
            sage: a != x.leading_term()
            True
        """
        return not self == other

    def __lt__(self, other):
        r"""
        Test if this term is less than ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: b = x.leading_term()
            sage: a < b
            False
        """
        return self._cmp_(other) < 0

    def __gt__(self, other):
        r"""
        Test if this term is greater than  ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: b = x.leading_term()
            sage: a > b
            True
        """
        return self._cmp_(other) > 0

    def __le__(self, other):
        r"""
        Test if this term is less than or equal than  ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: b = x.leading_term()
            sage: a <= b
            False
        """
        return self._cmp_(other) <= 0

    def __ge__(self, other):
        r"""
        Test if this term is greater than or equal than ``other``.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: b = x.leading_term()
            sage: a >= b
            True
        """
        return self._cmp_(other) >= 0

    def _repr_(self):
        r"""
        Return a string representing this term.

        EXAMPLES::

            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 ); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f
            (1 + O(2^20))*y + (1 + O(2^20))*x*y + (2 + O(2^21))*x
        """
        parent = self.parent()
        if self._coeff._is_atomic() or (-self._coeff)._is_atomic():
            s = repr(self._coeff)
            if s == "1":
                s = ""
        else:
            s = "(%s)" % self._coeff
        for i in range(parent._ngens):
            if self._exponent[i] == 1:
                s += "*%s" % parent._names[i]
            elif self._exponent[i] > 1 or self._exponent[i] < 0:
                s += "*%s^%s" % (parent._names[i], self._exponent[i])
        if s[0] == "*":
            return s[1:]
        else:
            return s


class PolyCircleTermMonoid(Monoid_class, UniqueRepresentation):
    Element = PolyCircleTerm

    def __init__(self, parent_algebra):
        names = parent_algebra.variable_names()
        Monoid_class.__init__(self, names=names)
        self._field = parent_algebra._field
        self._names = names
        self._ngens = len(names)
        self._order = parent_algebra._order
        self._parent_algebra = parent_algebra
        self._log_radii = parent_algebra.log_radii()

    def field(self):
        r"""
        Return the coefficient field of this term monoid.
        This is the same as for the parent algebra.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: M.field()
            2-adic Field with capped relative precision 20
        """
        return self._field

    def prime(self):
        r"""
        Return the prime number of this term monoid. This
        is the prime of the coefficient field.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: M.prime()
            2
        """
        return self._field.prime()

    def parent_algebra(self):
        r"""
        Return the parent algebra of this term monoid.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: PA = M.parent_algebra()
        """
        return self._parent_algebra

    def generalized_order(self):
        r"""
        Return the generalized order used to break ties.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: M.generalized_order()
            Generalized monomial order in 2 variables using (lex, min)
        """
        return self._order

    def variable_names(self):
        r"""
        Return the variables names of this term monoid.
        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: M.variable_names()
            ('x', 'y')

        """
        return self._names

    def ngens(self):
        r"""
        Return the number of generators of this term monoid.

        EXAMPLES::
            sage: A = PolyCircleAlgebra(Qp(2), (-1,-1), "x,y", prec=10 )
            sage: M = A.monoid_of_terms()
            sage: M.ngens()
            2

        """
        return self._ngens

    def _repr_(self):
        r"""
        Return a string representation of this term monoid.
        """
        return "Monoid of terms for poly-circle algebra"

    def _coerce_map_from_(self, R):
        r"""
        Currently, we only coerce from coefficient ring.
        """
        if self._field.has_coerce_map_from(R):
            return True
        else:
            return False
