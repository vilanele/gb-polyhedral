from sage.rings.integer_ring import ZZ
from sage.modules.free_module_element import vector
from sage.rings.polynomial.term_order import TermOrder
from sage.matrix.constructor import matrix
from sage.structure.sage_object import SageObject
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polydict import PolyDict
from sage.modules.free_module_element import vector
from sage.geometry.polyhedral_complex import Polyhedron
from sage.rings.polynomial.polydict import ETuple
from sage.rings.infinity import Infinity
from sage.rings.polynomial.laurent_polynomial_ring import LaurentPolynomialRing
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Parent
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.structure.category_object import normalize_names
from sage.categories.algebras import Algebras
from sage.modules.free_module_element import vector
from sage.geometry.polyhedral_complex import Polyhedron
from sage.geometry.fan import Fan
from sage.geometry.cone import Cone
from sage.rings.polynomial.polydict import ETuple
from sage.rings.infinity import Infinity
from sage.plot.plot import plot
from sage.rings.infinity import Infinity
from sage.structure.element import MonoidElement
from sage.rings.polynomial.polydict import ETuple
from sage.monoids.monoid import Monoid_class
from sage.structure.unique_representation import UniqueRepresentation


class GeneralizedOrderPolyAnnulus(SageObject):
    def __init__(self, n, cones, group_order="lex", score_function="norm"):
        r"""
        Create a generalized monomial order for a poly-annulus polyhedral algebra
        in ``n`` varaibles with optional group order and score function.

        INPUT:

        - ``n`` -- the number of variables
        - ``group_order`` (default: ``lex``) -- the name of a group order on `\ZZ^n`, choices are: "lex"
        - ``score_function`` (default: ``norm``) -- the name of a score function, choices are: "norm"

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: GeneralizedOrderPolyAnnulus(2, cones)
            Generalized monomial order for poly-annulus in 2 variables using (lex, norm)
        """
        self._n = n
        self._n_cones = 2**self._n
        # Build cones
        self._cones = cones
        if group_order in ["lex"]:
            self._group_order = TermOrder(name=group_order)
        else:
            raise ValueError("Available group order are: 'lex'")
        # Set score function. Add more here.
        if score_function == "norm":
            self._score_function = norm_score_function
        else:
            raise ValueError("Available score function are: 'norm'")
        # Store names
        self._group_order_name = group_order
        self._score_function_name = score_function

    def _repr_(self):
        r"""
        TESTS::
            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: GeneralizedOrderPolyAnnulus(2,cones)._repr_()
            'Generalized monomial order for poly-annulus in 2 variables using (lex, norm)'
        """
        group = self._group_order_name
        function = self._score_function_name
        n = str(self._n)
        return (
            "Generalized monomial order for poly-annulus in %s variables using (%s, %s)"
            % (n, group, function)
        )

    def __hash__(self):
        r"""
        Return the hash of self. It depends on the number of variables, the group_order and the score function.
        """
        return hash(self._group_order_name + self._score_function_name + str(self._n))

    def n_cones(self):
        r"""
        Return the number of cones (which is 2 to the power of the number of variables).

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.n_cones()
            4
        """
        return self._n_cones

    def cones(self):
        r"""
        Return the list of matrices containing the generators of the cones.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.cones()
            [
            [1 0]  [-1  0]  [-1  0]  [ 1  0]
            [0 1], [ 0 -1], [ 0  1], [ 0 -1]
            ]
        """
        return self._cones

    def cone(self, i):
        r"""
        Return the matrix whose columns are the generators of the ``i``-th cone.

        INPUT:

        - `i` -- an integer, a cone index

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.cone(0)
            [1 0]
            [0 1]
        """
        if i < 0 or i > self._n:
            raise IndexError("cone index out of range")
        return self._cones[i]

    def group_order(self):
        r"""
        Return the underlying group order.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.group_order()
            Lexicographic term order
        """
        return self._group_order

    def group_order_name(self):
        r"""
        Return the name of the underlying group order.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.group_order_name()
            'lex'
        """
        return self._group_order_name

    def score_function_name(self):
        r"""
        Return the name of the underlying score function.

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.score_function_name()
            'norm'
        """
        return self._score_function_name

    def score(self, t):
        r"""
        Compute the score of a tuple ``t``.

        INPUT:

        - `t` -- a tuple of integers

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.score((-2,3))
            5
        """
        return self._score_function(t)

    def compare(self, a, b):
        r"""
        Return 1, 0 or -1 whether tuple ``a`` is greater than, equal or less than to tuple ``b`` respectively.

        INPUT:

        - `a` and `b` -- two tuples of integers

        EXAMPLES::

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.compare((1,2), (3,-4))
            -1
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

            sage: from sage.rings.polytopal.GeneralizedOrders import GeneralizedOrderPolyAnnulus
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.greatest_tuple((1,-1),(2,-3),(3,-4))
            (3, -4)
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
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.translate_to_cone(2, [(1,2),(-2,-3),(1,-4)])
            (-1, 4)
        """
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
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: L = [(1,2), (-2,-2), (-4,5), (5,-6)]
            sage: t = order.greatest_tuple_for_cone(1, *L);t
            (-2, -2)

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
            sage: cones = []
            sage: cones.append(matrix(ZZ,[[1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,-1]]))
            sage: cones.append(matrix(ZZ,[[-1,0],[0,1]]))
            sage: cones.append(matrix(ZZ,[[1,0],[0,-1]]))
            sage: order = GeneralizedOrderPolyAnnulus(2, cones)
            sage: order.is_in_cone(0, (1,2))
            True
            sage: order.is_in_cone(0,(-2,3))
            False
        """
        return all(c >= 0 for c in self.cone(i) * vector(t))


def norm_score_function(t):
    return sum([abs(c) for c in t])


class PolyAnnulusAlgebraElement(CommutativeAlgebraElement):
    def __init__(self, parent, x=None, prec=None):
        CommutativeAlgebraElement.__init__(self, parent)
        if isinstance(x, PolyAnnulusAlgebraElement):
            self._poly = x._poly
            if prec is None:
                self._prec = x._prec
            else:
                self._prec = prec
        elif isinstance(x, PolyAnnulusTerm):
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y + y^-1 + 1*x^-3
            sage: f.terms()
            [(1 + O(2^20))*x^-3,
            (1 + O(2^20))*x*y,
            (1 + O(2^20))*y^-1,
            (1 + O(2^20))*y,
            (2 + O(2^21))*x]
        """
        terms = [
            PolyAnnulusTerm(self.parent()._monoid_of_terms, coeff=c, exponent=e)
            for e, c in self._poly.dict().items()
        ]
        return sorted(terms, reverse=True)

    def leading_term(self, i=None):
        r"""
        Return the leading term of this series if ``i`` is None, otherwise the
        leading term for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term()
            (1 + O(2^20))*x*y
            sage: f.leading_term(2)
            (1 + O(2^20))*y^-1

        """
        if i is None:
            return self.terms()[0]

        ini = self.initial(i)._poly
        le = self.parent()._order.greatest_tuple_for_cone(i, *ini.exponents())
        return PolyAnnulusTerm(self.parent()._monoid_of_terms, ini[le], le)

    def leading_monomial(self, i=None):
        r"""
        Return the leading monomial of this series if ``i`` is None, otherwise the
        leading monomial for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_monomial()
            (1 + O(2^20))*x*y
            sage: f.leading_monomial(2)
            (1 + O(2^20))*y^-1
        """
        return self.leading_term(i).monomial()

    def leading_coefficient(self, i=None):
        r"""
        Return the leading coefficient of this series if ``i`` is None, otherwise the
        leading coefficient for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_coefficient()
            1 + O(2^20)
            sage: f.leading_coefficient(2)
            1 + O(2^20)
        """
        return self.leading_term(i).coefficient()

    def leadings(self, i=None):
        r"""
        Return a list containing the leading coefficient, monomial and term
        of this series if ``i`` is None, or th same for the ``i``-th vertex otherwise.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leadings()
            (1 + O(2^20), (1 + O(2^20))*x*y, (1 + O(2^20))*x*y)
        """
        lt = self.leading_term(i)
        return (lt.coefficient(), lt.monomial(), lt)

    def initial(self, i=None):
        r"""
        Return the initial part of this series if ``i`` is None, otherwise
        the initial part for the ``i``-the vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.initial()
            (1 + O(2^20))*x*y
            sage: f.initial(2)
            (1 + O(2^20))*y^-1
        """
        terms = self.terms()
        if i is None:
            v = self.valuation()
        else:
            v = min([t.valuation(i) for t in terms])
        return sum(
            [self.__class__(self.parent(), t) for t in terms if t.valuation(i) == v]
        )

    def valuation(self, i=None):
        r"""
        Return the valuation of this series if ``i`` is None
        (which is the minimum over the valuations at each vertex),
        otherwise the valuation for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.valuation()
            -2
        """
        if self.is_zero():
            return Infinity
        return self.leading_term(i).valuation(i)

    def _normalize(self):
        r"""
        Normalize this series.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.add_bigoh(2) # Indirect doctest
            (1 + O(2^4))*x*y + (1 + O(2^3))*y^-1 + (2 + O(2^3))*x + OO(2)
        """
        if self._prec == Infinity:
            return
        terms = []
        for t in self.terms():
            exponent = t.exponent()
            coeff = t.coefficient()
            v = max(
                [
                    ETuple(tuple(vertex)).dotprod(exponent)
                    for vertex in self.parent().vertices()
                ]
            )
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.add_bigoh(2) # Indirect doctest
            (1 + O(2^4))*x*y + (1 + O(2^3))*y^-1 + (2 + O(2^3))*x + OO(2)
        """
        return self.parent()(self, prec=prec)

    def is_zero(self):
        r"""
        Test if this series is zero.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = 3*x*y
            sage: f + g
            (1 + O(2^20))*y^-1 + (2 + O(2^21))*x + (2^2 + O(2^20))*x*y
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = -3*x*y
            sage: f - g
            (1 + O(2^20))*y^-1 + (2 + O(2^21))*x + (2^2 + O(2^20))*x*y
        """
        prec = min(self._prec, other._prec)
        ans = self.__class__(self.parent(), self._poly - other._poly, prec=prec)
        ans._normalize()
        return ans

    def _mul_(self, other):
        r"""
        Return the multiplication of this series and ``other`.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = 3*x*y
            sage: f*g
            (1 + 2 + O(2^20))*x^2*y^2 + (2 + 2^2 + O(2^21))*x^2*y + (1 + 2 + O(2^20))*x
        """
        a = self.valuation() + other._prec
        b = other.valuation() + self._prec
        prec = min(a, b)
        ans = self.__class__(self.parent(), self._poly * other._poly, prec=prec)
        ans._normalize()
        return ans

    def _div_(self, other):
        r"""
        Return the multiplication of this series and ``other`.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

    def leading_module_polyhedron(self, i=None, backend="normaliz"):
        r"""
        Return the polyhedron whose intger points are exactly the exponents of all
        leadings terms of multiple of this series that lie in the restricted
        ``i``-th cone.

        Is ``i`` is None, return a list with all polyhedron for all vertices.

        The default backend for the polyhedron class is normaliz. It is needed
        for the method least_common_multiples to retrieve integral generators.
        If you only wan't to construct the polyhedron, any backend is ok.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_module_polyhedron(0)
            A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 2 vertices and 2 rays

        """
        if i is None:
            return [
                self.leading_module_polyhedron(i, backend)
                for i in range(self.parent()._nvertices)
            ]
        vertices = self.parent()._vertices

        ieqs = []
        shift = vector(self.leading_monomial(i).exponent())
        for j in [k for k in range(len(vertices)) if k != i]:
            v = vertices[j]
            a = tuple(vector(vertices[i]) - vector(v))
            c = self.valuation(i) - self.valuation(j)
            if j < i:
                c += 1
            ieqs.append((-c,) + a)

        P1 = Polyhedron(ieqs=ieqs)
        return Polyhedron(
            rays=self.parent().polyhedron_at_vertex(i).rays(),
            vertices=[list(vector(v) + shift) for v in P1.vertices()],
            backend=backend,
        )

    def least_common_multiples(self, other, i):
        r"""
        Return the finite sets of monomials which are the lcms of this series and ``other``
        for the ``i``-th cone.

        This method require the backend normaliz.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x - y
            sage: f.least_common_multiples(g,0)
            [(1 + O(2^20))*x^-1*y^-1, (1 + O(2^20))*y^-2]
        """
        return [
            self.parent()._monoid_of_terms(1, exponent=tuple(e))
            for e in self.leading_module_polyhedron(i)
            .intersection(other.leading_module_polyhedron(i))
            .integral_points_generators()[0]
        ]

    def critical_pair(self, other, i, lcm):
        r"""
        Return the S-pair of this series and ``other`` for the ``i``-th
        cone given a leading common multiple ``lcm``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y^-1
            sage: lcm = f.least_common_multiples(g,1)[0]
            sage: f.critical_pair(g, 1, lcm)
            (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7 + 2^8 + 2^9 + 2^10 + 2^11 + 2^12 + 2^13 + 2^14 + 2^15 + 2^16 + 2^17 + 2^18 + 2^19 + O(2^20))*x^-1*y + (1 + O(2^20))*x^-1 + (2 + O(2^21))*y
        """
        lcf, lmf, _ = self.leadings(i)
        lcg, lmg, _ = other.leadings(i)

        return lcg * lcm._divides(lmf) * self - lcf * lcm._divides(lmg) * other

    def all_critical_pairs(self, other, i=None):
        r"""
        Return all S-pairs of this series and ``other`` for the ``i``-th cone,
        or all S-pairs for all cones if ``i`` is None.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y^-1
            sage: f.all_critical_pairs(g,0)
            [(1 + O(2^20))*y + (1 + O(2^20)), (1 + O(2^20))*x*y^-1 + (1 + O(2^20))*x]
        """
        if i is None:
            return [
                item
                for j in range(self.parent()._nvertices)
                for item in self.all_critical_pairs(other, j)
            ]

        lcms = self.least_common_multiples(other, i)
        return [self.critical_pair(other, i, lcm) for lcm in lcms]


class PolyAnnulusAlgebra(Parent, UniqueRepresentation):
    r"""
    Parent class for series with finite precision in a polyhedral algebra with power series converging on a poly-annulus.


    EXAMPLES::

       sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
       sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
    """

    Element = PolyAnnulusAlgebraElement

    def __init__(
        self,
        field,
        polyhedron,
        names,
        score_function="norm",
        group_order="lex",
        prec=10,
    ):
        self._field = field
        self.element_class = PolyAnnulusAlgebraElement
        self._prec = prec
        self._polyhedron = polyhedron
        self._vertices = polyhedron.vertices()
        self._nvertices = len(self._vertices)
        self._names = normalize_names(-1, names)
        self._polynomial_ring = LaurentPolynomialRing(field, names=names)
        self._gens = [
            self.element_class(self, g, prec=Infinity)
            for g in self._polynomial_ring.gens()
        ]
        self._ngens = len(self._gens)
        self._cones = self._build_cones()
        self._order = GeneralizedOrderPolyAnnulus(
            self._ngens,
            self._cones,
            score_function=score_function,
            group_order=group_order,
        )
        self._monoid_of_terms = PolyAnnulusTermMonoid(self)
        Parent.__init__(self, names=names, category=Algebras(self._field).Commutative())

    def _build_cones(self):
        cones = [self.cone_at_vertex(i) for i in range(self._nvertices)]
        decompo_cones = []
        for cone in cones:
            m = matrix.zero(ZZ, self._ngens)
            for i, ray in enumerate(cone.rays()):
                m.set_column(i, vector(tuple(ray)))
            decompo_cones.append(m)
        return decompo_cones

    def generalized_order(self):
        r"""
        Return the term order used to break ties.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.generalized_order()
            Generalized monomial order for poly-annulus in 2 variables using (lex, norm)
        """
        return self._order

    def variable_names(self):
        r"""
        Return the variables names of this algebra.
        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.variable_names()
            ('x', 'y')

        """
        return self._names

    def _coerce_map_from_(self, R):
        r"""
        Currently, there are no inter-algebras coercions, we only coerce from
        the coefficient field or the corresponding term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x, y = A.gens()
            sage: 2*x #indirect doctest
            (2 + O(2^21))*x
        """
        if self._field.has_coerce_map_from(R):
            return True
        elif isinstance(R, PolyAnnulusTermMonoid):
            return True
        else:
            return False

    def field(self):
        r"""
        Return the coefficient field of this algebra (currently restricted to Qp).

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.field()
            2-adic Field with capped relative precision 20
        """
        return self._field

    def prime(self):
        r"""
        Return the prime number of this algebra. This
        is the prime of the coefficient field.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.prime()
            2
        """
        return self._field.prime()

    def gen(self, n=0):
        r"""
        Return the ``n``-th generator of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
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
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.gens()
            [(1 + O(2^20))*x, (1 + O(2^20))*y]
        """
        return self._gens

    def ngens(self):
        r"""
        Return the number of generators of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.ngens()
            2

        """
        return len(self._gens)

    def zero(self):
        r"""
        Return the element 0 of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.zero()

        """
        return self(0)

    def polyhedron(self):
        r"""
        Return the defining polyhedron of this algebra.

        EXAMPLES:
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.polyhedron()
            A 2-dimensional polyhedron in ZZ^2 defined as the convex hull of 4 vertices
        """
        return self._polyhedron

    def vertices(self):
        r"""
        Return the list of vertices of the defining polyhedron of this algebra.

        EXAMPLES:
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.vertices()
            (A vertex at (-1, -1),
            A vertex at (-1, 1),
            A vertex at (1, -1),
            A vertex at (1, 1))

        """
        return self._vertices

    def monoid_of_terms(self):
        r"""
        Return the monoid of terms for this algebra.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.monoid_of_terms()
            monoid of terms

        """
        return self._monoid_of_terms

    def vertex(self, i=0, etuple=False):
        r"""
        Return the ``i``-th vertex of the defining polyhedron (indexing starts at zero).
        The flag ``etuple`` indicates whether the result should be returned as an ``ETuple``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.vertex(0)
            A vertex at (-1, -1)
            sage: A.vertex(1, etuple=True)
            (-1, 1)
        """
        if i < 0 or i >= self._nvertices:
            raise IndexError("Vertex index out of range")
        if etuple:
            return ETuple(tuple(self.vertex(i)))
        return self._vertices[i]

    def vertex_indices(self):
        r"""
        Return the list of verex indices. Indexing starts at zero.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.vertex_indices()
            [0, 1, 2, 3]
        """
        return [i for i in range(self._nvertices)]

    def cone_at_vertex(self, i=0):
        r"""
        Return the cone for the ``i``-th vertex of this algebra.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: C = A.cone_at_vertex(0); C
            2-d cone in 2-d lattice N
            sage: C.rays()
            N( 0, -1),
            N(-1, 0)
            in 2-d lattice N
        """
        if i < 0 or i >= self._nvertices:
            raise IndexError("Vertex index out of range")
        return Cone(self.polyhedron_at_vertex(i))

    def polyhedron_at_vertex(self, i=0):
        r"""
        Return the the cone for the ``i``-th vertex as a polyhedron.
        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: C = A.polyhedron_at_vertex(0); C
            A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 1 vertex and 2 rays
        """
        if i < 0 or i >= self._nvertices:
            raise IndexError("Vertex index out of range")
        ieqs = []
        for v in self._vertices[:i] + self._vertices[i + 1 :]:
            a = tuple(vector(self._vertices[i]) - vector(v))
            ieqs.append((0,) + a)

        return Polyhedron(ieqs=ieqs)

    def plot_leading_monomials(self, G):
        r"""
        Experimental. Plot in the same graphics all polyhedrons containing
        the leading monomials of the series in the list ``G``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: A.plot_leading_monomials([2*x + y, x**2 - 3*y**3])
            Graphics object consisting of 35 graphics primitives
        """
        plots = []
        for g in G:
            plots += [plot(p) for p in g.leading_module_polyhedron()]
        return sum(plots)

    def fan(self):
        r"""
        Return the fan whose maximal cones are the cones at each vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: A.fan()
            Rational polyhedral fan in 2-d lattice N
        """
        cones = []
        for i in range(len(self._vertices)):
            cones.append(Cone(self.cone_at_vertex(i)))

        return Fan(cones=cones)


class PolyAnnulusTerm(MonoidElement):
    def __init__(self, parent, coeff, exponent=None):
        MonoidElement.__init__(self, parent)
        field = parent._field
        if isinstance(coeff, PolyAnnulusTerm):
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
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().coefficient()
            1 + O(2^20)

        """
        return self._coeff

    def exponent(self):
        r"""
        Return the exponent of this term.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().exponent()
            (1, 1)

        """
        return self._exponent

    def valuation(self, i=None, vertex_index=False):
        r"""
        Return the valuation of this term if ``i`` is None
        (which is the minimum over the valuations at each vertex),
        otherwise the valuation for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().valuation()
            -2
        """
        if self._coeff == 0:
            return Infinity
        if i is None:
            m = min([(self.valuation(j), j) for j in self.parent().vertex_indices()])
            if vertex_index:
                return m
            else:
                return m[0]
        return self._coeff.valuation() - self._exponent.dotprod(
            self.parent().vertex(i, etuple=True)
        )

    def monomial(self):
        r"""
        Return this term divided by its coefficient. x.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: f.leading_term().monomial()
            (1 + O(2^20))*x*y
        """
        coeff = self.parent()._field(1)
        exponent = self._exponent
        return self.__class__(self.parent(), coeff, exponent)

    def _mul_(self, other):
        r"""
        Return the multiplication of this term by ``other``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y
            sage: f.leading_term()*g.leading_term()
            (1 + O(2^20))*x*y^2
        """
        coeff = self._coeff * other._coeff
        exponent = self._exponent.eadd(other._exponent)
        return self.__class__(self.parent(), coeff, exponent)

    def _to_poly(self):
        r"""
        Return this term as a polynomial.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: g = x + y
            sage: a = f.leading_term()
            sage: b = g.leading_term()
            sage: a._divides(b)
            (1 + O(2^20))*x
        """
        coeff = self._coeff / t._coeff
        exponent = self._exponent.esub(t._exponent)
        return self.__class__(self.parent(), coeff=coeff, exponent=exponent)

    def _cmp_(self, other, i=None):
        r"""
        Compare this term with ``other``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
            sage: f = 2*x + y^-1 + x*y
            sage: a = f.leading_term()
            sage: a._cmp_(a)
            0
        """
        ov, oi = other.valuation(vertex_index=True)
        sv, si = self.valuation(vertex_index=True)
        c = ov - sv
        if c == 0:
            c = oi - si
        if c == 0 and self._exponent != other._exponent:
            if (
                self._order.greatest_tuple(self._exponent, other._exponent)
                == self._exponent
            ):
                c = 1
            else:
                c = -1
        return c

    def same(self, t):
        r"""
        Test wether this termù is the same as ``t``, meaning
        it has the same coefficient and exponent.
        This is not the same as equality.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f
            (1 + O(2^20))*y + (2 + O(2^21))*x + (1 + O(2^20))*x*y
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


class PolyAnnulusTermMonoid(Monoid_class, UniqueRepresentation):
    Element = PolyAnnulusTerm

    def __init__(self, parent_algebra):
        names = parent_algebra.variable_names()
        Monoid_class.__init__(self, names=names)
        self._field = parent_algebra._field
        self._names = names
        self._ngens = len(names)
        self._vertices = parent_algebra._vertices
        self._nvertices = parent_algebra._nvertices
        self._order = parent_algebra._order
        self._parent_algebra = parent_algebra

    def _coerce_map_from_(self, R):
        r"""
        Currently, we only coerce from coefficient ring.
        """
        if self._field.has_coerce_map_from(R):
            return True
        else:
            return False

    def _repr_(self):
        r"""
        Return a string representation of this term monoid.
        """
        return "monoid of terms"

    def vertex_indices(self):
        r"""
        Return the list of verex indices. Indexing starts at zero.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.vertex_indices()
            [0, 1, 2, 3]
        """
        return [i for i in range(self._nvertices)]

    def field(self):
        r"""
        Return the coefficient field of this term monoid.
        This is the same as for the parent algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
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
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.prime()
            2
        """
        return self._field.prime()

    def parent_algebra(self):
        r"""
        Return the parent algebra of this term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: PA = M.parent_algebra()
        """
        return self._parent_algebra

    def generalized_order(self):
        r"""
        Return the term order used to break ties.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.generalized_order()
            Generalized monomial order for poly-annulus in 2 variables using (lex, norm)
        """
        return self._order

    def variable_names(self):
        r"""
        Return the variables names of this term monoid.
        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.variable_names()
            ('x', 'y')

        """
        return self._names

    def vertices(self):
        r"""
        Return the list of vertices of the defining polyhedron of the parent algebra
        of this term monoid.

        EXAMPLES:
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.vertices()
            (A vertex at (-1, -1),
            A vertex at (-1, 1),
            A vertex at (1, -1),
            A vertex at (1, 1))
        """
        return self._vertices

    def vertex(self, i=0, etuple=False):
        r"""
        Return the ``i``-th vertex of the defining polyhedron (indexing starts at zero)
        of the parent algebra.
        The flag ``etuple`` indicates whether the result should be returned as an ``ETuple``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.vertex(0)
            A vertex at (-1, -1)
            sage: M.vertex(1, etuple=True)
            (-1, 1)
        """
        return self._parent_algebra.vertex(i, etuple=etuple)

    def ngens(self):
        r"""
        Return the number of generators of this term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(1,1), (1,-1), (-1,-1), (-1,1)])
            sage: A = PolyAnnulusAlgebra(Qp(2), P, "x,y")
            sage: M = A.monoid_of_terms()
            sage: M.ngens()
            2

        """
        return self._ngens
