from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.unique_representation import UniqueRepresentation
from sage.structure.parent import Parent
from sage.structure.category_object import normalize_names
from sage.categories.algebras import Algebras
from sage.modules.free_module_element import vector
from sage.geometry.polyhedral_complex import Polyhedron
from sage.geometry.fan import Fan
from sage.geometry.cone import Cone
from sage.rings.polynomial.polydict import ETuple
from sage.rings.infinity import Infinity
from sage.plot.plot import plot
from sage.structure.element import CommutativeAlgebraElement
from sage.rings.polynomial.polydict import PolyDict
from sage.modules.free_module_element import vector
from sage.geometry.polyhedral_complex import Polyhedron
from sage.rings.polynomial.polydict import ETuple
from sage.rings.infinity import Infinity
from sage.monoids.monoid import Monoid_class
from sage.structure.unique_representation import UniqueRepresentation
from sage.rings.infinity import Infinity
from sage.structure.element import MonoidElement
from sage.rings.polynomial.polydict import ETuple

class PositiveExponentsAlgebraElement(CommutativeAlgebraElement):
    def __init__(self, parent, x=None, prec=None):
        CommutativeAlgebraElement.__init__(self, parent)
        if isinstance(x, PositiveExponentsAlgebraElement):
            self._poly = x._poly
            if prec is None:
                self._prec = x._prec
            else:
                self._prec = prec
        elif isinstance(x, PositiveExponentsTerm):
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.terms()
            [(1 + O(2^20))*y, (2 + O(2^21))*x, (1 + O(2^20))*x*y]
        """
        terms = [
            PositiveExponentsTerm(self.parent()._monoid_of_terms, coeff=c, exponent=e)
            for e, c in self._poly.dict().items()
        ]
        return sorted(terms, reverse=True)

    def leading_term(self, i=None):
        r"""
        Return the leading term of this series if ``i`` is None, otherwise the
        leading term for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_term()
            (1 + O(2^20))*y
            sage: f.leading_term(1)
            (2 + O(2^21))*x

        """
        if i is None:
            return self.terms()[0]

        lt = self.initial(i)._poly.leading_term()
        return PositiveExponentsTerm(
            self.parent()._monoid_of_terms, lt.coefficients()[0], lt.exponents()[0]
        )

    def leading_monomial(self, i=None):
        r"""
        Return the leading monomial of this series if ``i`` is None, otherwise the
        leading monomial for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_monomial()
            (1 + O(2^20))*y
            sage: f.leading_monomial(1)
            (1 + O(2^20))*x
        """
        return self.leading_term(i).monomial()

    def leading_coefficient(self, i=None):
        r"""
        Return the leading coefficient of this series if ``i`` is None, otherwise the
        leading coefficient for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_coefficient()
            1 + O(2^20)
            sage: f.leading_coefficient(1)
            2 + O(2^21)
        """
        return self.leading_term(i).coefficient()

    def leadings(self, i=None):
        r"""
        Return a list containing the leading coefficient, monomial and term
        of this series if ``i`` is None, or th same for the ``i``-th vertex otherwise.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leadings()
            (1 + O(2^20), (1 + O(2^20))*y, (1 + O(2^20))*y)
        """
        lt = self.leading_term(i)
        return (lt.coefficient(), lt.monomial(), lt)

    def initial(self, i=None):
        r"""
        Return the initial part of this series if ``i`` is None, otherwise
        the initial part for the ``i``-the vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.initial()
            (1 + O(2^20))*y
            sage: f.initial(1)
            (1 + O(2^20))*y + (2 + O(2^21))*x
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.valuation()
            1
        """
        if self.is_zero():
            return Infinity
        return self.leading_term(i).valuation(i)

    def _normalize(self):
        r"""
        Normalize this series.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.add_bigoh(2) # Indirect doctest
            (1 + O(2))*y + OO(2)
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.add_bigoh(2)
            (1 + O(2))*y + OO(2)
        """
        return self.parent()(self, prec=prec)

    def is_zero(self):
        r"""
        Test if series is zero.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = 3*x*y
            sage: f + g
            (1 + O(2^20))*y + (2 + O(2^21))*x + (2^2 + O(2^20))*x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x + y
            sage: f - g
            (1 + O(2^20))*x + (1 + O(2^20))*x*y
        """
        prec = min(self._prec, other._prec)
        ans = self.__class__(self.parent(), self._poly - other._poly, prec=prec)
        ans._normalize()
        return ans

    def _mul_(self, other):
        r"""
        Return the multiplication of this series and ``other`.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x + y
            sage: f*g
            (1 + O(2^20))*y^2 + (1 + 2 + O(2^20))*x*y + (2 + O(2^21))*x^2 + (1 + O(2^20))*x*y^2 + (1 + O(2^20))*x^2*y
        """
        a = self.valuation() + other._prec
        b = other.valuation() + self._prec
        prec = min(a, b)
        ans = self.__class__(self.parent(), self._poly * other._poly, prec=prec)
        ans._normalize()
        return ans

    def _neg_(self):
        r"""
        Return the negation of this series.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
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

    def _quo_rem(self, divisors, quotients=False, verbose=False):
        r"""
        Perform the reduction of ``self`` by quotients. For this method to terminate,
        it is necessary that ``self`` have finite precision. If not, we use
        the precision cap of the parent algebra.
        The flag ``quotients`` indicate wether to return the quotients with the remainder.
        The flag ``verbose`` trigger some  basic traceback.

        """
        if self.is_zero():
            raise ValueError("Quo_rem for zero")

        divisors = [d for d in divisors if not d.is_zero()]
        zero = self.parent().zero()
        remainder = zero
        Q = [zero for _ in range(len(divisors))]

        f = self.parent()(self)
        if f._prec is Infinity:
            f.add_bigoh(f.parent()._prec)
        # print(f)

        if verbose:
            print("Starting main loop ...")
        while not f.is_zero():
            if verbose:
                print("NNEWWWWWWWWWWW LOOOOOOOOP")
                print(f)
                print(f._poly)
                print(f.is_zero())
            ltf = f.leading_term()
            v, i = ltf.valuation(vertex_index=True)
            if verbose:
                print(f"leading: {ltf}, valuation: {v}, index: {i}")
            found_reducer = False
            if verbose:
                print("Looking for reducer ...")
            for k in range(len(divisors)):
                d = divisors[k]
                ltig = d.leading_term(i)
                if verbose:
                    print(f"Tryind divisor {d} with lt at {i} : {ltig}")
                    print("Testing terms divisibility ...")
                if ltf._is_divisible_by(ltig):
                    t = ltf._divides(ltig)
                    if verbose:
                        print(f"Divisibility is ok, quotient is: {t}")
                    td = t * d
                    if verbose:
                        print(f"Tentative reducer: {td}")
                    if td.leading_term().same(ltf):
                        if verbose:
                            print("Reducter found !!")
                            print(f"Valuation of f before : {f.valuation()} ")
                            print(f"Poly of f before: {f._poly}")
                            print(f"TD = {td}")
                        found_reducer = True
                        f = f - td
                        # print(f._prec)
                        if verbose:
                            print(f"Valuation of f after : {f.valuation()} ")
                            print(f"Poly of f after {f._poly}")
                        if quotients:
                            Q[k] += self.__class__(self.parent(), t._to_poly())
                        break

            if not found_reducer:
                if verbose:
                    print("No reducer found, moving to remainder ...")
                    print(f"Valuation of f before : {f.valuation()} ")
                remainder += self.__class__(self.parent(), ltf._to_poly())
                f = f - self.__class__(self.parent(), ltf._to_poly())
                if verbose:
                    print(f"Valuation of f after : {f.valuation()} ")
                    print(f"Current state of remainder : {remainder}")

        rprec = min(f._prec, *[q.valuation() + d._prec for q, d in zip(Q, divisors)])
        remainder = remainder.add_bigoh(rprec)
        if quotients:
            return Q, remainder
        else:
            return remainder

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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_module_polyhedron()
            [A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 2 vertices and 2 rays, A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 1 vertex and 2 rays]
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

        P1 = Polyhedron(ieqs=ieqs).intersection(self.parent()._quadrant)
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x - y
            sage: f.least_common_multiples(g,0)
            [(1 + O(2^20))*y]
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x - y
            sage: lcm = f.least_common_multiples(g,1)[0]
            sage: f.critical_pair(g, 1, lcm)
            (1 + 2 + O(2^20))*x^2*y + (1 + O(2^20))*x^3*y
        """
        lcf, lmf, _ = self.leadings(i)
        lcg, lmg, _ = other.leadings(i)

        return lcg * lcm._divides(lmf) * self - lcf * lcm._divides(lmg) * other

    def all_critical_pairs(self, other, i=None):
        r"""
        Return all S-pairs of this series and ``other`` for the ``i``-th cone,
        or all S-pairs for all cones if ``i`` is None.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x + y
            sage: f.all_critical_pairs(g,0)
            [(1 + O(2^20))*x + (1 + O(2^20))*x*y]
        """
        if i is None:
            return [
                item
                for j in range(self.parent()._nvertices)
                for item in self.all_critical_pairs(other, j)
            ]

        lcms = self.least_common_multiples(other, i)
        return [self.critical_pair(other, i, lcm) for lcm in lcms]

class PositiveExponentsAlgebra(Parent, UniqueRepresentation):
    r"""
    Parent class for series with finite precision in a polyhedral algebra with positivce exponents.
    """

    Element = PositiveExponentsAlgebraElement

    def __init__(self, field, polyhedron, names, order="degrevlex", prec=10):
        self.element_class = PositiveExponentsAlgebraElement
        self._field = field
        self._prec = prec
        self._polyhedron = polyhedron
        self._vertices = polyhedron.vertices()
        self._nvertices = len(self._vertices)
        self._names = normalize_names(-1, names)
        self._polynomial_ring = PolynomialRing(field, names=names, order=order)
        self._monoid_of_terms = PositiveExponentsTermMonoid(self)
        self._gens = [
            self.element_class(self, g, prec=Infinity)
            for g in self._polynomial_ring.gens()
        ]
        self._ngens = len(self._gens)
        from sage.geometry.cone_catalog import nonnegative_orthant

        self._quadrant = nonnegative_orthant(self._ngens).polyhedron()
        Parent.__init__(self, names=names, category=Algebras(self._field).Commutative())

    def term_order(self):
        r"""
        Return the term order used to break ties.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.term_order()
            Degree reverse lexicographic term order
        """
        return self._polynomial_ring.term_order()

    def variable_names(self):
        r"""
        Return the variables names of this algebra.
        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.variable_names()
            ('x', 'y')

        """
        return self._names

    def _coerce_map_from_(self, R):
        r"""
        Currently, there are no inter-algebras coercions, we only coerce from
        the coefficient field or the corresponding term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: 2*x #indirect doctest
            (2 + O(2^21))*x
            sage: x.leading_term()*(x+y)
            (1 + O(2^20))*x^2 + (1 + O(2^20))*x*y
        """
        if self._field.has_coerce_map_from(R):
            return True
        elif isinstance(R, PositiveExponentsTermMonoid):
            return True
        else:
            return False

    def field(self):
        r"""
        Return the coefficient field of this algebra (currently restricted to Qp).

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.field()
            2-adic Field with capped relative precision 20
        """
        return self._field

    def prime(self):
        r"""
        Return the prime number of this algebra. This
        is the prime of the coefficient field.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.prime()
            2
        """
        return self._field.prime()

    def gen(self, n=0):
        r"""
        Return the ``n``-th generator of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
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
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.gens()
            [(1 + O(2^20))*x, (1 + O(2^20))*y]
        """
        return self._gens

    def ngens(self):
        r"""
        Return the number of generators of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.ngens()
            2

        """
        return len(self._gens)

    def zero(self):
        r"""
        Return the element 0 of this algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.zero()

        """
        return self(0)

    def polyhedron(self):
        r"""
        Return the defining polyhedron of this algebra.

        EXAMPLES:
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.polyhedron()
            A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 2 vertices and 2 rays
        """
        return self._polyhedron

    def vertices(self):
        r"""
        Return the list of vertices of the defining polyhedron of this algebra.

        EXAMPLES:
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.vertices()
            (A vertex at (-2, -1), A vertex at (-1, -2))

        """
        return self._vertices

    def monoid_of_terms(self):
        r"""
        Return the monoid of terms for this algebra.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.monoid_of_terms()
            monoid of terms

        """
        return self._monoid_of_terms

    def vertex(self, i=0, etuple=False):
        r"""
        Return the ``i``-th vertex of the defining polyhedron (indexing starts at zero).
        The flag ``etuple`` indicates whether the result should be returned as an ``ETuple``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.vertex(0)
            A vertex at (-2, -1)
            sage: A.vertex(1, etuple=True)
            (-1, -2)
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.vertex_indices()
            [0, 1]
        """
        return [i for i in range(self._nvertices)]

    def cone_at_vertex(self, i=0):
        r"""
        Return the cone for the ``i``-th vertex of this algebra.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: C = A.cone_at_vertex(0); C
            2-d cone in 2-d lattice N
            sage: C.rays()
            N(1, 1),
            N(0, 1)
            in 2-d lattice N
        """
        if i < 0 or i >= self._nvertices:
            raise IndexError("Vertex index out of range")
        return Cone(self.polyhedron_at_vertex(i))

    def polyhedron_at_vertex(self, i=0):
        r"""
        Return the the cone for the ``i``-th vertex as a polyhedron.
        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: C = A.polyhedron_at_vertex(0); C
            A 2-dimensional polyhedron in QQ^2 defined as the convex hull of 1 vertex and 2 rays
        """
        if i < 0 or i >= self._nvertices:
            raise IndexError("Vertex index out of range")
        ieqs = []
        for v in self._vertices[:i] + self._vertices[i + 1 :]:
            a = tuple(vector(self._vertices[i]) - vector(v))
            ieqs.append((0,) + a)

        return Polyhedron(ieqs=ieqs).intersection(self._quadrant)

    def plot_leading_monomials(self, G):
        r"""
        Experimental. Plot in the same graphics all polyhedrons containing
        the leading monomials of the series in the list ``G``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.plot_leading_monomials([2*x + y, x**2 - 3*y**3])
            Graphics object consisting of 17 graphics primitives
        """
        plots = []
        for g in G:
            plots += [plot(p) for p in g.leading_module_polyhedron()]
        return sum(plots)

    def fan(self):
        r"""
        Return the fan whose maximal cones are the cones at each vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: A.fan()
            Rational polyhedral fan in 2-d lattice N
        """
        cones = []
        for i in range(len(self._vertices)):
            cones.append(Cone(self.cone_at_vertex(i)))

        return Fan(cones=cones)

    def ideal(self, G):
        r"""
        Return the ideal generated by the list of series ``G``.
        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: I = A.ideal([x,y]);
        """
        return PositiveExponentsAlgebraIdeal(G)




class PositiveExponentsTerm(MonoidElement):
    def __init__(self, parent, coeff, exponent=None):
        MonoidElement.__init__(self, parent)
        field = parent._field
        if isinstance(coeff, PositiveExponentsTerm):
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
        for i in self._exponent.nonzero_positions():
            if self._exponent[i] < 0:
                raise ValueError("only nonnegative exponents are allowed")

    def coefficient(self):
        r"""
        Return the coefficient of this term.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_term().coefficient()
            1 + O(2^20)

        """
        return self._coeff

    def exponent(self):
        r"""
        Return the exponent of this term.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_term().exponent()
            (0, 1)

        """
        return self._exponent

    def valuation(self, i=None, vertex_index=False):
        r"""
        Return the valuation of this term if ``i`` is None
        (which is the minimum over the valuations at each vertex),
        otherwise the valuation for the ``i``-th vertex.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_term().valuation()
            1
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: f.leading_term().monomial()
            (1 + O(2^20))*y
        """
        coeff = self.parent()._field(1)
        exponent = self._exponent
        return self.__class__(self.parent(), coeff, exponent)

    def _mul_(self, other):
        r"""
        Return the multiplication of this term by ``other``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: g = x + y
            sage: f.leading_term()*g.leading_term()
            (1 + O(2^20))*y^2
        """
        coeff = self._coeff * other._coeff
        exponent = self._exponent.eadd(other._exponent)
        return self.__class__(self.parent(), coeff, exponent)

    def _to_poly(self):
        r"""
        Return this term as a polynomial.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: type(f.leading_term()._to_poly())
            <class 'sage.rings.polynomial.multi_polynomial_element.MPolynomial_polydict'>
        """
        R = self.parent()._parent_algebra._polynomial_ring
        return self._coeff * R.monomial(self._exponent)

    def _is_divisible_by(self, t):
        r"""
        Test if this term is divisible by ``t``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: a = f.leading_term()
            sage: a._is_divisible_by(a)
            True
        """
        return t.exponent().divides(self.exponent())

    def _divides(self, t):
        r"""
        Return this term divided by ``t``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: a = f.leading_term()
            sage: a._divides(a)
            (1 + O(2^20))
        """
        if not self._is_divisible_by(t):
            raise ValueError("Trying to divides two terms that can't be divided")
        coeff = self._coeff / t._coeff
        exponent = self._exponent.esub(t._exponent)
        return self.__class__(self.parent(), coeff=coeff, exponent=exponent)

    def _cmp_(self, other, i=None):
        r"""
        Compare this term with ``other``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: a = f.leading_term()
            sage: a._cmp_(a)
            0
        """
        ov, oi = other.valuation(vertex_index=True)
        sv, si = self.valuation(vertex_index=True)
        c = ov - sv
        if c == 0:
            c = oi - si
        if c == 0:
            skey = self.parent()._sortkey
            ks = skey(self._exponent)
            ko = skey(other._exponent)
            c = (ks > ko) - (ks < ko)
        return c

    def same(self, t):
        r"""
        Test wether this termù is the same as ``t``, meaning
        it has the same coefficient and exponent.
        This is not the same as equality.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
            sage: a = f.leading_term()
            sage: a.same(a)
            True
        """
        return self._coeff == t._coeff and self._exponent == t._exponent

    def __eq__(self, other):
        r"""
        Test quality of this term with ``other``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 2*x + y + x*y
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
            elif self._exponent[i] > 1:
                s += "*%s^%s" % (parent._names[i], self._exponent[i])
        if s[0] == "*":
            return s[1:]
        else:
            return s


class PositiveExponentsTermMonoid(Monoid_class, UniqueRepresentation):
    Element = PositiveExponentsTerm

    def __init__(self, parent_algebra):
        self.element_class = PositiveExponentsTerm
        names = parent_algebra.variable_names()
        Monoid_class.__init__(self, names=names)
        self._field = parent_algebra._field
        self._names = names
        self._ngens = len(names)
        self._vertices = parent_algebra._vertices
        self._nvertices = parent_algebra._nvertices
        self._order = parent_algebra.term_order()
        self._sortkey = self._order.sortkey
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

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.vertex_indices()
            [0, 1]
        """
        return [i for i in range(self._nvertices)]

    def field(self):
        r"""
        Return the coefficient field of this term monoid.
        This is the same as for the parent algebra.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
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
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.prime()
            2
        """
        return self._field.prime()

    def parent_algebra(self):
        r"""
        Return the parent algebra of this term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: PA = M.parent_algebra()
        """
        return self._parent_algebra

    def term_order(self):
        r"""
        Return the term order used to break ties.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.term_order()
            Degree reverse lexicographic term order
        """
        return self._order

    def variable_names(self):
        r"""
        Return the variables names of this term monoid.
        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
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
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.vertices()
            (A vertex at (-2, -1), A vertex at (-1, -2))

        """
        return self._vertices

    def vertex(self, i=0, etuple=False):
        r"""
        Return the ``i``-th vertex of the defining polyhedron (indexing starts at zero)
        of the parent algebra.
        The flag ``etuple`` indicates whether the result should be returned as an ``ETuple``.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.vertex(0)
            A vertex at (-2, -1)
            sage: M.vertex(1, etuple=True)
            (-1, -2)
        """
        return self._parent_algebra.vertex(i, etuple=etuple)

    def ngens(self):
        r"""
        Return the number of generators of this term monoid.

        EXAMPLES::
            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: M = A.monoid_of_terms()
            sage: M.ngens()
            2

        """
        return self._ngens


class PositiveExponentsAlgebraIdeal:
    def __init__(self, G):
        r"""
        Create a new ideal from list of generators ``G``.
        """
        self._G = G
        self._parent = G[0].parent()

    def groebner_basis(self):
        r"""
        Basic naive Buchberger algorithm.

        EXAMPLES::

            sage: P = Polyhedron(vertices=[(-2,-1),(-1,-2)], rays=[(-1,0),(0,-1)])
            sage: A = PositiveExponentsAlgebra(Qp(2),P,"x,y"); x,y = A.gens()
            sage: f = 3*x*y^2 + 4*x^3
            sage: f = f.add_bigoh(10)
            sage: g = 3*x^2*y^2 + 8*x^2*y
            sage: g = g.add_bigoh(10)
            sage: I = A.ideal([g,f])
            sage: I.groebner_basis()
            [(1 + 2 + O(2^4))*x^2*y^2 + (2^3 + O(2^6))*x^2*y + OO(10),
            (1 + 2 + O(2^6))*x*y^2 + (2^2 + O(2^7))*x^3 + OO(10),
            (2^2 + 2^4 + 2^5 + O(2^6))*x^4 + (2^3 + 2^4 + O(2^6))*x^2*y + OO(10)]
        """
        G = self._G
        G = [g if g._prec != Infinity else g.add_bigoh(g.parent()._prec) for g in G]
        P = [[G[i], G[j]] for i in range(len(G)) for j in range(len(G)) if i < j]
        while len(P) > 0:
            [f, g] = P.pop()
            SPairs = f.all_critical_pairs(g)
            for S in SPairs:
                if not S.is_zero():
                    r = S._quo_rem(G)
                    if not r.is_zero():
                        P += [[g, r] for g in G]
                        G = G + [r]

        return G
