#!/usr/bin/env python3
"""integral"""


def poly_integral(poly, C=0, power=1):
    """ This function is used to generate poly integral function """
    if type(poly) is not list or type(C) not in (int, float):
        return None
    if len(poly) == 0 and power == 1:
        return None

    coefficient, *poly = [*poly, None]

    integrals = [C] if power == 1 else []

    if coefficient is None:
        return []

    if coefficient == 0 and len(integrals) == 1 and len(poly) == 1:
        return [*integrals, *poly_integral(poly, C, power + 1)]

    if coefficient == 0:
        return [*integrals, 0, *poly_integral(poly, C, power + 1)]

    if power != 0:
        result = (coefficient / power)
        result = result if round(result) - result != 0 else round(result)
        return [*integrals, result, *poly_integral(poly, C, power + 1)]

    return [*integrals, coefficient, *poly_integral(poly, C, power + 1)]
