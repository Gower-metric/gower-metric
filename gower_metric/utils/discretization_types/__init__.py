from gower_metric.utils.discretization_types import knn, silverman

__all__ = ["knn", "silverman"]

__doc__ = """
This module provides kernel density estimation (KDE) bandwidth selection methods,
to be applied for discretization.
Currently, it includes Silverman's and knn rule of thumb for bandwidth selection.
"""
