"""Predictjra maintenance, update, rebuild, and prediction modules.

Production entry points are still executed as scripts by GitHub Actions.  Making this
an explicit package also lets root-level compatibility shims import the canonical
implementations without duplicating source files.
"""
