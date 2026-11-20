"""Smoke scripts that run INSIDE a candidate's isolated environment, never imported by `llb`.

Each is executed as `<env-python> -I <script> ...`: standard library plus the candidate's own
package only, writing its reading or its preprocessed image into the output directory it is given.
"""
