# Clamp operating note

The `clamp-cell` operation `set_grip` accepts the boolean argument `closed`. Closing is allowed only
when `object_present` is true. The clamp depends on `arm-cell`, so both devices must be available for
the operation to be serialized safely. A low-risk write needs proposal-bound approval.

