# Linear axis operating note

For a homed `arm-cell`, `move_absolute` assigns `position_mm`. The deployment range is 5 through
80 mm even though the fake driver can represent 0 through 100 mm. A low-risk write requires an
approval bound to the exact proposal. `jog_relative` is non-idempotent and must never be retried
after an unknown outcome until reconciliation has established what happened.

