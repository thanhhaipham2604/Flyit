"""Rate limiting so the service does not get swamped (slowapi/limits).

Default: settings.rate_limit_per_minute per client IP. Return 429 with a
Retry-After header. Must hold up in the concurrent-user demo.
"""

# TODO: configure slowapi Limiter and wire into api.main / routes.
