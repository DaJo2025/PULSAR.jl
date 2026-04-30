# ============================================================================
# Computation/PropagatorCache.jl
#
# Lesson 6: opt-in propagator cache for shared-Hamiltonian ensembles.
#
# Pattern source: Krotov.jl Python (`krotov/parallelization.py:35-47`) caches
# forward propagators across multi-objective members that share a Hamiltonian.
#
# The cache is keyed by `(objectid(system), hash(controls))` — `objectid`
# captures the identity of the immutable system struct (drift Hamiltonian,
# control Hamiltonians) without hashing matrix contents on every lookup, and
# `hash(controls)` distinguishes different waveforms or perturbations.
#
# Usage (gated by an opt-in flag in callers):
#
#     cache  = PropagatorCache()
#     U      = cached_propagator(cache, system, controls) do
#                  _compute_propagator(system, controls)
#              end
#     # ... later in the same iteration ...
#     U_again = cached_propagator(cache, system, controls) do
#                  _compute_propagator(system, controls)        # not called
#              end
#     cache_clear!(cache)   # at end of iteration to bound memory
#
# Cache safety: stale entries are avoided by clearing after each optimizer
# iteration; correctness depends on the caller passing a controls matrix that
# is byte-equal to the one used at lookup time. Numerical instability from
# floating-point round-trips is not an issue because `hash` is exact.
# ============================================================================

"""
    PropagatorCache()

Per-iteration cache mapping `(system_id, controls_hash) → U` to avoid recomputing
identical propagator stacks across ensemble members that share a Hamiltonian.

Always opt-in: callers must explicitly create, populate, and clear the cache.
"""
mutable struct PropagatorCache
    entries :: Dict{Tuple{UInt,UInt}, Any}
    hits    :: Int
    misses  :: Int
    PropagatorCache() = new(Dict{Tuple{UInt,UInt}, Any}(), 0, 0)
end

"""
    cached_propagator(compute_fn, cache, system, controls)

Return the cached propagator for `(system, controls)`, calling `compute_fn()` to
build it on a cache miss.  Used as

    U = cached_propagator(cache, system, controls) do
        _compute_propagator(system, controls)
    end

`compute_fn` is a zero-arg closure so the heavy work is skipped on cache hits.
"""
function cached_propagator(compute_fn, cache::PropagatorCache,
                            system, controls::AbstractMatrix)
    key = (objectid(system), hash(controls))
    entry = get(cache.entries, key, nothing)
    if entry !== nothing
        cache.hits += 1
        return entry
    end
    cache.misses += 1
    U = compute_fn()
    cache.entries[key] = U
    return U
end

"""
    cache_clear!(cache::PropagatorCache)

Drop all cached propagators.  Call at the end of each optimizer iteration to
bound memory.  Resets hit/miss counters.
"""
function cache_clear!(cache::PropagatorCache)
    empty!(cache.entries)
    cache.hits   = 0
    cache.misses = 0
    return cache
end

"""
    cache_stats(cache::PropagatorCache) -> (; hits, misses, hit_rate)

Diagnostic accessor for cache hit-rate measurements.
"""
function cache_stats(cache::PropagatorCache)
    total = cache.hits + cache.misses
    rate  = total > 0 ? cache.hits / total : 0.0
    return (hits = cache.hits, misses = cache.misses, hit_rate = rate)
end
