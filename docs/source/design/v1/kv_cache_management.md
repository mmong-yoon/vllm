(design-kv-cache-management)=
# KV Cache Management

This document explains how vLLM allocates and caches KV (key/value) memory. It
is complementary to {doc}`prefix_caching <design/v1/prefix_caching>` and focuses
on the code structure in `vllm.v1`.

## Overview

vLLM divides the KV cache into fixed size *blocks*. These blocks are reused
across requests and may be stored in a global cache. Each request maintains a
mapping from its logical positions to the physical blocks. When a prefix of a
new request matches previous tokens, those blocks can be reused without
recomputation.

### Key modules

- `{mod}`vllm.v1.kv_cache_interface` defines dataclasses describing cache format
  (`KVCacheSpec`, `FullAttentionSpec`, etc.).
- `{mod}`vllm.v1.core.kv_cache_utils` implements the block structures and helper
  utilities such as hashing.
- `{mod}`vllm.v1.core.block_pool` stores all `KVCacheBlock` objects and manages
  eviction.
- `{mod}`vllm.v1.core.single_type_kv_cache_manager` contains managers for
  different attention types (`FullAttentionManager`, `SlidingWindowManager`).
- `{mod}`vllm.v1.core.kv_cache_manager` orchestrates the per-request operations
  and interfaces with the scheduler.

The following sections highlight how these pieces work together.

## Cache data structures

`KVCacheSpec` describes the layout of a layer's cache. For example, a full
attention layer records block size and tensor dimensions:

```python
{lines:vllm/v1/kv_cache_interface.py:70-104}
```

Each physical block is represented by `KVCacheBlock` which holds metadata such as
its reference count and optional hash value:

```python
{lines:vllm/v1/core/kv_cache_utils.py:140-158}
```

`FreeKVCacheBlockQueue` organizes unused blocks in a doubly linked list. This
allows O(1) removal when a cached block is touched:

```python
{lines:vllm/v1/core/kv_cache_utils.py:161-177}
```

The `BlockPool` owns all blocks and handles caching or eviction. Upon
initialization it allocates the blocks and sets up the free list:

```python
{lines:vllm/v1/core/block_pool.py:20-47}
```

Cached blocks are stored in `cached_block_hash_to_block`. When allocating a new
block, `_maybe_evict_cached_block` removes any existing mapping:

```python
{lines:vllm/v1/core/block_pool.py:186-233}
```

## Prefix caching workflow

`KVCacheManager` exposes high level APIs for request scheduling. When a request
is scheduled, the scheduler calls `get_computed_blocks` to find existing cached
blocks:

```python
{lines:vllm/v1/core/kv_cache_manager.py:118-169}
```

If enough blocks remain free, `allocate_slots` reserves new ones and updates the
cache:

```python
{lines:vllm/v1/core/kv_cache_manager.py:180-278}
```

Internally the manager delegates to a `SingleTypeKVCacheManager` for attention
layer specific logic. The full-attention implementation simply retrieves cached
blocks sequentially:

```python
{lines:vllm/v1/core/single_type_kv_cache_manager.py:229-247}
```

For sliding‑window attention, a window of contiguous cached blocks is searched
from the end to maximize reuse:

```python
{lines:vllm/v1/core/single_type_kv_cache_manager.py:283-317}
```

When a request finishes, its blocks are freed in reverse order so that tail
blocks are evicted first:

```python
{lines:vllm/v1/core/kv_cache_manager.py:280-288}
```

## Resetting the cache

`BlockPool.reset_prefix_cache` clears all cached hashes and verifies that only the
null block remains in use. This is useful for RLHF fine‑tuning or benchmarks:

```python
{lines:vllm/v1/core/block_pool.py:272-300}
```

## Summary

vLLM's KV cache manager maintains a pool of blocks and maps hashed prefixes to
physical storage. By sharing blocks across requests and carefully evicting least
recently used data, it maximizes GPU memory utilization while avoiding
fragmentation. The components above implement this design in a modular manner so
that new attention types or caching policies can be integrated easily.
