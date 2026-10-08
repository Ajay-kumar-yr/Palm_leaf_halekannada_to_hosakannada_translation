# Worked recovery examples — the soft bridge at frame level

Cache: `runs\20261007T022632Z_cache_s2_recogniser` · temperature 1.5 · top-5

## What this shows

On these frames the recogniser's top choice was **wrong** and the correct
symbol was still inside its top-5. Argmax keeps one symbol and discards the
rest, so it assigns the correct symbol weight **0** — by construction, not by
measurement. The soft bridge carries it forward with the weight shown.

**This does not show that the bridge produced a correct final answer.** It
shows the information survived the interface. That distinction matters under
questioning.

## Aggregate

- aligned real (non-blank) frames: **383,427**
- frames where top-1 was wrong: **3,607**
- of those, correct symbol still in top-5: **2,907 (80.6%)**
- mean weight the bridge puts on the correct symbol there: **0.200**
- mean weight argmax puts on it: **0.000**

## Examples

### 1. `line_000940` frame 101/230  (frozen test)

Image columns ≈ **808–816px** (frame × 8, the CRNN's width downsample).

- true symbol: **`l`** (rank 2, raw confidence 0.497)
- recogniser chose: **`<blank>`** (raw confidence 0.503)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `<blank>` | 0.502 | 0.503 |
| 2 | `l` ← correct | 0.498 | 0.497 |
| 3 | `r` | 0.000 | 0.000 |
| 4 | `rY` | 0.000 | 0.000 |
| 5 | `a` | 0.000 | 0.000 |

**argmax carries `l` forward with weight 0.000; the bridge carries it with 0.498.**

### 2. `line_001316` frame 191/380  (frozen test)

Image columns ≈ **1528–1536px** (frame × 8, the CRNN's width downsample).

- true symbol: **`x`** (rank 2, raw confidence 0.500)
- recogniser chose: **`X`** (raw confidence 0.500)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `X` | 0.498 | 0.500 |
| 2 | `x` ← correct | 0.498 | 0.500 |
| 3 | `B` | 0.002 | 0.000 |
| 4 | `<blank>` | 0.001 | 0.000 |
| 5 | `W` | 0.001 | 0.000 |

**argmax carries `x` forward with weight 0.000; the bridge carries it with 0.498.**

### 3. `line_000466` frame 220/616  (frozen test)

Image columns ≈ **1760–1768px** (frame × 8, the CRNN's width downsample).

- true symbol: **`t`** (rank 2, raw confidence 0.486)
- recogniser chose: **`<blank>`** (raw confidence 0.514)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `<blank>` | 0.509 | 0.514 |
| 2 | `t` ← correct | 0.490 | 0.486 |
| 3 | `R` | 0.000 | 0.000 |
| 4 | `p` | 0.000 | 0.000 |
| 5 | `d` | 0.000 | 0.000 |

**argmax carries `t` forward with weight 0.000; the bridge carries it with 0.490.**

### 4. `line_000987` frame 207/231  (frozen test)

Image columns ≈ **1656–1664px** (frame × 8, the CRNN's width downsample).

- true symbol: **`x`** (rank 2, raw confidence 0.488)
- recogniser chose: **`X`** (raw confidence 0.511)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `X` | 0.503 | 0.511 |
| 2 | `x` ← correct | 0.488 | 0.488 |
| 3 | `d` | 0.004 | 0.000 |
| 4 | `m` | 0.003 | 0.000 |
| 5 | `B` | 0.001 | 0.000 |

**argmax carries `x` forward with weight 0.000; the bridge carries it with 0.488.**

### 5. `line_000653` frame 189/293  (frozen test)

Image columns ≈ **1512–1520px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.486)
- recogniser chose: **`i`** (raw confidence 0.513)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `i` | 0.505 | 0.513 |
| 2 | `a` ← correct | 0.487 | 0.486 |
| 3 | `e` | 0.005 | 0.001 |
| 4 | `u` | 0.002 | 0.000 |
| 5 | `<blank>` | 0.001 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.487.**

### 6. `line_000953` frame 155/181  (frozen test)

Image columns ≈ **1240–1248px** (frame × 8, the CRNN's width downsample).

- true symbol: **`v`** (rank 2, raw confidence 0.486)
- recogniser chose: **`x`** (raw confidence 0.513)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `x` | 0.503 | 0.513 |
| 2 | `v` ← correct | 0.486 | 0.486 |
| 3 | `d` | 0.004 | 0.000 |
| 4 | `n` | 0.004 | 0.000 |
| 5 | `X` | 0.003 | 0.000 |

**argmax carries `v` forward with weight 0.000; the bridge carries it with 0.486.**

### 7. `line_001167` frame 168/237  (frozen test)

Image columns ≈ **1344–1352px** (frame × 8, the CRNN's width downsample).

- true symbol: **`sp`** (rank 2, raw confidence 0.481)
- recogniser chose: **`<blank>`** (raw confidence 0.518)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `<blank>` | 0.510 | 0.518 |
| 2 | `sp` ← correct | 0.486 | 0.481 |
| 3 | `A` | 0.004 | 0.000 |
| 4 | `i` | 0.000 | 0.000 |
| 5 | `a` | 0.000 | 0.000 |

**argmax carries `sp` forward with weight 0.000; the bridge carries it with 0.486.**

### 8. `line_001028` frame 156/242  (frozen test)

Image columns ≈ **1248–1256px** (frame × 8, the CRNN's width downsample).

- true symbol: **`i`** (rank 2, raw confidence 0.496)
- recogniser chose: **`l`** (raw confidence 0.498)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `l` | 0.484 | 0.498 |
| 2 | `i` ← correct | 0.482 | 0.496 |
| 3 | `r` | 0.020 | 0.004 |
| 4 | `<blank>` | 0.011 | 0.002 |
| 5 | `x` | 0.003 | 0.000 |

**argmax carries `i` forward with weight 0.000; the bridge carries it with 0.482.**

### 9. `line_001255` frame 149/404  (frozen test)

Image columns ≈ **1192–1200px** (frame × 8, the CRNN's width downsample).

- true symbol: **`n`** (rank 2, raw confidence 0.477)
- recogniser chose: **`s`** (raw confidence 0.523)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `s` | 0.513 | 0.523 |
| 2 | `n` ← correct | 0.482 | 0.477 |
| 3 | `r` | 0.002 | 0.000 |
| 4 | `h` | 0.001 | 0.000 |
| 5 | `g` | 0.001 | 0.000 |

**argmax carries `n` forward with weight 0.000; the bridge carries it with 0.482.**

### 10. `line_000695` frame 271/370  (frozen test)

Image columns ≈ **2168–2176px** (frame × 8, the CRNN's width downsample).

- true symbol: **`x`** (rank 2, raw confidence 0.474)
- recogniser chose: **`X`** (raw confidence 0.526)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `X` | 0.516 | 0.526 |
| 2 | `x` ← correct | 0.482 | 0.474 |
| 3 | `<blank>` | 0.001 | 0.000 |
| 4 | `B` | 0.000 | 0.000 |
| 5 | `W` | 0.000 | 0.000 |

**argmax carries `x` forward with weight 0.000; the bridge carries it with 0.482.**

### 11. `line_000211` frame 585/878  (frozen test)

Image columns ≈ **4680–4688px** (frame × 8, the CRNN's width downsample).

- true symbol: **`v`** (rank 2, raw confidence 0.475)
- recogniser chose: **`p`** (raw confidence 0.525)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `p` | 0.515 | 0.525 |
| 2 | `v` ← correct | 0.482 | 0.475 |
| 3 | `m` | 0.003 | 0.000 |
| 4 | `x` | 0.000 | 0.000 |
| 5 | `<blank>` | 0.000 | 0.000 |

**argmax carries `v` forward with weight 0.000; the bridge carries it with 0.482.**

### 12. `line_001117` frame 16/260  (frozen test)

Image columns ≈ **128–136px** (frame × 8, the CRNN's width downsample).

- true symbol: **`m`** (rank 2, raw confidence 0.484)
- recogniser chose: **`e`** (raw confidence 0.514)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.501 | 0.514 |
| 2 | `m` ← correct | 0.481 | 0.484 |
| 3 | `<blank>` | 0.010 | 0.002 |
| 4 | `a` | 0.005 | 0.000 |
| 5 | `w` | 0.002 | 0.000 |

**argmax carries `m` forward with weight 0.000; the bridge carries it with 0.481.**

