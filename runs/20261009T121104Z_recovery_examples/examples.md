# Worked recovery examples — the soft bridge at frame level

Cache: `runs/20261009T120133Z_cache_s2_recogniser` · temperature 1.5 · top-5

## What this shows

On these frames the recogniser's top choice was **wrong** and the correct
symbol was still inside its top-5. Argmax keeps one symbol and discards the
rest, so it assigns the correct symbol weight **0** — by construction, not by
measurement. The soft bridge carries it forward with the weight shown.

**This does not show that the bridge produced a correct final answer.** It
shows the information survived the interface. That distinction matters under
questioning.

## Aggregate

- aligned real (non-blank) frames: **382,520**
- frames where top-1 was wrong: **3,919**
- of those, correct symbol still in top-5: **2,980 (76.0%)**
- mean weight the bridge puts on the correct symbol there: **0.203**
- mean weight argmax puts on it: **0.000**

## Examples

### 1. `line_000594` frame 205/522  (frozen test)

Image columns ≈ **1640–1648px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.493)
- recogniser chose: **`e`** (raw confidence 0.506)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.501 | 0.506 |
| 2 | `a` ← correct | 0.492 | 0.493 |
| 3 | `i` | 0.005 | 0.000 |
| 4 | `u` | 0.002 | 0.000 |
| 5 | `<blank>` | 0.001 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.492.**

### 2. `line_000733` frame 4/531  (frozen test)

Image columns ≈ **32–40px** (frame × 8, the CRNN's width downsample).

- true symbol: **`I`** (rank 2, raw confidence 0.492)
- recogniser chose: **`E`** (raw confidence 0.507)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `E` | 0.500 | 0.507 |
| 2 | `I` ← correct | 0.491 | 0.492 |
| 3 | `O` | 0.005 | 0.000 |
| 4 | `a` | 0.003 | 0.000 |
| 5 | `V` | 0.001 | 0.000 |

**argmax carries `I` forward with weight 0.000; the bridge carries it with 0.491.**

### 3. `line_000745` frame 92/626  (frozen test)

Image columns ≈ **736–744px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.487)
- recogniser chose: **`e`** (raw confidence 0.513)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.508 | 0.513 |
| 2 | `a` ← correct | 0.490 | 0.487 |
| 3 | `E` | 0.001 | 0.000 |
| 4 | `<blank>` | 0.000 | 0.000 |
| 5 | `i` | 0.000 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.490.**

### 4. `line_000736` frame 2/351  (frozen test)

Image columns ≈ **16–24px** (frame × 8, the CRNN's width downsample).

- true symbol: **`b`** (rank 2, raw confidence 0.485)
- recogniser chose: **`B`** (raw confidence 0.515)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `B` | 0.509 | 0.515 |
| 2 | `b` ← correct | 0.489 | 0.485 |
| 3 | `c` | 0.001 | 0.000 |
| 4 | `C` | 0.001 | 0.000 |
| 5 | `<blank>` | 0.001 | 0.000 |

**argmax carries `b` forward with weight 0.000; the bridge carries it with 0.489.**

### 5. `line_001275` frame 182/493  (frozen test)

Image columns ≈ **1456–1464px** (frame × 8, the CRNN's width downsample).

- true symbol: **`r`** (rank 2, raw confidence 0.493)
- recogniser chose: **`<blank>`** (raw confidence 0.505)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `<blank>` | 0.496 | 0.505 |
| 2 | `r` ← correct | 0.489 | 0.493 |
| 3 | `g` | 0.012 | 0.002 |
| 4 | `T` | 0.003 | 0.000 |
| 5 | `x` | 0.001 | 0.000 |

**argmax carries `r` forward with weight 0.000; the bridge carries it with 0.489.**

### 6. `line_000247` frame 183/419  (frozen test)

Image columns ≈ **1464–1472px** (frame × 8, the CRNN's width downsample).

- true symbol: **`i`** (rank 2, raw confidence 0.486)
- recogniser chose: **`a`** (raw confidence 0.513)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `a` | 0.506 | 0.513 |
| 2 | `i` ← correct | 0.488 | 0.486 |
| 3 | `e` | 0.003 | 0.000 |
| 4 | `U` | 0.002 | 0.000 |
| 5 | `I` | 0.001 | 0.000 |

**argmax carries `i` forward with weight 0.000; the bridge carries it with 0.488.**

### 7. `line_000183` frame 583/777  (frozen test)

Image columns ≈ **4664–4672px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.489)
- recogniser chose: **`e`** (raw confidence 0.509)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.501 | 0.509 |
| 2 | `a` ← correct | 0.488 | 0.489 |
| 3 | `i` | 0.008 | 0.001 |
| 4 | `E` | 0.002 | 0.000 |
| 5 | `<blank>` | 0.001 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.488.**

### 8. `line_001267` frame 97/393  (frozen test)

Image columns ≈ **776–784px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.482)
- recogniser chose: **`e`** (raw confidence 0.518)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.511 | 0.518 |
| 2 | `a` ← correct | 0.488 | 0.482 |
| 3 | `i` | 0.001 | 0.000 |
| 4 | `<blank>` | 0.000 | 0.000 |
| 5 | `E` | 0.000 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.488.**

### 9. `line_000744` frame 371/541  (frozen test)

Image columns ≈ **2968–2976px** (frame × 8, the CRNN's width downsample).

- true symbol: **`w`** (rank 2, raw confidence 0.487)
- recogniser chose: **`a`** (raw confidence 0.511)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `a` | 0.503 | 0.511 |
| 2 | `w` ← correct | 0.487 | 0.487 |
| 3 | `k` | 0.003 | 0.000 |
| 4 | `<blank>` | 0.003 | 0.000 |
| 5 | `r` | 0.002 | 0.000 |

**argmax carries `w` forward with weight 0.000; the bridge carries it with 0.487.**

### 10. `line_001104` frame 179/240  (frozen test)

Image columns ≈ **1432–1440px** (frame × 8, the CRNN's width downsample).

- true symbol: **`a`** (rank 2, raw confidence 0.483)
- recogniser chose: **`e`** (raw confidence 0.517)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `e` | 0.510 | 0.517 |
| 2 | `a` ← correct | 0.487 | 0.483 |
| 3 | `<blank>` | 0.001 | 0.000 |
| 4 | `sp` | 0.001 | 0.000 |
| 5 | `u` | 0.001 | 0.000 |

**argmax carries `a` forward with weight 0.000; the bridge carries it with 0.487.**

### 11. `line_000499` frame 142/491  (frozen test)

Image columns ≈ **1136–1144px** (frame × 8, the CRNN's width downsample).

- true symbol: **`n`** (rank 2, raw confidence 0.490)
- recogniser chose: **`s`** (raw confidence 0.508)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `s` | 0.500 | 0.508 |
| 2 | `n` ← correct | 0.487 | 0.490 |
| 3 | `<blank>` | 0.008 | 0.001 |
| 4 | `g` | 0.003 | 0.000 |
| 5 | `v` | 0.003 | 0.000 |

**argmax carries `n` forward with weight 0.000; the bridge carries it with 0.487.**

### 12. `line_001178` frame 210/288  (frozen test)

Image columns ≈ **1680–1688px** (frame × 8, the CRNN's width downsample).

- true symbol: **`k`** (rank 2, raw confidence 0.481)
- recogniser chose: **`<blank>`** (raw confidence 0.519)

| rank | symbol | bridge weight | raw confidence |
|---|---|---|---|
| 1 | `<blank>` | 0.513 | 0.519 |
| 2 | `k` ← correct | 0.487 | 0.481 |
| 3 | `w` | 0.000 | 0.000 |
| 4 | `r` | 0.000 | 0.000 |
| 5 | `R` | 0.000 | 0.000 |

**argmax carries `k` forward with weight 0.000; the bridge carries it with 0.487.**

