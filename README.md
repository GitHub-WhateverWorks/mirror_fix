# Mirror-Aware Depth Correction for Monocular Depth Estimation

> Using Semantic Segmentation and Boundary-Guided Recovery

**Authors:** Bo-Han Ho, Yun-An Chaung, Han-Hsiang Li
**Affiliation:** Department of Computer Science, National Yang Ming
Chiao Tung University, Hsinchu, Taiwan

## Overview

Modern monocular depth estimation models perform well on ordinary scenes
but can fail on mirrors. Reflected objects may be interpreted as real
scene geometry, causing the predicted depth inside a mirror to
correspond to the reflected scene rather than the physical mirror
surface.

This project develops a modular mirror-depth correction pipeline
combining:

1.  **DINOv3-based mirror segmentation**
2.  **Depth Anything 3 (DA3)** for monocular depth estimation
3.  Three mirror depth recovery methods:
    -   **Plane Fitting**
    -   **Gaussian Recovery**
    -   **Propagation Recovery**

The system corrects only the predicted mirror region while preserving
the original DA3 depth elsewhere.

## Pipeline

The overall pipeline is:

``` text
RGB Image
   ├──> DINOv3 Semantic Segmentation ──> Mirror Mask
   │
   └──> Depth Anything 3 ──────────────> Raw Depth
                                             │
Mirror Mask ─────────────────────────────────┤
                                             v
                                  Depth Recovery Method
                                             │
                                             v
                                      Corrected Depth
```

For an RGB image (I), the system predicts a mirror mask (M) and raw
depth (D\_{`\text{raw}`{=tex}}). A recovery method estimates corrected
depth (D\_{`\text{corr}`{=tex}}) inside the usable mirror mask:

\[ D\_{`\text{final}`{=tex}}(p)=
```{=tex}
\begin{cases}
D_{\text{corr}}(p), & p\in M_{\text{used}},\\
D_{\text{raw}}(p), & \text{otherwise}.
\end{cases}
```
\]

## Mirror Segmentation

Mirror regions are detected using a **DINOv3 ViT-based semantic
segmentor** with the ADE20K mirror class (class 27).

Two mask variants were evaluated:

-   **Base** --- direct mirror prediction from the segmentor.
-   **Base Threshold** --- confidence-thresholded prediction intended to
    remove low-confidence false positives.

Operational post-processing includes small-component removal,
conservative hole filling, and optional erosion.

### Segmentation as a Bottleneck

A major finding is that segmentation quality cannot be described only by
metrics such as precision or IoU.

For downstream depth recovery, the mask must also have:

-   correct **topology**;
-   sufficient **spatial continuity**;
-   high **semantic completeness**.

Thresholding may increase mask purity while fragmenting a continuous
mirror region. This can significantly reduce the effectiveness of
boundary-conditioned recovery.

A more fundamental problem occurs when reflected objects are classified
according to their apparent semantic category instead of as part of the
physical mirror. Those pixels are absent from the mirror mask and
therefore cannot be modified by any downstream mask-guided recovery
method.

## Depth Estimation

[Depth Anything 3](https://arxiv.org/abs/2511.10647) is used to generate
the raw monocular depth map.

DA3 itself is not retrained. Instead, the project investigates whether
its mirror-induced errors can be corrected externally while retaining
its predictions in non-mirror regions.

## Recovery Methods

### 1. Plane Fitting

Plane Fitting assumes that many indoor mirrors are mounted on
approximately planar structures such as walls, cabinets, or doors.

Depth samples around the mirror boundary are used to estimate:

\[ z=ax+by+c. \]

The fitted plane is then evaluated over the mirror region.

**Strengths**

-   Strongest peak recovery performance.
-   Can almost completely remove reflected-object geometry when the
    planar assumption is correct.
-   Particularly effective for mirrors embedded in planar surfaces.

**Limitations**

-   Sensitive to contaminated support pixels.
-   Mirror frames, nearby objects, or segmentation errors can bias the
    fitted plane.

### 2. Gaussian Recovery

Gaussian Recovery uses spatially varying depth information derived from
the mirror boundary to initialize the mirror interior.

Gaussian smoothing then updates each pixel using distance-weighted
neighboring depth values:

\[ `\hat `{=tex}D(p)=
`\frac{\sum_{q\in\mathcal N(p)}G_\sigma(p-q)D(q)}`{=tex}
{`\sum`{=tex}*{q`\in`{=tex}`\mathcal `{=tex}N(p)}G*`\sigma`{=tex}(p-q)}.
\]

The estimate can be updated iteratively:

\[
D\_{t+1}(p)=`\alpha`{=tex}`\hat `{=tex}D_t(p)+(1-`\alpha`{=tex})D_t(p).
\]

An early implementation initialized the entire mirror region with one
scalar boundary depth. This produced a nearly constant-depth blob.
Replacing this with **boundary-aware spatial initialization** introduced
meaningful depth variation and substantially improved the result.

**Strengths**

-   More tolerant to moderate segmentation noise than Plane Fitting.
-   Smooth, spatially varying reconstruction.
-   Higher recovery coverage across the evaluated scenes.

### 3. Propagation Recovery

Propagation treats the mirror interior as initially unknown.

Depth is progressively propagated from valid pixels outside the mirror
toward the center. Newly reached pixels receive the average of
already-known neighboring values:

\[ D\_{`\text{new}`{=tex}}(p)= `\frac{1}{|\mathcal K(p)|}`{=tex}
`\sum`{=tex}\_{q`\in`{=tex}`\mathcal `{=tex}K(p)}D(q). \]

The key distinction is:

> **Gaussian Recovery:** boundary-derived initialization +
> Gaussian-weighted smoothing\
> **Propagation Recovery:** empty-region initialization +
> neighbor-average inward growth

Propagation can create visually smooth depth surfaces. However,
surrounding geometric structures may also propagate inward, creating
coherent gradients or triangular structures that appear smooth but are
geometrically inaccurate.

## Evaluation

### Mirror--Covered Image Pairs

The evaluation dataset contains approximately **70 paired scenes**.

Each scene contains:

-   a photograph with the **mirror visible**;
-   a photograph from approximately the same viewpoint with the **mirror
    covered**.

Because physical sensor ground truth was not available, DA3 depth
estimated from the covered image is used as a **pseudo-reference**, not
true ground truth.

### ECC Image Registration

The mirror and covered photographs are separate captures, so small
camera movements can produce translation, rotation, and other geometric
differences.

**Enhanced Correlation Coefficient (ECC)** image registration is used to
align the covered image with the mirror image before depth comparison.

Conceptually, ECC estimates a warp (W) such that:

\[ I\_{`\text{covered}`{=tex}}(W(p)) `\approx`{=tex}
I\_{`\text{mirror}`{=tex}}(p). \]

The same transformation is applied to the covered depth map.

The evaluation implementation uses approximately:

-   **100 ECC iterations**
-   convergence tolerance of **(10\^{-5})**
-   light pre-alignment blur

ECC reduces global misalignment, although residual perspective
differences and parallax can remain.

### Depth Calibration

Independent monocular depth predictions may differ in scale and shift.

Before comparison, an affine depth transformation is fitted:

\[ D\_{`\text{ref}`{=tex}}\^{\*}=sD\_{`\text{ref}`{=tex}}+t. \]

This places the mirror and covered predictions into a comparable depth
coordinate system.

## Metrics

Metrics are evaluated over three regions:

-   **Full image**
-   **Used mirror mask**
-   **Changed region**

### Mean Absolute Depth Deviation

\[ `\mathrm{MADD}`{=tex} = `\frac{1}{N}`{=tex} `\sum`{=tex}\_{i=1}\^{N}
\|D_i-D_i\^{\*}\|. \]

### Root Mean Squared Depth Deviation

\[ `\mathrm{RMSDD}`{=tex} = `\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(D_i-D_i^{*})^2
}`{=tex}. \]

Improvement is defined as:

\[ `\Delta `{=tex}E=E\_{`\text{raw}`{=tex}}-E\_{`\text{fixed}`{=tex}}.
\]

Therefore, a **positive value indicates improvement**.

Relative MADD improvement is:

\[ R\_{`\mathrm{MADD}`{=tex}} = `\frac{
\mathrm{MADD}_{\text{raw}}-
\mathrm{MADD}_{\text{fixed}}
}{
\mathrm{MADD}_{\text{raw}}
}`{=tex} `\times100`{=tex}%. \]

A recovery is considered successful when the correction head is applied
and the used-mask MADD improves.

## Results

### Representative Best Base-Mask Results

  Method            Pair   Raw MADD   Fixed MADD   Improvement
  --------------- ------ ---------- ------------ -------------
  Plane Fitting     0059      0.432        0.095     **78.0%**
  Plane Fitting     0011      0.322        0.079     **75.3%**
  Plane Fitting     0032      0.651        0.245     **62.3%**
  Gaussian          0032      0.651        0.223     **65.8%**
  Gaussian          0011      0.322        0.146     **54.7%**
  Gaussian          0059      0.432        0.309     **28.5%**
  Propagation       0011      0.322        0.233     **27.4%**
  Propagation       0060      0.152        0.121     **20.6%**
  Propagation       0032      0.651        0.527     **19.1%**

### Plane Fitting

Plane Fitting has the highest observed recovery ceiling.

The best case reduces used-mask MADD by **78.0%**, and multiple strong
cases exceed **70%** improvement.

When the mirror is correctly localized and lies on an approximately
planar surface, the geometric prior is highly effective.

### Gaussian Recovery

Gaussian Recovery has a lower peak than Plane Fitting but succeeds
across more evaluated cases.

Its weaker geometric assumptions and tolerance to moderate segmentation
noise make it a strong general baseline.

### Propagation Recovery

Propagation provides measurable improvements but generally leaves larger
residual geometric error.

Its main failure mode is the propagation of external geometry into the
mirror region. Consequently, a smooth-looking output does not
necessarily correspond to accurate recovered geometry.

## Base vs. Threshold Masks

Higher segmentation confidence does **not** necessarily produce better
downstream recovery.

Examples:

  Method / Pair               Base   Base Threshold
  -------------------- ----------- ----------------
  Gaussian / 0032        **65.8%**            19.2%
  Gaussian / 0011        **54.7%**            13.1%
  Propagation / 0011     **27.4%**             8.5%

Thresholding can remove uncertain pixels but also fragment the mirror
mask. Gaussian and Propagation rely heavily on coherent boundary
information, so this loss of spatial continuity can substantially reduce
recovery quality.

This leads to an important conclusion:

> **Higher segmentation accuracy does not automatically imply better
> downstream depth recovery.**

Mask topology and spatial continuity are also important.

## Main Findings

The three methods exhibit complementary behavior:

  -----------------------------------------------------------------------
  Method                  Main Advantage          Main Limitation
  ----------------------- ----------------------- -----------------------
  **Plane Fitting**       Highest peak recovery;  Sensitive to incorrect
                          best observed           support geometry
                          improvement **78.0%**   

  **Gaussian Recovery**   Higher recovery         Lower peak than Plane
                          coverage and tolerance  Fitting
                          to segmentation noise   

  **Propagation           Smooth interpolation    Can propagate incorrect
  Recovery**              and simple inward       external geometry
                          filling                 inward
  -----------------------------------------------------------------------

The largest system-level limitation is not necessarily the recovery
method itself. It is often **segmentation completeness**.

If reflection-induced depth artifacts are not included in the mirror
mask, no mask-guided recovery method can correct them.

## Limitations

### Residual Image Misalignment

ECC reduces global alignment error but cannot fully remove all
perspective changes or parallax between separately captured images.

Even a small residual pixel displacement can increase MADD and RMSDD.

### Pseudo-Ground-Truth Error

Covered-scene depth is generated by DA3 rather than a physical depth
sensor.

The reference can therefore contain its own errors, particularly around:

-   object boundaries;
-   occlusions;
-   complex geometry;
-   low-texture regions.

The reported numbers measure agreement with an **aligned and calibrated
DA3 pseudo-reference**, not absolute metric depth accuracy.

### Mirror Segmentation

General-purpose semantic segmentation may classify reflected objects as
their apparent object categories rather than as part of the mirror.

This creates incomplete mirror masks and places a hard upper bound on
downstream recovery.

## Future Work

Future work should prioritize:

-   mirror-specific or **reflection-aware segmentation**;
-   glass-only surface localization;
-   fixed-camera capture or stronger image registration;
-   physical sensor depth for ground-truth evaluation;
-   adaptive recovery based on estimated surface planarity;
-   learned recovery using RGB appearance, raw depth, mask confidence,
    and boundary geometry.

An adaptive system could apply Plane Fitting when planarity confidence
is high and use a boundary-conditioned recovery method for more
uncertain or non-planar regions.

## Conclusion

We propose and compare three mirror depth recovery methods.

**Plane Fitting** achieves the strongest peak recovery, reducing
used-mask depth error by more than 70% in multiple cases and reaching
**78.0%** in the best observed case.

**Gaussian Recovery** has a lower peak but succeeds across more cases
and is more tolerant to segmentation noise.

**Propagation Recovery** can produce smooth depth surfaces but may
propagate surrounding geometric structures into the mirror, resulting in
larger residual geometric error.

The experiments also demonstrate that mirror segmentation quality cannot
be characterized by precision or IoU alone. **Mask topology, spatial
continuity, and semantic completeness directly affect downstream
recovery.**

Finally, quantitative results are affected by residual pair misalignment
and the use of DA3-generated covered depth as pseudo-ground truth. More
accurate registration, sensor-quality reference depth, and especially a
mirror-specialized segmentor provide the clearest directions for
improving the system.

## References

1.  O. Siméoni *et al.*, **"DINOv3,"** arXiv:2508.10104, 2025.\
    https://arxiv.org/abs/2508.10104

2.  H. Lin, S. Chen, J. H. Liew, D. Y. Chen, Z. Li, G. Shi, J. Feng,
    and B. Kang, **"Depth Anything 3: Recovering the visual space from
    any views,"** arXiv:2511.10647, 2025.\
    https://arxiv.org/abs/2511.10647

3.  G. D. Evangelidis and E. Z. Psarakis, **"Parametric Image Alignment
    Using Enhanced Correlation Coefficient Maximization,"** *IEEE
    Transactions on Pattern Analysis and Machine Intelligence*, vol. 30,
    no. 10, pp. 1858--1865, 2008.
