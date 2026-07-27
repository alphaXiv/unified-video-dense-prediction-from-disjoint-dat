import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    RESULTS = [
        {"seed": 0, "latent": 0.9735, "ft": 1.0563, "direct": 0.9408, "frozen": 0.3341, "pixel": 0.9424, "latent_cross": 0.4762, "pixel_cross": 0.3567},
        {"seed": 1, "latent": 0.9744, "ft": 1.0605, "direct": 1.0169, "frozen": 0.3296, "pixel": 0.9293, "latent_cross": 0.4992, "pixel_cross": 0.4143},
        {"seed": 2, "latent": 0.9495, "ft": 1.0473, "direct": 0.8628, "frozen": 0.3261, "pixel": 0.9541, "latent_cross": 0.3081, "pixel_cross": 0.4847},
        {"seed": 3, "latent": 0.9652, "ft": 1.0199, "direct": 1.0122, "frozen": 0.3311, "pixel": 0.9774, "latent_cross": 0.3638, "pixel_cross": 0.3999},
        {"seed": 4, "latent": 0.9877, "ft": 1.1583, "direct": 1.1802, "frozen": 0.3456, "pixel": 0.9241, "latent_cross": 0.2700, "pixel_cross": 0.4489},
        {"seed": 5, "latent": 0.9999, "ft": 1.1482, "direct": 1.0738, "frozen": 0.3490, "pixel": 0.9768, "latent_cross": 0.4180, "pixel_cross": 0.4341},
        {"seed": 6, "latent": 0.9883, "ft": 1.2006, "direct": 1.1192, "frozen": 0.3576, "pixel": 1.0143, "latent_cross": 0.5261, "pixel_cross": 0.4880},
        {"seed": 7, "latent": 0.9485, "ft": 1.0241, "direct": 1.0038, "frozen": 0.3408, "pixel": 0.9557, "latent_cross": 0.2920, "pixel_cross": 0.4060},
    ]
    MEMORY = {
        2: {"latent_peak": 0.5382, "pixel_peak": 2.2840, "incremental_ratio": 5.528},
        8: {"latent_peak": 0.5382, "pixel_peak": 4.0421, "incremental_ratio": 10.088},
    }
    return MEMORY, RESULTS


@app.cell
def _(mo):
    mo.md(r"""
    # Can compact latent teaching unify disjoint dense tasks?

    Depth labels and semantic labels are usually collected on different images. **UniD (arXiv 2607.21592)** proposes training separate specialists, then teaching one shared diffusion backbone to reproduce each specialist's compact internal representation through a small task projector.

    **Verdict: partially reproduced.** In this two-task reconstruction, latent distillation crushed a frozen-backbone control and reproduced the memory advantage, but did not reliably beat direct joint training. Projector-only fine-tuning was the most consistent accuracy result.
    """)
    return


@app.cell
def _(RESULTS, mo):
    def mean(key):
        return sum(row[key] for row in RESULTS) / len(RESULTS)

    bars = [
        ("Specialists", 1.0, "#172033"),
        ("Latent", mean("latent"), "#356ae6"),
        ("Latent + FT", mean("ft"), "#129c8b"),
        ("Direct", mean("direct"), "#ef8b2c"),
        ("Frozen", mean("frozen"), "#d85555"),
        ("Pixel", mean("pixel"), "#7b61c9"),
    ]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 430" style="width:100%;background:#fbfaf7">']
    svg.append('<text x="30" y="34" font-size="22" font-weight="700" fill="#172033">Mean specialist accuracy retained (8 seeds)</text>')
    svg.append('<line x1="70" y1="110" x2="860" y2="110" stroke="#172033" stroke-dasharray="6 5"/>')
    for i, (label, value, color) in enumerate(bars):
        x = 78 + i * 128
        height = value / 1.25 * 290
        y = 342 - height
        svg.append(f'<rect x="{x}" y="{y}" width="92" height="{height}" rx="6" fill="{color}"/>')
        svg.append(f'<text x="{x+46}" y="{y-8}" text-anchor="middle" font-size="14" font-weight="700">{100*value:.1f}%</text>')
        svg.append(f'<text x="{x+46}" y="370" text-anchor="middle" font-size="12">{label}</text>')
    svg.append('<text x="70" y="405" font-size="13" fill="#667085">Dashed line = matched specialist parity. Retention averages normalized NYU depth and ADE20K segmentation.</text></svg>')
    mo.Html("".join(svg))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reconstructed training path

    1. Train one diffusion U-Net specialist for NYU depth and another for ADE20K segmentation.
    2. Initialize a fresh shared U-Net from the same public diffusion checkpoint.
    3. Alternate disjoint task batches. A 1×1 task projector maps the shared three-channel 32×32 representation to the selected specialist's representation.
    4. Compare against a frozen pretrained backbone, direct supervised joint training, and teacher matching after the full pixel decoder.
    5. Freeze the shared U-Net and fine-tune only the segmentation projector and pixel head.

    The public task data were real, but the setup was deliberately small: 64×64 inputs, 512 NYU training examples, 800 ADE20K training examples, and a 35M-parameter CIFAR-10 DDPM U-Net. It did **not** test video, temporal memory, or the paper's Stable Diffusion model.
    """)
    return


@app.cell
def _(mo):
    seed = mo.ui.slider(0, 7, value=0, step=1, label="Inspect a long-schedule seed")
    seed
    return (seed,)


@app.cell
def _(RESULTS, mo, seed):
    row = RESULTS[seed.value]
    winner = max(("latent", "ft", "direct", "frozen", "pixel"), key=lambda key: row[key])
    mo.md(
        f"""
        **Seed {seed.value}:** latent {row['latent']:.3f}, latent + fine-tuning {row['ft']:.3f},
        direct joint {row['direct']:.3f}, frozen {row['frozen']:.3f}, pixel {row['pixel']:.3f}.
        The best retention for this seed is **{winner}**. Latent/pixel edge consistency is
        {row['latent_cross']:.3f}/{row['pixel_cross']:.3f}.
        """
    )
    return


@app.cell
def _(MEMORY, mo):
    mo.md(
        f"""
        ## The memory claim is the clearest result

        | Active task decoders | Latent peak | Per-pixel peak | Pixel / latent incremental memory |
        |---:|---:|---:|---:|
        | 2 | {MEMORY[2]['latent_peak']:.2f} GiB | {MEMORY[2]['pixel_peak']:.2f} GiB | {MEMORY[2]['incremental_ratio']:.2f}× |
        | 8 | {MEMORY[8]['latent_peak']:.2f} GiB | {MEMORY[8]['pixel_peak']:.2f} GiB | {MEMORY[8]['incremental_ratio']:.2f}× |

        These are measured CUDA allocations at 256×256 and batch 8. The paper estimated
        650 GiB for pixel distillation versus 78 GiB for latent distillation (8.33×).
        Our much smaller absolute tensors show the same scaling mechanism.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim ledger

    | Claim | Compact observation | Assessment |
    |---|---|---|
    | Latent beats frozen features | 97.3% vs 33.9% retention | Aligned |
    | Latent beats direct joint | 97.3% vs 102.6%; latent won 2/8 seeds | Not shown here |
    | Projector fine-tuning recovers segmentation | +11.6 retention points; 8/8 improvements | Aligned mechanism |
    | Latent lowers memory | 7.51× lower total peak at 8 task decoders | Aligned |
    | Consistency survives latent compression | 0.394 vs pixel 0.429; paired CI spans zero | No detectable loss |

    The random-initialization control worsened every matched absolute depth and segmentation comparison, supporting a useful pretrained diffusion prior. Still, CIFAR-10 pretraining is not evidence for the paper's claimed internet-scale domain bridge.

    ## Compute and reproducibility

    Every scientific result came from Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs,
    four GPUs per experiment and 16 peak concurrent. The compute phase took 0.52 elapsed
    wall-clock hours. The fixed command was `bash scripts/run_reproduction.sh`.
    """)
    return


if __name__ == "__main__":
    app.run()
