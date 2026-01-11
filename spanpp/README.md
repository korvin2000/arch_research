# SpanPP

**SpanPP — Span Plus Plus**

SpanPP is a super-resolution architecture focused on two main goals: making early training more stable, and improving final image quality.
This is achieved through richer layer reparameterization and by training the model on multiple upscaling factors from the start.

The training pipeline includes an IGConv-based scheme that allows the model to learn across multiple scales (`N` different upscales) at once. This gives you two benefits:

1. The model generalizes better.
2. You can reuse the same model in more scenarios (×2, ×3, ×4, etc.) without training a completely separate model for each scale.

Despite that, inference speed is not affected: at inference time, the convolution kernel is recalculated for the requested scale, and then the model just uses a direct `pixelshuffle` path. So runtime stays lightweight.

---

## Training your own model

1. Take the `traiNNer` folder from this repo and merge it into the same folder in your `traiNNer-redux` project.
2. In your training config, replace the default `network_g` block with:

```yaml
network_g:
  type: SpanPP
  scale_list: [1, 2, 3, 4]
  eval_base_scale: 2
```

Parameter details:

* `scale_list` — the list of scales the model will be trained on.
  In the example above, the model is trained for ×1, ×2, ×3, ×4.

* `eval_base_scale` — the scale that will be used for validation during training (the scale used to run validation and compute metrics).

Important: your dataset must be prepared for the maximum scale in `scale_list`.
So if you’re training up to ×4, your LR/HR pairs and crops should match ×4.
Also, in your config, the global `scale` parameter must be set to that maximum scale (×4 in this example).

---

## Exporting to ONNX

Right now, the most reliable ONNX export flow is:

1. Set `eval_base_scale` to one of the values from `scale_list`.
2. Export the model to ONNX.
3. Repeat for each scale.

In the end you get N ONNX models — one per scale (×2, ×3, ×4, etc.).

This is usually the most compatible approach: most runtimes expect a fixed internal upscale factor, so having one ONNX per scale avoids weird framework-specific hacks and just works.

---

USDT TON: ```UQD66mA1FoZ5U49trzfgvdNIB_NsXBaYFTJi88rRaXIzT_LJ```
