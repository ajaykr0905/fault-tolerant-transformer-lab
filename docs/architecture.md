# Architecture and evidence boundary

```text
Versioned manifest -> deterministic token stream -> decoder-only transformer
                              |                         |
                              |                         v
                              +----------------> atomic checkpoint
                                                        |
                                                        v
                                             restart verification
                                                        |
                                                        v
                                              result.json evidence
```

The CPU smoke path proves model wiring, causal behavior, gradient correctness, atomic checkpoint writes, configuration matching, and deterministic restart. It does not prove GPU throughput, distributed scaling, vLLM behavior, or production readiness.

GPU, serving, LoRA, telemetry, and failure-recovery benchmarks will be added behind separate evidence gates. A claim becomes public only when its configuration, environment, raw result, and limitation note are checked in.
