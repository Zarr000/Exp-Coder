# Roadmap v2

## Expera AI Phase 11+ Roadmap

### Completed (Phase 10)

- ✅ Dataset Pipeline
- ✅ Model Configurations (120M, 350M, 1B, 3B)
- ✅ Pretraining Scripts
- ✅ Instruction Tuning
- ✅ Preference Optimization (DPO, ORPO)
- ✅ Coding Specialization
- ✅ Image System Integration
- ✅ Hybrid Runtime
- ✅ Agent System
- ✅ Evaluation
- ✅ Quantization
- ✅ Release System
- ✅ Documentation

### Phase 11: Model Training

Target: Train actual model weights for Expera-Coder-350M-v1

#### Milestones
- [ ] Train 120M baseline
- [ ] Train 350M full model
- [ ] Run evaluation benchmarks
- [ ] Publish model weights

#### Requirements
- GPU cluster (8x A100 minimum)
- Dataset downloads
- ~9 hours training time

### Phase 12: Production

Deploy and test the model

#### Milestones
- [ ] Deploy local API
- [ ] Deploy VPS API
- [ ] Integration tests
- [ ] Performance optimization

### Future Enhancements

#### Larger Models
- [ ] 3B model training
- [ ] 7B model training
- [ ] MoE variants

#### Enhanced Capabilities
- [ ] Vision understanding
- [ ] Longer context (8K → 32K)
- [ ] Better multilingual support
- [ ] Function calling improvements

#### Advanced Agents
- [ ] Multi-agent coordination
- [ ] Planning improvements
- [ ] Better tool use
- [ ] Self-improvement loop

#### Ecosystem
- [ ] VS Code extension
- [ ] JetBrains plugins
- [ ] Web UI improvements
- [ ] Mobile support

## Priority Matrix

| Priority | Task | Impact |
|----------|------|-------|
| P0 | Train 350M | Core feature |
| P1 | Deploy API | Usability |
| P1 | Evaluate | Quality |
| P2 | Train 1B | Performance |
| P2 | 3B model | Capability |
| P3 | VS Code | Ecosystem |

## Contribution

Contributions welcome! Areas needing help:

1. Dataset curation
2. Training infrastructure
3. Evaluation benchmarks
4. Documentation
5. Testing

## Contact

- GitHub Issues: Report bugs/feature requests
- Discussions: Ask questions
- Pull Requests: Submit changes