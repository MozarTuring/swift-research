# SWIFT Research Project - Setup & Plan

## 📚 Project Overview

**Goal**: Implement SWIFT baseline → Extend with learned layer skip policy → Publish to NeurIPS 2025 or ICLR 2026

**Timeline**: 8-10 weeks (starting March 2026)

---

## 🎯 SWIFT Paper Details

**Title**: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration  
**Authors**: Heming Xia, Yongqi Li, Jun Zhang, Cunxiao Du, Wenjie Li  
**Venue**: ICLR 2025  
**arXiv**: 2410.06916  
**Code**: https://github.com/hemingkx/SWIFT  
**License**: Apache-2.0

### Key Insights

1. **Layer Sparsity**: LLMs exhibit task-specific layer sparsity patterns
2. **Plug-and-Play**: No auxiliary models or training required
3. **Adaptive Selection**: Uses random search + interval Bayesian optimization
4. **Performance**: 1.3x-1.6x speedup across diverse tasks

### SWIFT Algorithm

**Two Phases:**

1. **Optimization Phase** (per decoding step):
   - Propose layer set candidates via random search + Bayesian opt
   - Parallel evaluation using LLM-generated tokens as ground truth
   - Select best-performing layer set

2. **Acceleration Phase**:
   - Use selected configuration to skip layers during inference
   - Maintain output distribution (greedy or sampling)

---

## 🚀 Our Extension: Learned Layer Policy

### Research Gap

**SWIFT's Limitation**: Uses heuristic search (random + Bayesian opt) for layer selection  
**Our Solution**: Replace with **learned policy network** trained via RL

### Method Overview

```
Input: [CLS] token embedding (4096 dims) + task metadata
       ↓
Policy Network: MLP (4096 → 512 → 64 → num_layers)
       ↓
Output: P(skip_layer=k) for k ∈ {4, 8, 12, ..., 32}
       ↓
Training: PPO/DQN with reward = acceptance_rate × speedup
```

### Expected Improvements

| Metric | SWIFT (Heuristic) | Ours (Learned) |
|--------|-------------------|----------------|
| Avg Speedup | 1.3x-1.6x | **1.8x-2.0x** |
| Acceptance Rate | 70% | **75-80%** |
| Overhead | Search per step | Tiny policy net (<10M params) |

---

## 📋 Implementation Plan

### Phase 1: Baseline Setup (Week 1) ✅

**Tasks:**
- [ ] Clone SWIFT repository
- [ ] Set up conda environment
- [ ] Download test models (Llama-3-8B, Mistral-7B)
- [ ] Run baseline evaluation
- [ ] Reproduce 1.3x-1.6x speedup

**Commands:**
```bash
git clone https://github.com/hemingkx/SWIFT.git
cd SWIFT
conda create -n swift python=3.9
conda activate swift
pip install -r requirements.txt
```

### Phase 2: Policy Network Design (Week 2-3)

**Tasks:**
- [ ] Design policy network architecture
- [ ] Implement RL training loop (PPO)
- [ ] Create reward function: `r = acc_rate * (1 + α*log(speedup))`
- [ ] Set up training data pipeline

**Architecture:**
```python
class LayerSkipPolicy(nn.Module):
    def __init__(self, embed_dim=4096, num_layers=32):
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers)  # Logits for skip layers
        )
    
    def forward(self, prompt_embed):
        return F.softmax(self.mlp(prompt_embed), dim=-1)
```

### Phase 3: Training & Experiments (Week 4-7)

**Datasets:**
- HumanEval (code generation)
- GSM8K (math reasoning)
- MMLU (multiple choice QA)
- Wikipedia (creative writing)

**Baselines:**
- Standard speculative decoding
- SWIFT (heuristic search)
- Fixed skip depth
- Random selection

**Metrics:**
- Acceptance rate
- Speedup (tokens/sec)
- Perplexity preservation
- Policy overhead

### Phase 4: Analysis & Writing (Week 8-10)

**Analysis:**
- [ ] What patterns does policy learn?
- [ ] Task-specific vs. general policies
- [ ] Error analysis: when does it fail?
- [ ] Ablation studies (network size, reward design)

**Paper Targets:**
- NeurIPS 2025 (May 1, 2025 deadline)
- ICLR 2026 (Aug 2025 deadline)

---

## 🛠️ Environment Setup

### Requirements

```yaml
python: 3.9
cuda: 11.8+
gpu: 1x A100 or 2x RTX 4090
memory: 40GB+ VRAM
```

### Dependencies

```bash
torch>=2.0
transformers>=4.35.0
accelerate
stable-baselines3  # For PPO
ray[rllib]          # Alternative RL framework
```

### Model Checkpoints

```bash
# Download via HuggingFace
llama-3-8b-instruct
mistral-7b-instruct
```

---

## 📊 Evaluation Protocol

### Speedup Calculation

```python
speedup = tokens_per_second_with_swift / tokens_per_second_vanilla
```

### Acceptance Rate

```python
acceptance_rate = accepted_tokens / total_draft_tokens
```

### Distribution Preservation

```python
# KL divergence between vanilla and SWIFT outputs
kl_div = torch.nn.KLDivLoss()(log_swift_probs, vanilla_probs)
```

---

## 🎓 Next Steps

### Immediate (Today)

1. **Clone repository** and set up environment
2. **Run baseline** on small model (Mistral-7B)
3. **Verify speedup** matches paper claims

### This Week

1. Understand SWIFT codebase structure
2. Identify where to inject policy network
3. Design policy network interface

### This Month

1. Implement policy network
2. Train on single task (HumanEval)
3. Show improvement over SWIFT heuristic

---

## 📝 Notes

- SWIFT built on top of Self-SD and EAGLE codebases
- Logo designed by GPT-4 😄
- Supports both greedy and sampling inference
- Can cache layer configurations for repeated patterns

---

**Last Updated**: March 26, 2026  
**Status**: Phase 1 - Environment Setup
