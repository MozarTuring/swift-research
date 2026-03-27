# SWIFT Codebase Analysis & Implementation Guide

## 📁 Repository Structure

```
SWIFT/
├── model/
│   ├── swift/
│   │   ├── modeling_llama.py      # Core SWIFT Llama model with layer skipping
│   │   ├── modeling_mistral.py    # SWIFT Mistral implementation
│   │   ├── kv_cache.py            # KV cache management for layer skipping
│   │   └── utils.py               # Utility functions for tree decoding
│   └── __init__.py
├── evaluation_llama/
│   ├── inference_swift.py         # Main inference script with SWIFT
│   ├── eval.py                    # Evaluation framework
│   └── prepare_data.py            # Dataset preparation
├── evaluation_mistral/
│   └── (similar structure for Mistral)
├── requirements.txt
└── README.md
```

---

## 🔧 Core SWIFT Mechanism

### 1. Layer Skipping Architecture

**Key File**: `model/swift/modeling_llama.py`

```python
class LlamaForCausalLM:
    def __init__(self, config):
        self.attn_skip_layer_id_set = None  # Layers to skip in attention
        self.mlp_skip_layer_id_set = None   # Layers to skip in MLP
    
    def set_skip_layers(self, attn_layers, mlp_layers):
        """Configure which layers to skip"""
        self.attn_skip_layer_id_set = attn_layers
        self.mlp_skip_layer_id_set = mlp_layers
```

**Forward Pass with Skipping**:
```python
for i, layer in enumerate(self.model.layers):
    if i in self.attn_skip_layer_id_set:
        # Skip attention computation
        continue
    if i in self.mlp_skip_layer_id_set:
        # Skip MLP computation
        continue
    # Normal computation
    hidden_states = layer(hidden_states, attention_mask)
```

### 2. Tree-Based Speculative Decoding

**Key File**: `model/swift/utils.py`

```python
def swift_draft(model, input_ids, ...):
    """
    Generate draft tokens using SWIFT (layer-skipped model)
    Returns: swift_logits, top1_prob
    """
    # Forward pass with layer skipping
    outputs = model(input_ids, ...)
    
    # Extract top-1 predictions at each position
    top1_prob = torch.topk(outputs.logits, 1)
    
    return outputs.logits, top1_prob

def generate_candidates(swift_logits, tree_indices, ...):
    """
    Build tree of candidate tokens from draft
    """
    # Creates a tree structure of speculative tokens
    # Allows parallel verification
```

### 3. Layer Set Optimization (The Heuristic Part)

**Key File**: `evaluation_llama/inference_swift.py`

```python
def swift_optimization(model, generated_tokens, past_key_values, 
                      statistics, optimizer, utility):
    """
    Optimize layer skip configuration using:
    1. Random search (initial exploration)
    2. Bayesian optimization (refinement)
    """
    
    # Random search phase
    if statistics["opt_iter"] < statistics["bayes_interval"]:
        # Randomly sample layer configurations
        candidate_layers = random_layer_selection()
    
    # Bayesian optimization phase
    else:
        # Use GP-based optimization
        candidate_layers = optimizer.suggest_params()
    
    # Evaluate candidate configuration
    score = evaluate_layer_set(candidate_layers, model, generated_tokens)
    
    # Update best configuration
    if score > statistics["origin_score"]:
        statistics["origin_score"] = score
        model.set_skip_layers(candidate_layers)
```

**Optimization Parameters**:
```python
statistics = {
    "skip_ratio": 0.45,           # Target: skip 45% of layers
    "opt_interval": 1,            # Optimize every 1 token
    "bayes_interval": 25,         # Switch to Bayes after 25 random samples
    "max_opt_iter": 1000,         # Max optimization iterations
    "max_score": 0.95,            # Early stop if score > 0.95
    "context_window": 32,         # Optimize after 32 tokens generated
}
```

---

## 🎯 Our Extension: Learned Policy Network

### Where to Inject the Policy

**Current Flow** (Heuristic):
```
Generate token → Check if opt_interval reached → Random/Bayes search → Update layers
```

**New Flow** (Learned):
```
Generate token → Check if opt_interval reached → Policy network → Update layers
```

### Implementation Plan

#### Step 1: Policy Network Module

Create `model/swift/policy_network.py`:

```python
import torch
import torch.nn as nn

class LayerSkipPolicy(nn.Module):
    """
    Learned policy for layer skip selection.
    Replaces random search + Bayesian optimization in SWIFT.
    """
    
    def __init__(self, embed_dim=4096, num_layers=32, hidden_dim=512):
        super().__init__()
        
        # Input: [CLS] token embedding + task metadata
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 10, hidden_dim),  # +10 for task metadata
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_layers * 2)  # *2 for attn + mlp
        )
        
    def forward(self, cls_embedding, task_embedding=None):
        """
        Args:
            cls_embedding: [batch, embed_dim] - [CLS] token from prompt
            task_embedding: [batch, 10] - one-hot task type + metadata
        
        Returns:
            attn_skip_probs: [batch, num_layers] - P(skip attention at layer i)
            mlp_skip_probs: [batch, num_layers] - P(skip MLP at layer i)
        """
        if task_embedding is not None:
            x = torch.cat([cls_embedding, task_embedding], dim=-1)
        else:
            x = cls_embedding
        
        logits = self.mlp(x)
        
        # Split into attention and MLP skip probabilities
        attn_logits, mlp_logits = logits.chunk(2, dim=-1)
        
        attn_probs = torch.sigmoid(attn_logits)  # Independent probabilities
        mlp_probs = torch.sigmoid(mlp_logits)
        
        return attn_probs, mlp_probs
    
    def sample_skip_layers(self, cls_embedding, skip_ratio=0.45):
        """
        Sample layer configuration based on policy probabilities.
        Ensures approximately skip_ratio of layers are skipped.
        """
        attn_probs, mlp_probs = self.forward(cls_embedding)
        
        # Sample layers respecting skip ratio
        attn_layers = self._sample_layers_with_ratio(attn_probs, skip_ratio)
        mlp_layers = self._sample_layers_with_ratio(mlp_probs, skip_ratio)
        
        return attn_layers, mlp_layers
    
    def _sample_layers_with_ratio(self, probs, target_ratio):
        """Sample layers to achieve target skip ratio"""
        num_layers = probs.shape[-1]
        num_to_skip = int(num_layers * target_ratio)
        
        # Use probs as weights for sampling
        indices = torch.arange(num_layers)
        sampled = torch.multinomial(probs, num_to_skip, replacement=False)
        
        return sampled.cpu().numpy()
```

#### Step 2: RL Training with PPO

Create `training/train_policy.py`:

```python
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class SwiftPolicyEnv(gym.Env):
    """
    Gym environment for training layer skip policy.
    """
    
    def __init__(self, model, tokenizer, dataset, max_steps=512):
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        # Action space: which layers to skip (binary for each layer)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(64,), dtype=np.float32  # 32 attn + 32 mlp
        )
        
        # Observation space: [CLS] embedding + task metadata
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(4106,), dtype=np.float32  # 4096 + 10
        )
        
    def reset(self):
        """Sample new prompt and return [CLS] embedding"""
        prompt = random.choice(self.dataset)
        
        # Get [CLS] embedding from model
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.base_model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[0, 0]  # [CLS] token
        
        # Add task metadata
        task_embedding = self._get_task_embedding(prompt)
        
        obs = torch.cat([cls_embedding, task_embedding]).numpy()
        
        self.current_prompt = prompt
        self.step_count = 0
        
        return obs
    
    def step(self, action):
        """
        Apply layer skip configuration and compute reward.
        
        Args:
            action: [64] - skip probabilities for each layer
        
        Returns:
            observation, reward, done, info
        """
        # Convert action to layer sets
        attn_skip = np.where(action[:32] > 0.5)[0]
        mlp_skip = np.where(action[32:] > 0.5)[0]
        
        # Apply to model
        self.model.set_skip_layers(attn_skip, mlp_skip)
        
        # Generate tokens and measure performance
        acceptance_rate, speedup = self._evaluate_generation()
        
        # Reward: acceptance_rate * (1 + alpha * log(speedup))
        alpha = 0.5
        reward = acceptance_rate * (1 + alpha * np.log(speedup + 1e-6))
        
        done = False
        info = {"acceptance_rate": acceptance_rate, "speedup": speedup}
        
        # Return same observation (context doesn't change during generation)
        return self.observation, reward, done, info
    
    def _evaluate_generation(self, max_tokens=32):
        """Generate tokens and compute metrics"""
        # Run SWIFT forward pass
        # Compare with vanilla generation for acceptance rate
        # Measure tokens/sec for speedup
        pass
```

#### Step 3: Training Loop

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Initialize environment
env = SwiftPolicyEnv(model, tokenizer, train_dataset)

# Initialize policy network
policy_net = LayerSkipPolicy(embed_dim=4096, num_layers=32)

# Create PPO agent
model_ppo = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_swift_logs/"
)

# Train
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./ppo_swift_checkpoints/"
)

model_ppo.learn(
    total_timesteps=100000,
    callback=checkpoint_callback,
    progress_bar=True
)

# Save
model_ppo.save("./ppo_swift_final")
```

---

## 📊 Evaluation Protocol

### Baselines to Compare Against

1. **Vanilla generation** (no speculation)
2. **Standard speculative decoding** (with draft model)
3. **SWIFT with heuristic** (random + Bayes optimization)
4. **Fixed skip depth** (skip same layers always)
5. **Random selection** (random layers each step)
6. **Ours: Learned policy** (PPO-trained)

### Metrics

```python
metrics = {
    "speedup": tokens_per_second_ours / tokens_per_second_vanilla,
    "acceptance_rate": accepted_tokens / total_draft_tokens,
    "perplexity": compute_perplexity(generated_text, test_set),
    "policy_overhead": policy_inference_time / total_inference_time,
    "bleu_score": compute_bleu(generated, reference),
    "rouge_score": compute_rouge(generated, reference)
}
```

### Datasets

| Dataset | Task Type | Samples | Purpose |
|---------|-----------|---------|---------|
| HumanEval | Code generation | 164 | Training & eval |
| GSM8K | Math reasoning | 1319 | Training & eval |
| MMLU | Multiple choice QA | 14,042 | Zero-shot eval |
| Wikipedia | Creative writing | 1000 | Diversity |
| Alpaca | Instruction following | 52K | Training |

---

## 🚀 Quick Start Commands

### 1. Set Up Environment

```bash
cd SWIFT
conda create -n swift python=3.9 -y
conda activate swift
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install stable-baselines3[extra] gymnasium
```

### 2. Download Models

```bash
# Using HuggingFace CLI
huggingface-cli download meta-llama/Llama-3-8B-Instruct --local-dir ./models/llama-3-8b
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir ./models/mistral-7b
```

### 3. Run SWIFT Baseline

```bash
cd evaluation_llama

python3 inference_swift.py \
    --model-path ../models/llama-3-8b \
    --model-id llama-3-8b \
    --task-name humaneval \
    --data-num 20 \
    --optimization \
    --bayes \
    --skip-ratio 0.45 \
    --opt-interval 1 \
    --temperature 0.7 \
    --top-p 0.9
```

### 4. Train Policy Network

```bash
cd ../training

python3 train_policy.py \
    --model-path ../models/llama-3-8b \
    --dataset humaneval \
    --total-timesteps 100000 \
    --save-dir ./checkpoints/ppo_swift
```

### 5. Evaluate Learned Policy

```bash
python3 evaluate_policy.py \
    --model-path ../models/llama-3-8b \
    --policy-checkpoint ./checkpoints/ppo_swift \
    --task-name humaneval \
    --data-num 100 \
    --compare-baseline swift-heuristic
```

---

## 📝 Next Steps (This Week)

### Day 1-2: Environment Setup
- [x] Clone repository
- [ ] Install dependencies
- [ ] Download Llama-3-8B model
- [ ] Run SWIFT baseline on small dataset

### Day 3-4: Understand Codebase
- [ ] Read through `modeling_llama.py` completely
- [ ] Trace through `swift_forward` function
- [ ] Understand tree decoding mechanism
- [ ] Document layer skipping logic

### Day 5-7: Implement Policy Network
- [ ] Create `policy_network.py` module
- [ ] Integrate with SWIFT forward pass
- [ ] Test with simple rule-based policy
- [ ] Verify it works end-to-end

---

## 🐛 Known Issues

1. **Memory usage**: SWIFT can be memory-intensive due to tree decoding
2. **GPU compatibility**: Tested on A100, may need adjustments for consumer GPUs
3. **Batch size**: Only supports batch size 1 currently
4. **Layer skipping**: First and last layers are always kept (hardcoded)

---

## 📚 References

- **SWIFT Paper**: https://arxiv.org/abs/2410.06916
- **SWIFT Code**: https://github.com/hemingkx/SWIFT
- **PPO Implementation**: https://stable-baselines3.readthedocs.io/
- **Speculative Decoding**: https://arxiv.org/abs/2211.17192

---

**Last Updated**: March 26, 2026  
**Status**: Ready for implementation
