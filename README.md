# 🧠 LLM From Scratch

This project implements a Transformer-based Language Model (LLM) from scratch using **PyTorch**, inspired by GPT architecture. With ~16 million parameters, it was trained on 300M tokens and evaluated on 100M tokens. The project showcases the full pipeline—**building**, **training**, and **evaluating** a language model end-to-end. My intention was to build a scaled down version of the 124M parmeter GPT-2 Model

---

## 🚀 Features

- **Custom Transformer Implementation**
  - Multi-head self-attention
  - Feedforward layers
  - Layer normalization
  - Causal masking for autoregressive tasks

- **Training Pipeline**
  - Trained on a 300M-token dataset
  - Gradient accumulation for large effective batch size
  - Cosine learning rate schedule with warmup

- **Validation and Evaluation**
  - Evaluated on a 100M-token validation set
  - HellaSwag accuracy evaluation for downstream tasks
  - Text generation with top-k sampling

- **Checkpointing**
  - Periodic checkpoint saving for training resume or inference

- **Device Compatibility**
  - Supports **CPU** and **MPS (Metal Performance Shaders)**

---

## 🏗️ Model Architecture

| Component               | Value             |
|------------------------|-------------------|
| Layers                 | 4                 |
| Attention Heads        | 8                 |
| Embedding Dimension    | 256               |
| Max Sequence Length    | 1024              |
| Vocabulary Size        | 50,304 (GPT-2 BPE)|
| Total Parameters       | ~16 million       |

---

## 📊 Training Details

### 🗂 Dataset
This model was trained on a subset of the **[FineWeb-Edu](https://huggingface.co/datasets/cerebras/SlimPajama-627B)** dataset from Hugging Face. A high-quality, deduplicated, and filtered version of Common Crawl. FineWeb-Edu is curated specifically for educational and reasoning-based tasks, making it well-suited for training language models focused on coherent, structured text.

- **Training**: 300M tokens  
- **Validation**: 100M tokens  
- **Tokenizer**: GPT-2 Byte Pair Encoding (BPE)

### ⚙️ Configuration

- **Micro-Batch Size**: 4  
- **Effective Batch Size**: 32,768 tokens (via gradient accumulation)  
- **Sequence Length**: 512  
- **Optimizer**: AdamW (`weight_decay=0.1`)  
- **Learning Rate**:
  - Max: `6e-4`
  - Min: `6e-5` (10% of max)
  - Warmup Steps: 469  
  - Scheduler: Cosine decay  
- **Training Steps**: 12,207  
- **Checkpoint Frequency**: Every 5,000 steps or final step  

---

## 🧪 Validation and Evaluation

### 📉 Validation Loss

- **Initial Loss**: 10.8677  
- **Final Loss**: 4.3147  
- Validation loss was logged periodically across 12,207 training steps using a 100M-token validation set.

### 🧠 HellaSwag Accuracy

- **Initial Accuracy**: 25.78%  
- **Final Accuracy**: 25.31%  
- Evaluated to benchmark the model’s reasoning and generalization ability on a downstream multiple-choice task.
- Despite improvements in language fluency, the model’s accuracy on HellaSwag—a multiple-choice commonsense reasoning dataset—remained around 25% before and after training, suggesting limited gains in deeper reasoning ability.

### ✍️ Text Generation

Text samples were generated using top-k sampling at both the start and end of training.

#### 🔹 Initial Samples

**Sample 0**: Hello, I'm a language model,mon limbs reactive hurting printers approves approvesmarriagefishformsni get repealing Blitz Gamergate lax commenters repealing Mé Gamergate Gamergate governingencesFemale  
**Sample 1**: Hello, I'm a language model,acio urgedæ audi Post hastilyusercontentinalusercontent speech Ambassador assurance adherent approvesihil coal coalrote switchedbecause lead utter Gry stabilize  
**Sample 2**: Hello, I'm a language model, regrett Millsalsh Hugh stole Pumpkin tr triggeredhellYRchannel BearsEr academics beg bed historians Publishing focalie Lucas Vag FAM FAM  
**Sample 3**: Hello, I'm a language model, hurting imperial basically Joined ace ace desp Bearsganilar Cuttinggan superintendent ticking Nexus needy WriterrepreneoveryNight approvescut Bay85  

---

#### 🔹 Final Samples

**Sample 0**: 'Hello, I'm a language model, where I'm a social-based, cultural culture from the Western world to the Mediterranean. He was a people of a'  
**Sample 1**: 'Hello, I'm a language model, but my father didn't understand it. I started an example of why my husband had some children in England. We're'  
**Sample 2**: 'Hello, I'm a language model, which is based on the three methods of interpreting.'  
**Sample 3**: 'Hello, I'm a language model, but I'm interested in the story that I would like to show you how it gives me, and I'm sure that'  

> These examples show a clear improvement in fluency and structure after training.

---

## ✍️ Text Generation

Supports **top-k sampling** for generating text.

**Example**  
**Input**: `"Hello, I'm a language model,"`  
**Output**: `"Hello, I'm a language model, trained to assist with various tasks and provide helpful information."`

---

## 🔮 Future Improvements

- **Scaling Up**
  - Add more layers, heads, and higher embedding dimensions
  - Train on larger datasets

- **Optimization**
  - Use mixed-precision (FP16) training
  - Test alternative optimizers and schedulers
  - Use CUDA with PyTorch by renting NVIDIA GPUs online to significantly speed up training and improve performance. This allows for faster experimentation and efficient model scaling.


---

## 🙏 Acknowledgments

Inspired by GPT and the broader Transformer family, along with Andrej Karpathy tutorials.  
Thanks to the open-source community and tools like **PyTorch** and **tiktoken** for making this project possible.
