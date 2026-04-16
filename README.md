# NER Benchmark

A dissertation research project benchmarking Named Entity Recognition (NER) models on the [AMI meeting corpus](https://groups.inf.ed.ac.uk/ami/corpus/), with a focus on how ASR (Automatic Speech Recognition) errors propagate to NER performance.

## Pipeline

```
Raw AMI corpus (XML + audio)
  → Ground-truth extraction
  → JSONL manifests
  → ASR predictions
  → NER predictions
  → Jupyter analysis
```

## Setup

```bash
pip install -r requirements.txt
```

Some backends require additional packages: `faster-whisper`, `whisperx`, `speechbrain`, `gliner`, `presidio`, `spacy`.

LLM-based NER requires a local [Ollama](https://ollama.com) instance at `http://localhost:11434`.

## Usage

### 1. Prepare data

```bash
python scripts/dataset/prepare_dataset.py
```

### 2. Run ASR

```bash
python scripts/asr_models/run_all_asr.py
```

### 3. Run NER

```bash
# Transformer (BERT-family)
python scripts/ner_models/transformer_ner/run_all_transformer_ner.py

# BiLSTM-CRF (Flair)
python scripts/ner_models/bilstm_crf_ner/run_all_bilstm_crf_ner.py

# LLM-prompted (Ollama)
python scripts/ner_models/llm_prompted_ner/run_all_llm_ner.py

# Zero-shot (GLiNER, Presidio)
python scripts/ner_models/zero_shot_ner/run_zero_shot_ner.py --backend gliner ...

# Speech-aware (WhisperNER)
python scripts/ner_models/speech_aware_ner/run_whisperner_ner.py
```

### 4. Analyse results

Open the notebooks in `notebooks/` — `asr_analysis.ipynb` and `ner_analysis.ipynb`.

## NER Models Covered

| Category | Examples |
|---|---|
| Transformer | BERT, RoBERTa, DeBERTa |
| BiLSTM-CRF | Flair |
| LLM-prompted | Llama, Mistral, Qwen, Phi |
| Zero-shot | GLiNER, Presidio |
| Rule-based | Philter |
| Speech-aware | WhisperNER |
