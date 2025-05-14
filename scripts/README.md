# DNA-Protein Interaction Prediction Agent (DPI-Agent)

**Version:** 0.0.1 (or your current version)
**Author/Organization:** [Cong Liu / Wei Wang Lab / UCSD]
**Contact:** [wargm003@gmail.com]

## Table of Contents
1.  [Overview](#overview)
2.  [Features](#features)
3.  [System Requirements](#system-requirements)
4.  [Installation](#installation)
    *   [Prerequisites](#prerequisites)
    *   [Cloning the Repository](#cloning-the-repository)
    *   [Setting up the Environment](#setting-up-the-environment)
    *   [Installing Dependencies](#installing-dependencies)
5.  [Downloading Large Resources](#downloading-large-resources)
6.  [Configuration](#configuration)
    *   [Essential Path Configuration](#essential-path-configuration)
    *   [API Keys](#api-keys)
7.  [Usage](#usage)
    *   [Input Data Format](#input-data-format)
    *   [Command-Line Arguments](#command-line-arguments)
    *   [Example Invocation](#example-invocation)
8.  [Output Files](#output-files)
9.  [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)
12. [Citation](#citation)
13. [Acknowledgements](#acknowledgements)

## Overview

The DPI-Agent is a sophisticated computational pipeline designed to predict interactions between DNA sequences and proteins. It leverages a multi-modal approach, integrating evidence from:
*   **Direct Motif Analysis:** Searching for known binding motifs of the target protein in the DNA sequence using CisBP database and FIMO scans.
*   **Indirect Motif Analysis:** Identifying TF interactors of query protein (via STRING DB) and find if indirect binding exists.
*   **Transformer-based Prediction:** Utilizing a in-house SOTA deep learning model trained on ChIP-seq data to predict interaction probability.
*   **Large Language Model (LLM) Synthesis:** Employing an LLM (e.g., Gemini, or open-source Hugging Face models) to synthesize all gathered evidence and provide a final interaction prediction (0 or 1) along with a detailed explanation.
*   **Confidence level estimation:** Using rule-based method to estimate how confident the prediction is.


## Features

*   Modular workflow managed by LangGraph.
*   Support for multiple LLM backends: Google Gemini, open source Hugging Face models .
*   Comprehensive motif analysis using CisBP and FIMO.
*   Protein-protein interaction data integration from STRING DB.
*   On-the-fly DNABERT embedding calculation if pre-computed embeddings are not found.
*   Rule-based confidence scoring for LLM predictions.
*   Configurable prompt styles for LLM interaction.
*   Detailed JSON outputs for both summary and comprehensive results.

## System Requirements

*   **Operating System:** Linux (recommended due to FIMO dependency). macOS might work with FIMO installed. Windows may require WSL.
*   **Python:** 3.8 - 3.10 (tested with 3.10.17)
*   **CPU/GPU:**
    *   CPU is sufficient for most tasks, but LLM inference (especially local HF models) and Transformer model predictions will be significantly faster on a CUDA-enabled GPU.
    *   DNABERT embedding calculation also benefits from a GPU.
*   **RAM:** Minimum 16GB, 32GB+ recommended, especially for larger LLMs or large batch processing.
*   **Disk Space:** Several GBs for the Python environment and scripts. **Significant additional space (tens to hundreds of GBs) is required for the large resources (models, embeddings, databases) - see [Downloading Large Resources](#downloading-large-resources).**

## Installation

### Prerequisites

1.  **MEME Suite (for FIMO):**
    FIMO is required for motif scanning. Install the MEME Suite from [http://meme-suite.org/doc/install.html](http://meme-suite.org/doc/install.html). Ensure `fimo` is in your system's `PATH`.
    ```bash
    # Example check
    fimo --version
    ```
2.  **Git:** For cloning the repository.

### Cloning the Repository
```bash
git clone [https://github.com/cong-003/DPI]
cd [REPOSITORY_NAME] # e.g., dpi-agent
```

### Setting up the Environment
It is highly recommended to use a virtual environment:

**Using `venv` (Python's built-in):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate  # On Windows
```

**Using `conda`:**
```bash
conda create -n dpi_env python=3.9 # Or your preferred Python version
conda activate dpi_env
```

### Installing Dependencies
A `requirements.txt` file should be provided in the repository.
```bash
pip install -r requirements.txt
```
If using Unsloth and a compatible GPU, Unsloth will be installed via `requirements.txt`. Ensure you have the necessary CUDA toolkit version compatible with PyTorch and Unsloth.

## Downloading Large Resources

Due to their size, the following resources are not included in the GitHub repository and must be downloaded separately. Place them in the specified directory structure relative to the main project directory (e.g., `dpi-agent/data/`).

**Recommended Directory Structure:**
```
dpi-agent/
├── test4_eval.py
├── README.md
├── requirements.txt
├── data/
│   ├── dnabert_models/
│   │   ├── 6-new-12w-0/          # For DNABERT_KMER=6
│   │   │   └── ... (model files)
│   │   └── src/transformers/dnabert-config/bert-config-6/
│   │       ├── config.json
│   │       └── vocab.txt
│   ├── transformer_models/
│   │   └── main_singletask_Encode3and4_all_847_proteins-....pt
│   ├── embeddings/
│   │   ├── dna_embeddings.duckdb
│   │   └── all_human_tfs_protein_embedding_mean.pkl
│   ├── cisbp_database/
│   │   ├── pwms_all_motifs/
│   │   │   └── ... (*.txt PWM files)
│   │   └── TF_Information.txt # Or TF_Information_all_motifs.txt
│   └── training_lists/
│       └── proteins.txt # List of training proteins for transformer
└── ... (other files)
```

**Resource List & Download Links:**

1.  **DNABERT Model & Config (Example for 6-mer):**
    *   **Description:** Pretrained DNABERT model weights and configuration.
    *   **Download:**
        *   Model Files: `[Link to DNABERT 6-mer model files on Zenodo/Your_Cloud_Storage]` -> Place in `data/dnabert_models/6-new-12w-0/`
        *   Config & Vocab: `[Link to DNABERT 6-mer config/vocab on Zenodo/Your_Cloud_Storage]` -> Place in `data/dnabert_models/src/transformers/dnabert-config/bert-config-6/`
    *   *Note: Adjust paths in `test4_eval.py` or use environment variables if your k-mer or paths differ.*

2.  **DPI Transformer Model:**
    *   **Description:** The pre-trained transformer model for DPI prediction.
    *   **Download:** `[Link to your transformer_model.pt on Zenodo/Your_Cloud_Storage]` -> Place in `data/transformer_models/`

3.  **DNA Embeddings Database:**
    *   **Description:** DuckDB database containing pre-computed DNA sequence embeddings.
    *   **Download:** `[Link to dna_embeddings.duckdb on Zenodo/Your_Cloud_Storage]` -> Place in `data/embeddings/`

4.  **Protein Embeddings:**
    *   **Description:** Pickle file with pre-computed protein embeddings.
    *   **Download:** `[Link to all_human_tfs_protein_embedding_mean.pkl on Zenodo/Your_Cloud_Storage]` -> Place in `data/embeddings/`

5.  **CisBP Motif Database:**
    *   **Description:** TF Information file and PWM files from CisBP (filtered for *Homo sapiens*).
    *   **Download:**
        *   TF Information: `[Link to TF_Information.txt on Zenodo/Your_Cloud_Storage]` -> Place in `data/cisbp_database/`
        *   PWMs: `[Link to pwms_all_motifs.zip/tar.gz on Zenodo/Your_Cloud_Storage]` -> Extract into `data/cisbp_database/pwms_all_motifs/`

6.  **Transformer Training Protein List:**
    *   **Description:** Text file listing proteins used in the transformer model's training set.
    *   **Download:** `[Link to proteins.txt on Zenodo/Your_Cloud_Storage]` -> Place in `data/training_lists/`

**(Optional: Provide a download script `download_resources.sh` or `download_resources.py` to automate this.)**

## Configuration

### Essential Path Configuration

The script `test4_eval.py` contains several hardcoded paths for the resources mentioned above. **You MUST update these paths if you do not use the exact directory structure and filenames as suggested, or if your base directory for `data` differs.**

It is highly recommended to modify the script to use **environment variables** for these paths or pass them as **command-line arguments**.

**Key path constants in `test4_eval.py` to check/modify:**
*   `DNABERT_MODEL_DIR`
*   `DNABERT_CONFIG_PATH`
*   `DNABERT_VOCAB_PATH`
*   `FINETUNED_LOCAL_MODEL_PATH_ID` (if using a specific local fine-tuned HF model identified by path)
*   `TRANSFORMER_MODEL_PATH`
*   `DNA_EMB_DB_PATH`
*   `PRO_EMB_PATH`
*   `TRAINING_PROTEIN_LIST_PATH`
*   `CISBP_BASE_DIR`

**Example: Using Environment Variables (Recommended)**
Modify the script:
```python
# test4_eval.py
# Before:
# TRANSFORMER_MODEL_PATH = '/new-stg/home/cong/DPI/scripts/model2_Transformer/v5/output/model/main_singletask_Encode3and4_all_847_proteins-....pt'
# After:
TRANSFORMER_MODEL_PATH = os.environ.get('DPI_TRANSFORMER_MODEL_PATH', '/default/path/if/not/set.pt')
```
Then, before running the script:
```bash
export DPI_TRANSFORMER_MODEL_PATH="/path/to/your/data/transformer_models/model.pt"
# ... set other environment variables similarly
python test4_eval.py --evaluation-file ...
```

### API Keys

*   **Google Gemini:** If using a `gemini/` LLM model, set the `GOOGLE_API_KEY` environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
*   **Hugging Face:** If using gated models from Hugging Face Hub, you might need to provide a token via the `--hf-token` argument or by logging in via `huggingface-cli login`.

## Usage

The script is designed for batch evaluation using an input file.

### Input Data Format
The evaluation file (`--evaluation-file`) should be a `.tsv` (tab-separated) or `.pkl` (pickled Pandas DataFrame) file with the following columns:
*   `dna`: The DNA sequence string.
*   `protein`: The protein name (e.g., gene symbol like "SOX2").
*   `label`: The ground truth interaction label (0 for no interaction, 1 for interaction).

### Command-Line Arguments

*   `--evaluation-file FILE`: (Required) Path to the evaluation dataset (.tsv or .pkl).
*   `--llm-model MODEL_ID`: LLM to use. Prefixes:
    *   `gemini/`: e.g., `gemini/gemini-1.5-flash-latest`
    *   `ollama/`: e.g., `ollama/mistral` (requires Ollama server running)
    *   `hf/`: e.g., `hf/unsloth/mistral-7b-bnb-4bit` or `hf//path/to/local/model`
    *   Default: Varies based on available libraries.
*   `--ollama-url URL`: URL for Ollama API if using `ollama/` model. Default: `http://localhost:11434/api/generate`.
*   `--force-cpu`: Force CPU usage even if CUDA is available.
*   `--hf-token TOKEN`: Hugging Face API token (for gated models).
*   `--output-dir DIR`: Directory to save output JSON files. Default: `./dpi_agent_outputs`.
*   `--prompt-style STYLE`: Prompt style for LLM. Choices: `verbose`, `concise`, `transformer-priority`. Default: `verbose`.
*   `--random-state INT`: Random state for shuffling evaluation data. Default: 13.
*   `--limit INT`: Limit processing to the first N samples from the (shuffled) evaluation file.
*   `--api-delay FLOAT`: Delay in seconds between Gemini API calls. Default: 1.5.

### Example Invocation

```bash
python test4_eval.py \
    --evaluation-file /path/to/your/eval_data.tsv \
    --llm-model hf/unsloth/mistral-7b-bnb-4bit \
    --output-dir ./my_dpi_results \
    --prompt-style concise \
    --limit 100
```

## Output Files

For each input sample, the agent generates two JSON files in the specified output directory (within a subdirectory named after the LLM and prompt style):

1.  **`sample<index>_<protein_name>_<llm_safe_name>_simple.json`:**
    Contains a summary of the prediction:
    *   `protein_name`
    *   `dna_sequence_length`
    *   `ground_truth_label`
    *   `predicted_label` (LLM's vote: 0 or 1)
    *   `confidence_score` (Rule-based confidence in the prediction)
    *   `llm_explanation` (Parsed explanation from the LLM)
    *   `error` (Any error message, or `null` if successful)

2.  **`sample<index>_<protein_name>_<llm_safe_name>_comp.json`:**
    Contains the complete agent state after processing, including all intermediate results from motif fetching, scanning, STRING DB queries, transformer predictions, raw LLM response, etc. This file is useful for detailed debugging and analysis.

Additionally, an aggregated summary file is created:
*   **`<llm_safe_name>_prompt_<style>_aggregated_simple_results.json`:**
    Located in the parent `--output-dir`. This file is an array of all the `_simple.json` contents, useful for overall performance evaluation.

## Troubleshooting

*   **`fimo: command not found`**: Ensure MEME Suite is installed and `fimo` is in your system `PATH`.
*   **Python `ModuleNotFoundError`**: Make sure you have activated the correct virtual environment and installed all packages from `requirements.txt`.
*   **Resource File Not Found Errors**: Double-check the paths in `test4_eval.py` or ensure your environment variables for paths are correctly set and point to the downloaded resources. Verify the directory structure.
*   **CUDA/GPU Issues**:
    *   Ensure PyTorch is installed with CUDA support (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` - adjust `cu118` to your CUDA version).
    *   Check `nvidia-smi` for GPU status.
    *   If using Unsloth, ensure your GPU architecture is supported and drivers are up-to-date.
*   **LLM API Errors (Gemini, Ollama)**:
    *   Gemini: Check `GOOGLE_API_KEY` and network connectivity. Rate limits can occur; adjust `--api-delay`.
    *   Ollama: Ensure the Ollama server is running, accessible at the specified URL, and the model is pulled (`ollama pull mistral`).
*   **Hugging Face Model Loading Errors**:
    *   Check model ID or path.
    *   For private/gated models, ensure `--hf-token` is provided or you're logged in.
    *   Memory issues: Large models require significant RAM/VRAM. Try a smaller model or use quantization.

## Contributing
[Optional: Add guidelines if you expect contributions, e.g., pull request process, coding standards.]
We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
This project is licensed under the [NAME OF LICENSE, e.g., MIT License]. See the `LICENSE` file for details.
(You should add a `LICENSE` file to your repository, e.g., from choosealicense.com)

## Citation
[Optional: If this work is associated with a publication or preprint, please add citation information here.]
If you use this software in your research, please cite:
> [Your Name/Lab. (Year). DPI-Agent: A Multi-modal Agent for DNA-Protein Interaction Prediction. (Version X.Y.Z). Available at: [Link to your GitHub Repo]]
> [Any relevant paper or preprint]

## Acknowledgements
*   The LangGraph team for their excellent library.
*   Developers of Unsloth, Hugging Face Transformers, PyTorch.
*   Data providers: CisBP, STRING DB, ENCODE Project.
*   [Any other individuals or groups you wish to thank.]

```

**Key things for YOU to do:**

1.  **Replace all placeholders** like `[Your Name / Your Lab / Your Organization]`, `[URL_OF_YOUR_GITHUB_REPOSITORY]`, `[Link to ... on Zenodo/Your_Cloud_Storage]`, `[NAME OF LICENSE]`, etc.
2.  **Create a `LICENSE` file** in your repository (e.g., choose MIT or Apache 2.0 from [https://choosealicense.com/](https://choosealicense.com/)).
3.  **Generate `requirements.txt`:**
    ```bash
    # Activate your virtual environment
    pip freeze > requirements.txt
    ```
    Review and clean up this file if necessary (sometimes it includes dependencies of dependencies which isn't strictly needed but usually harmless).
4.  **Verify all paths and instructions.** Test the installation and resource download steps yourself from a clean environment to ensure they are accurate.
5.  **Consider adding a small example evaluation file** to your repository so users can test the setup quickly.
6.  **Strongly consider making the paths in your Python script configurable via environment variables or command-line arguments** instead of hardcoding them. This is the most important step for shareability. Your README currently reflects the hardcoded paths and suggests this change.
