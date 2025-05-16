# DNA-Protein Interaction Prediction Agent (DPI-Agent)

## Overview

The DPI-Agent is a llm-based pipeline designed to predict interactions between DNA and protein. It leverages a multi-modal approach, integrating evidence from:
*   **Direct Motif Analysis:** Searching for known binding motifs of the target protein in the DNA sequence using CisBP database and FIMO scans.
*   **Indirect Motif Analysis:** Identifying TF interactors of query protein (via STRING DB) and find if indirect binding exists.
*   **Transformer-based Prediction:** Utilizing SOTA deep learning model trained on ChIP-seq data to predict interaction probability.
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

2.  **Git:** For cloning the repository.

### Cloning the Repository
```bash
git clone [https://github.com/cong-003/DPI-agent]
cd [DPI-agent] 
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
conda create -n dpi_env python=3.10
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
│   ├── transformer_models/
│   │   └── main_singletask_Encode3and4_all_847_proteins-....pt
│   ├── embeddings/
│   │   ├── dna_embeddings_test_demo_100.duckdb
│   │   └── all_human_tfs_protein_embedding_mean.pkl
│   ├── cisbp_database/
│   │   ├── pwms_all_motifs/
│   │   │   └── ... (*.txt PWM files)
│   │   └── TF_Information.txt
│   └── training_lists/
│       └── proteins.txt # List of training proteins for transformer
│   └── evaluation_files/
│       └── test_min10_rs_3_1000_pairs.pkl # example evaluation file
└── ... (other files)
```

**Resource List & Download Links:**

1.  **DNA Embeddings:**
    *   **Description:** DuckDB database containing pre-computed DNA sequence embeddings from DNABERT.
    *  [**Download:**](https://drive.google.com/drive/folders/1x4TKNuO42AYeopGIVH5i8ICsmgY9J_8z) -> Place in `data/embeddings/`

2.  **Protein Embeddings:**
    *   **Description:** Pickle file with pre-computed protein embeddings from AlphaFold.
    *   [**Download:**](https://drive.google.com/drive/folders/1x4TKNuO42AYeopGIVH5i8ICsmgY9J_8z) -> Place in `data/embeddings/`

3.  **CisBP Motif Database:**
    *   **Description:** TF Information file and PWM files from CisBP (filtered for *Homo sapiens*).
    *   [**Download:**](https://github.com/cong-003/DPI-agent/blob/main/data/Homo_sapiens_2025_05_16_4_38_am.zip)
        *   TF Information: meta data for the all the TFs -> Place in `data/cisbp_database/`
        *   PWMs: Extract into `data/cisbp_database/pwms_all_motifs/`

4.  **Transformer Training Protein List:**
    *   **Description:** Text file listing proteins used in the DPI model's training set.
    *   [**Download:**](https://github.com/cong-003/DPI-agent/tree/main/data/training_lists) -> Place in `data/training_lists/`

5.  **DPI Transformer Model:**
    *   **Description:** The pre-trained model for DPI prediction.
    *   [**Download:**](https://github.com/cong-003/DPI-agent/tree/main/data/transformer_models) -> Place in `data/transformer_models/`

## Configuration

### Essential Path Configuration

The script `test4_eval.py` contains several hardcoded paths for the resources mentioned above. **You MUST update these paths if you do not use the exact directory structure and filenames as suggested, or if your base directory for `data` differs.**

It is highly recommended to modify the script to use **environment variables** for these paths or pass them as **command-line arguments**.

**Key path constants in `test4_eval.py` to check/modify:**
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

[example evaluation file](https://github.com/cong-003/DPI-agent/blob/main/data/evaluation_files/test_min10_rs_3_1000_pairs.pkl)

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
    --evaluation-file /path/to/your/eval_data.pkl \
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


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation
If you use this software in your research, please cite:
> [2025. DPI-Agent: A Multi-modal Agent for DNA-Protein Interaction Prediction. Available at: [https://github.com/cong-003/DPI-agent]]
> [Any relevant paper or preprint]
