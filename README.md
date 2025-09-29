# Qimai: DNA-Protein Interaction Prediction Agent

## Overview

Qimai is a comprehensive pipeline designed to predict interactions between DNA and protein sequences. It synthesizes information from multiple sources to make an informed prediction, providing not just a result but also a detailed explanation of the evidence found.

It leverages a multi-modal approach, integrating evidence from:
*   **Direct Motif Analysis:** Searching for known binding motifs of the target protein in the DNA sequence using the CisBP database and FIMO scans.
*   **Indirect Motif Analysis:** Identifying transcription factor (TF) interactors of the query protein (via STRING DB) to find evidence of indirect binding.
*   **Prediction from specialized models:** Utilizing a state-of-the-art deep learning model trained on ChIP-seq data to predict interaction probability.
*   **Large Language Model (LLM) Synthesis:** Employing an LLM (e.g., Gemini, or open-source Hugging Face models) to synthesize all gathered evidence and provide a final interaction prediction (0 or 1) along with a detailed explanation.
*   **Confidence Level Estimation:** Using a rule-based method to estimate how confident the prediction is.

## Features

*   Modular workflow managed by LangGraph.
*   Support for multiple LLM backends: Google Gemini, and open source Hugging Face models.
*   Comprehensive motif analysis using CisBP and FIMO.
*   Protein-protein interaction data integration from STRING DB.
*   Rule-based confidence scoring for LLM predictions.
*   Configurable prompt styles for LLM interaction.
*   Detailed JSON outputs for both summary and comprehensive results.

## Quick Start

This section provides the fastest way to get the agent running with the included example data.

**1. Clone the Repository**
```bash
git clone https://github.com/cong-003/DPI-agent.git
cd DPI-agent
```

**2. Set up Environment and Install Dependencies**
This assumes you have Python 3.10.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Prepare Data and Models**

The agent requires several large data files. For this quick start, you only need to unzip the included motif database.
More details are in the [Data and Models](#data-and-models) section.

```bash
unzip data/Homo_sapiens_2025_05_16_4_38_am.zip -d data/
```

**4. Set Environment Variables**

The script uses environment variables to find necessary files. For the quick start, you can set them directly in your terminal.

```bash
export DPI_FASTA_PATH="data/uniprot_ChIP_690_Encode3and4_2024_03_01.fasta"
export DPI_TF_FAMILY_PATH="data/all_pros_family.csv"
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"  # Required if using Gemini
```
*Note: If you are not using a Gemini model, you can leave `GOOGLE_API_KEY` empty.*

**5. Run the Agent**

Now, run the main script on the example evaluation file. This example uses a local CPU-based model to avoid GPU requirements.

```bash
python test8_eval_reduce_FP.py \
    --input-file data/evaluation_files/test_min10_rs_3_1000_pairs.pkl \
    --llm-model hf/unsloth/mistral-7b-bnb-4bit \
    --limit 10 \
    --force-cpu
```

This command will process the first 10 samples from the test file. Output JSON files will be saved in the `./dpi_agent_outputs/` directory.

---

## Detailed Setup

### System Requirements

*   **Operating System:** Linux (recommended due to FIMO dependency). macOS might work with FIMO installed. Windows may require WSL.
*   **Python:** 3.8 - 3.10 (tested with 3.10)
*   **CPU/GPU:**
    *   CPU is sufficient for most tasks.
    *   A CUDA-enabled GPU is highly recommended for faster LLM inference and Transformer model predictions.
*   **RAM:** Minimum 16GB, 32GB+ recommended for larger LLMs.
*   **Disk Space:** Several GBs for the environment and scripts, plus significant additional space (up to 100GB) for the required databases and models.

### Installation

1.  **MEME Suite (for FIMO):**
    FIMO is required for motif scanning. Install the MEME Suite from [http://meme-suite.org/doc/install.html](http://meme-suite.org/doc/install.html). Ensure `fimo` is in your system's `PATH`.

2.  **Clone and Set up Environment:**
    ```bash
    git clone https://github.com/cong-003/DPI-agent.git
    cd DPI-agent
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### Data and Models

The agent requires several large data and model files. Some are included, but others must be downloaded.

**Included in the Repository:**

*   `data/evaluation_files/test_min10_rs_3_1000_pairs.pkl`: Example evaluation file.
*   `data/uniprot_ChIP_690_Encode3and4_2024_03_01.fasta`: FASTA file for proteins.
*   `data/all_pros_family.csv`: Protein family information.
*   `data/Homo_sapiens_2025_05_16_4_38_am.zip`: CisBP Motif database. **You must unzip this file.**
    ```bash
    unzip data/Homo_sapiens_2025_05_16_4_38_am.zip -d data/
    ```

**To Be Downloaded:**

Due to their size, the following resources must be downloaded separately.

1.  **DNA & Protein Embeddings:**
    *   **Description:** Pre-computed embeddings from DNABERT and AlphaFold.
    *   [**Download Link**](https://drive.google.com/drive/folders/1x4TKNuO42AYeopGIVH5i8ICsmgY9J_8z)
    *   **Action:** Download the `embeddings` folder and place it inside the `data/` directory. The final path should be `data/embeddings/`.

2.  **SEI Model:**
    *   **Description:** The pre-trained SEI model for sequence-based prediction.
    *   [**Download Link**](https://drive.google.com/file/d/1NETAfDVTBvQbp8XivUPlY0Is_tprxeCk/view?usp=drive_link)
    *   **Action:** Download the `.pt` file and place it in `data/prediction_models/sei/`.

### Configuration

The script `test8_eval_reduce_FP.py` uses environment variables to locate data files.

**Set Environment Variables:**

Export these variables in your shell session or add them to your `~/.bashrc` or `~/.zshrc` file.

```bash
# Path to the main protein FASTA file
export DPI_FASTA_PATH="/path/to/your/DPI-agent/data/uniprot_ChIP_690_Encode3and4_2024_03_01.fasta"

# Path to the protein family CSV file
export DPI_TF_FAMILY_PATH="/path/to/your/DPI-agent/data/all_pros_family.csv"

# Google API Key for Gemini models
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```
*Replace `/path/to/your/DPI-agent/` with the actual absolute path to the cloned repository.*

### API Keys

*   **Google Gemini:** If using a `gemini/` model, set the `GOOGLE_API_KEY` environment variable.
*   **Hugging Face:** If using gated models, provide a token via the `--hf-token` argument or by logging in with `huggingface-cli login`.

## Usage

The script is designed for batch evaluation using an input file.

### Input Data Format
The `--input-file` should be a `.tsv` or `.pkl` file with these columns:
*   `dna_sequence`: The DNA sequence.
*   `protein_name`: The protein name (e.g., "SOX2").
*   `label`: The ground truth label (0 or 1).

### Command-Line Arguments

*   `--input-file FILE`: (Required) Path to the evaluation dataset.
*   `--llm-model MODEL_ID`: LLM to use (e.g., `gemini/gemini-1.5-flash-latest`, `hf/unsloth/mistral-7b-bnb-4bit`).
*   `--output-dir DIR`: Directory to save output JSON files (Default: `./dpi_agent_outputs`).
*   `--limit INT`: Process only the first N samples.
*   `--force-cpu`: Force CPU usage.
*   `--prompt-style STYLE`: `verbose`, `concise`, or `transformer-priority` (Default: `verbose`).
*   `--api-delay FLOAT`: Delay between Gemini API calls (Default: 1.5).
*   `--hf-token TOKEN`: Hugging Face API token.

### Example Invocation

This example runs the agent on 100 samples using a Hugging Face model, a concise prompt style, and saves results to a custom directory.

```bash
python test8_eval_reduce_FP.py \
    --input-file data/evaluation_files/test_min10_rs_3_1000_pairs.pkl \
    --llm-model hf/unsloth/mistral-7b-bnb-4bit \
    --output-dir ./my_dpi_results \
    --prompt-style concise \
    --limit 100
```

## Output Files

For each input sample, the agent generates two JSON files:

1.  **`..._simple.json`:** A summary of the prediction, including the predicted label, confidence score, and the LLM's explanation.
2.  **`..._comp.json`:** The complete agent state with all intermediate results, useful for debugging.

An aggregated summary file, **`..._aggregated_simple_results.json`**, is also created in the output directory, containing all the simple results for easy analysis.

## Troubleshooting

*   **`fimo: command not found`**: Ensure MEME Suite is installed and `fimo` is in your system `PATH`.
*   **`ModuleNotFoundError`**: Activate the correct virtual environment (`source .venv/bin/activate`) and run `pip install -r requirements.txt`.
*   **File Not Found Errors**: Double-check that you have downloaded all necessary data, unzipped the motif database, and correctly set the environment variables (`DPI_FASTA_PATH`, `DPI_TF_FAMILY_PATH`).
*   **CUDA/GPU Issues**: Ensure PyTorch is installed with CUDA support and that your GPU drivers are up-to-date.
*   **LLM API Errors**: 
    *   **Gemini:** Check your `GOOGLE_API_KEY` and network connection.
    *   **Hugging Face:** For gated models, ensure you have provided a token. Large models may require significant RAM/VRAM.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation
If you use this software in your research, please cite:
> [2025. DPI-Agent: A Multi-modal Agent for DNA-Protein Interaction Prediction. Available at: https://github.com/cong-003/DPI-agent]
