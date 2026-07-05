# 🔨 Forge: Enterprise Legal SLM Extraction Engine

Forge is a high-throughput, local AI inference pipeline designed to ingest complex, multi-page legal documents (like PDFs) and deterministically extract structured JSON data using a fine-tuned Small Language Model (SLM).

By utilizing local inference and context-window chunking, Forge bypasses the high costs, latency, and privacy concerns of closed-source APIs like GPT-4, running entirely on consumer hardware or free cloud tiers.

## 🧠 System Architecture & Repository Structure

This repository contains the entire end-to-end lifecycle of the AI application, from synthetic data generation to full-stack deployment.

* **`/scripts` (Data Engineering):** Contains `synthetic-data-generator.py`, which leverages an LLM to generate hundreds of highly realistic, OCR-distorted legal contracts to train our model on messy, real-world data.

* **`/notebooks` (Fine-Tuning):** Contains the PyTorch/Unsloth training notebook. The base Llama-3 model was fine-tuned using Low-Rank Adaptation (LoRA) to specifically master deterministic JSON extraction.

* **`/cloud-backend` (Inference Engine):** A FastAPI server that receives PDF byte streams, uses `PyMuPDF` for lightning-fast text extraction, chunks the text to fit the SLM's context window, and streams it through the quantized `.gguf` model using `llama.cpp`.

* **`/frontend` (Client Dashboard):** A vanilla JavaScript and Tailwind CSS telemetry dashboard for drag-and-drop file processing and latency monitoring.

## 🛠️ The Tech Stack

* **Data Generation:** Python, Groq API, Pydantic
* **Training:** PyTorch, Unsloth, Hugging Face `trl`
* **Quantization:** `Q4_K_M` (4-bit) format, shrinking the model from ~16GB to ~2GB.
* **Backend API:** Python, FastAPI, PyMuPDF, `llama-cpp-python`
* **Frontend:** HTML, JavaScript, Tailwind CSS

## 🚀 How to Run Locally

### 1. Prerequisites

* Python 3.10+
* Note: The fine-tuned `legal_slm-unsloth.Q4_K_M.gguf` model file must be placed in the `cloud-backend` folder to run the server. *This file is excluded from Git due to size limits.*

### 2. Booting the Cloud Backend

```bash
cd cloud-backend
pip install -r requirements.txt
python main.py