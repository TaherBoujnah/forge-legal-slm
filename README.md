# 🔨 Forge: Enterprise Legal SLM Extraction Engine

Forge is a high-throughput, privacy-first inference pipeline designed to ingest complex, multi-page legal documents (such as NDAs, MSAs, and Lease Agreements) and deterministically extract structured JSON data.

Powered by a fine-tuned Small Language Model (SLM), Forge relies on local inference and intelligent context-window chunking to bypass the high costs, latency bottlenecks, and data privacy concerns associated with closed-source APIs like GPT-4. It is built to run efficiently on consumer hardware or scale infinitely via serverless cloud infrastructure.

## 🧠 System Architecture & Repository Structure

* **`/scripts` (Data Engineering):**  Contains the synthetic data generation pipeline (synthetic-data-generator.py). This leverages an LLM to generate hundreds of highly realistic, OCR-distorted legal contracts, ensuring the model is trained on messy, real-world data distributions.

* **`/notebooks` (Fine-Tuning):** The core API (FastAPI/AWS Lambda compatible) that receives PDF byte streams. It uses PyMuPDF for lightning-fast text extraction, chunks the text to respect the SLM's context window, and streams it through the quantized .gguf model using llama.cpp.

* **`/cloud-backend` (Inference Engine):** A FastAPI server that receives PDF byte streams, uses `PyMuPDF` for lightning-fast text extraction, chunks the text to fit the SLM's context window, and streams it through the quantized `.gguf` model using `llama.cpp`.

* **`/frontend` (Client Dashboard):** A vanilla JavaScript and Tailwind CSS telemetry dashboard featuring drag-and-drop file processing, latency monitoring, and real-time extraction results.

## 🛠️ The Tech Stack

* **Data Generation:** Python, Groq API, Pydantic
* **Training:** PyTorch, Unsloth, Hugging Face `trl`
* **Quantization:** `Q4_K_M` (4-bit) format, shrinking the model from ~16GB to ~2GB.
* **Backend API:** Python, FastAPI, PyMuPDF, `llama-cpp-python`
* **Frontend:** HTML, JavaScript, Tailwind CSS
* **Cloud Infrastructure:** AWS Lambda, ECR, S3, Terraform
