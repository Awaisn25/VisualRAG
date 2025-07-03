## Overview
This project provides a FastAPI backend for querying tabular data in PDFs using advanced vision-language models. It converts PDFs to images, indexes them, and allows users to query the content via an API. The system is designed as an alternative to traditional OCR, leveraging multimodal retrieval and generation.

## Features
- **PDF to Image Conversion:** Converts all PDFs in a folder to images for processing.
- **Multimodal Indexing:** Uses the Byaldi engine to index images for fast retrieval.
- **Vision-Language Models:** Utilizes ColQwen2.0 (via Byaldi) and Qwen2-VL-2B-Instruct for understanding and generating responses from tabular data.
- **API Endpoint:** Accepts user queries and returns generated responses based on the most relevant document images.
- **Adaptable:** While built for PDFs, the pipeline can be modified to work with standalone images.

## Models & Engine
- **ColQwen2.0:** A powerful vision-language model for multimodal retrieval, used here via the Byaldi engine for indexing and searching document images.
- **Byaldi Engine:** Handles the creation and management of the multimodal index, enabling efficient search and retrieval.
- **Qwen2-VL-2B-Instruct:** Used for conditional generation, producing natural language answers from retrieved images and user queries.

## Usage
- Place your PDFs in the pdf_data folder.
- Run the API:
```sh
python main.py
```
- Send POST requests to /response with your query (see below).

## Example Query
```sh
POST /response
{
  "usr_query": "What is the total revenue in the table?"
}
  ```

## Notes
- This setup is intended for PDF files, but can be modified to work with images directly.
- Created to test an alternative to OCR for extracting information from tabular data in documents.
