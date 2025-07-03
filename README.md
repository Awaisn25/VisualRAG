## VLM PDF-to-Query API
This project provides a FastAPI backend for querying tabular data in PDFs using a multimodal retrieval and generation pipeline. It converts PDFs to images, indexes them, and allows users to query the content via an API. The system leverages advanced vision-language models as an alternative to traditional OCR.

## Features
- Converts all PDFs in a folder to images.
- Indexes images for fast retrieval using a multimodal RAG model.
- Accepts user queries and returns generated responses based on the most relevant document images.
- Designed for tabular data, but can be adapted for direct image input.

## Usage
- Place your PDFs in the pdf_data folder.
- Run the API:
  - Send POST requests to /response with your query.

## Notes
This setup is intended for PDF files, but can be modified to work with standalone images.
The project was created to test an alternative to OCR for extracting information from tabular data in documents.
