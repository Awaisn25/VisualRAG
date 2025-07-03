from fastapi import FastAPI
from pydantic import BaseModel
import os
import torch
import gc
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
import natsort

import warnings
warnings.filterwarnings(action="ignore")

##############WHEN USING VSCODE DEBUGGER##############

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

# Set up logging configuration
logging.basicConfig(filename="output.log", level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialize models
RAG = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v0.1", verbose=0, device="cuda")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map='cuda'
).cuda().eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True, max_pixels=256*28*28)



input_pdfs_folder = "./pdf_data"  # Input folder for PDFs
index_name = "image_index4"  # Name of index

# FastAPI app
app = FastAPI()

# Pydantic model for query input
class QueryModel(BaseModel):
    usr_query: str

# Convert all PDFs in the folder to images
def convert_all_pdfs_to_images(folder_path):
    pdf_images_dict = {}
    doc_id = 0
    file_list = os.listdir(folder_path)
    file_list = natsort.os_sorted(file_list)

    for filename in file_list:
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            logger.info(f"Converting {filename} to images...")
            pdf_images = convert_from_path(pdf_path)
            pdf_images_dict[doc_id] = pdf_images
            logger.info(f"Added {len(pdf_images)} images for document {doc_id} ({filename})")
            doc_id += 1

    return pdf_images_dict

# Create or load the index
def create_load_index(RAG, index_name, folder_path):
    index_dir = f"{os.path.join(os.getcwd())}/.byaldi/{index_name}"

    if os.path.exists(index_dir):
        logger.info(f"Index '{index_name}' already exists at {index_dir}. No need to recreate it.")
        RAG = RAGMultiModalModel.from_index(index_dir, verbose=0)
    else:
        logger.info(f"Index '{index_name}' does not exist. Creating the index at {index_dir}.")
        RAG.index(
            input_path=folder_path,
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=True
        )
        logger.info("Index created successfully.")

    return RAG

# Retrieve similar images based on query
def retriever(RAG, query):
    results = RAG.search(query, k=1)
    return results

# Generate final output based on query and retrieved results
def generate_final_output(query, results, images_dict):
    final_images = []
    first_doc_id=results[0]["doc_id"]
    final_images=images_dict[first_doc_id]
    
    
    # for result in results:
    #     image_index = result["page_num"] - 1
    #     doc_id = result["doc_id"]
    #     image = images_dict[doc_id][image_index]
    #     final_images.append(image)



    content = []
    for image in final_images:
        formDict_image = {"type": "image", "image": image}
        content.append(formDict_image)

    formDict_query = {"type": "text", "text": query}
    content.append(formDict_query)
    
    messages = [{"role": "user", "content": content}]
    
    torch.cuda.empty_cache()
    gc.collect()

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text

# Prepare the PDF images and load the index before API starts
images_dict = convert_all_pdfs_to_images(input_pdfs_folder)
index = create_load_index(RAG, index_name, input_pdfs_folder)


# API route for querying
@app.post("/response")
async def get_response(query_model: QueryModel):
    query = query_model.usr_query
    
    results = retriever(index, query)
    
    final_output = generate_final_output(query, results, images_dict)
    return {"response": final_output}


@app.get('/')
def home():
    return {"message" : "Tabular data Assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)