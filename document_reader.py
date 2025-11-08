# document_reader.py
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from typing import List

class DocumentAIReader:
    def __init__(self, project_id: str, location: str, processor_id: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id

        # Initialize Document AI client
        self.client = documentai.DocumentProcessorServiceClient()
        self.processor_name = self.client.processor_path(project_id, location, processor_id)

    def process_pdf_from_gcs(self, gcs_uri: str, mime_type: str = "application/pdf", use_raw_document: bool = False):
        """Process a PDF from GCS"""
        if use_raw_document:
            storage_client = storage.Client()
            bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
            blob_path = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            content = blob.download_as_bytes()

            raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
            request = documentai.ProcessRequest(name=self.processor_name, raw_document=raw_document)
        else:
            gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type)
            input_config = documentai.ProcessRequest.InputDocuments(
                gcs_documents=documentai.GcsDocuments(documents=[gcs_document])
            )
            request = documentai.ProcessRequest(name=self.processor_name, input_documents=input_config)

        result = self.client.process_document(request=request)
        return result.document

    def extract_text(self, document: documentai.Document) -> str:
        """Extract all text from the document"""
        return document.text

    def summarize_text(self, text: str, max_words: int = 150) -> str:
        """Simple summary generator (first few sentences)"""
        words = text.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
