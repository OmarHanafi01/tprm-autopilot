import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader, DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from google.oauth2 import service_account


class embedExistingDocuments:
    def __init__(self, model_name: str = "textembedding-gecko@003"):
        self.cts_vulns = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )  # noqa
        self.vertex_embeddings = VertexAIEmbeddings(
            model_name=model_name, credentials=credentials
        )

    def get_assessment_embeddings(self, assessment_path=str):
        assessment_file_extension = os.path.splitext(
            os.path.basename(assessment_path)
        )[1]
        if assessment_file_extension == ".csv":
            loader = CSVLoader(file_path=assessment_path)
        elif assessment_file_extension == ".xlsx":
            assessment_df = pd.read_excel(assessment_path)
            loader = DataFrameLoader(data_frame=assessment_df,
                                     page_content_column="Question")
        loaded_assessment_docs = loader.load()
        split_assessment_docs = self.cts_vulns.split_documents(loaded_assessment_docs) # noqa
        try:
            faiss_assessment_documents = FAISS.from_documents(
                documents=split_assessment_docs, embedding=self.vertex_embeddings # noqa
            )
        except Exception as e:
            raise e
        self.faiss_assessment_documents = faiss_assessment_documents

    def get_similar_context_from_assessments(
        self, prompt: str, k: int = 2, threshold: int = 0.5
    ) -> pd.DataFrame:
        try:
            results = self.faiss_assessment_documents.similarity_search_with_relevance_scores(  # noqa
                query=prompt, k=k, score_threshold=threshold
            )
        except Exception:
            return
        new_results = []
        for item in results:
            new_results.append(item[0].page_content)
        return new_results
