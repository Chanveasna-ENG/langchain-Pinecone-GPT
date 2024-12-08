import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from openai import OpenAI

import pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# uvicorn app:app --host 0.0.0.0 --port 10000
app = FastAPI()

# Setup environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
environment = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX")

# Debugging: Print the OpenAI API key
print(f"OpenAI API Key: {openai_api_key}")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Initialize pinecone client
pc = Pinecone(api_key=pinecone_api_key)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=environment
        )
    )
index = pc.Index(index_name)

# Middleware to secure HTTP endpoint
security = HTTPBearer()

def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
):
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            logger.error("Invalid token")
            raise HTTPException(status_code=403, detail="Invalid token")
    else:
        logger.error("Invalid authentication scheme")
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")

class QueryModel(BaseModel):
    query: str

@app.post("/")
async def get_context(
    query_data: QueryModel,
    credentials: HTTPAuthorizationCredentials = Depends(validate_token),
):
    try:
        # convert query to embeddings
        res = openai_client.embeddings.create(
            input=[query_data.query], model="text-embedding-3-large"
        )
        embedding = res.data[0].embedding
        # Search for matching Vectors
        results = index.query(vector=embedding, top_k=6, include_metadata=True).to_dict()
        # Filter out metadata from search result
        context = [match["metadata"]["text"] for match in results["matches"]]
        # Return context
        return context
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")