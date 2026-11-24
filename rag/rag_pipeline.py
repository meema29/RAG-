#!/usr/bin/env python
# coding: utf-8

# # Pharmaly Chatbot — RAG Pipeline
# 
# This notebook connects the product retrieval system to Gemini.
# 
# It handles:
# - Retrieving relevant product information.
# - Checking whether sufficient evidence is available.
# - Generating answers using Gemini.
# - Returning sources with the answers.

# **Import the required libraries and functions**

# In[39]:


import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from retrieval import search_products


# **os**	Access environment variables, including the API key
# 
# **Path**	Locate the .env file
# 
# **load_dotenv**	Load the API key from .env
# 
# **genai**	Communicate with Gemini
# 
# **search_products**	Retrieve relevant products from ChromaDB

# **Configure Gemini**

# In[40]:


# Locate and load the API key
env_path = Path.cwd().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Select the model
MODEL_NAME = "gemini-3.5-flash"

print("Gemini configured successfully!")


# **Create the RAG function**

# In[41]:


SYSTEM_PROMPT = """
You are Pharmaly, a chatbot specialized in providing information
about Pharmaly products.

Your purpose is to help users understand and explore the products
available in the Pharmaly product database.

You must answer questions using only the information provided
in the retrieved product context.
"""

SYSTEM_PROMPT += """
RULES:

1. Use only the information provided in the retrieved context.
2. Never invent product names, ingredients, prices, brands,
   availability, or product characteristics.
3. Do not use your general knowledge to fill in missing information.
4. If the retrieved context does not contain the answer,
   clearly tell the user that the information is not available
   in the Pharmaly database.
5. If the information is incomplete, explain what is missing.
6. If multiple products are relevant, distinguish between them
   clearly.
"""

SYSTEM_PROMPT += """
SOURCE HANDLING:

1. Support your answers with the retrieved product information.
2. When available, mention the product name and its source URL.
3. Never invent or modify a source URL.
4. If no source URL is provided, do not create one.
5. Make it clear which product each piece of information refers to.
6. If retrieved information conflicts, mention the inconsistency
   instead of choosing one version.
"""


# **Test product retrieval**

# In[42]:


test_result = search_products("vitamin C")

print("Retrieval status:", test_result["status"])
print("Number of documents:", len(test_result["documents"]))

if test_result["documents"]:
    print("\nFirst retrieved document:")
    print(test_result["documents"][0])


# What search_products() returned ? 
# 
# 
# **id**	Unique document identifier
# 
# **content**	Product information that Gemini can read
# 
# **metadata**	Structured information such as brand, price, category and URL
# 
# **similarity**	How closely the document matches the question

# **Prepare the retrieved context**

# In[43]:


def format_context(documents):
    context = ""

    for i, doc in enumerate(documents, start=1):
        metadata = doc["metadata"]

        context += f"""
PRODUCT {i}

{doc["content"]}

Product name: {metadata.get("product_name", "Not available")}
Brand: {metadata.get("brand", "Not available")}
Category: {metadata.get("category", "Not available")}
Price: {metadata.get("current_price", "Not available")}
Source URL: {metadata.get("product_url", "Not available")}

--------------------------------
"""

    return context


# Understanding the function
# 
# - def format_context(documents): defines a function that receives retrieved documents.
# - for goes through each document.
# - metadata.get() extracts information safely, even if a field is missing.
# - context += combines all the documents into one text block.
# - return context gives us the final text.

# In[44]:


context = format_context(test_result["documents"])

print(context[:2000])


# **Create the Gemini generation function**

# In[45]:


def generate_answer(question, context):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"""
{SYSTEM_PROMPT}

RETRIEVED PRODUCT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear and helpful answer based only on the retrieved context.
"""
    )

    return response.text


# 
# **question**	The user's original question
# 
# **context**	The five retrieved product documents
# 
# **SYSTEM_PROMPT**	The chatbot's rules
# 
# **generate_content()**	Sends everything to Gemini
# 
# **response.text**	Contains Gemini's generated answer

# **Test Gemini with retrieved information**

# In[46]:


answer = generate_answer(
    question="Which products contain vitamin C?",
    context=context
)

print(answer)


# In[51]:


def rag_answer(question):

    # Step 1: Retrieve relevant products
    result = search_products(question)

    # Step 2: Check whether sufficient evidence exists
    if result["status"] == "insufficient_evidence":
        return {
            "answer": "Sorry, I couldn't find sufficient information about this question in the Pharmaly product database.",
            "sources": []
        }

    # Step 3: Format the retrieved products
    context = format_context(result["documents"])

    # Step 4: Generate the answer using Gemini
    answer = generate_answer(question, context)

    # Step 5: Collect only relevant product sources
    sources = []

    for doc in result["documents"]:
        metadata = doc["metadata"]

        product_name = metadata.get("product_name", "Unknown product")

        if product_name.lower() in answer.lower():
            sources.append({
                "product_name": product_name,
                "url": metadata.get("product_url")
            })

    return {
        "answer": answer,
        "sources": sources
    }


# - search_products(question) searches the database for products relevant to the question.
# - result["status"] tells us whether the retrieval system found sufficient evidence.
# - If evidence is insufficient, we return a refusal immediately.
# - "sources": [] means there are no reliable product sources to show.

# In[52]:


result = rag_answer("Which products contain vitamin C?")

print("ANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print(source["product_name"])
    print(source["url"])
    print()


# In[53]:


result = rag_answer("Who won the FIFA World Cup in 2018?")

print("ANSWER:")
print(result["answer"])

print("\nSOURCES:")
print(result["sources"])


# In[54]:


result = rag_answer("Does Pharmaly sell laptops?")

print("ANSWER:")
print(result["answer"])

print("\nSOURCES:")
print(result["sources"])


# In[55]:


get_ipython().run_line_magic('pip', 'install nbconvert')


# In[59]:


import sys
import subprocess

subprocess.run([
    sys.executable, "-m", "nbconvert",
    "--to", "script",
    "rag_pipeline.ipynb"
], check=True)


# In[60]:


get_ipython().system('jupyter nbconvert --to script rag_pipeline.ipynb')


# In[61]:


import os
print("rag_pipeline.py" in os.listdir())

