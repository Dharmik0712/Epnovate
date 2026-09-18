import os
from fastmcp import FastMCP
import tools
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from json2pdf_converter import generate
import json
import requests
load_dotenv()
import pypdf2
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import numpy as np


user_input = input("Enter your query: ")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

mcp = FastMCP("MyServer")

mcp = FastMCP(
    "news_search",
    instructions="Provides tools for analyzing numerical datasets. Start with get_summary() for an overview.",
)

@mcp.tool(
    name="scrape_news_articles",           # Custom tool name for the LLM
    description="Scrape news articles from a given category.", # Custom description
    tags={"news", "scraping"},      # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "news-team"}  # Custom metadata
)
def scrape_news_articles(category)->json:
    """Scrape news articles from a given category using the NewsAPI."""
    url = ('https://newsapi.org/v2/top-headlines?'
           'country=us&'
           'category=' + category + '&'
           'apiKey=' + os.environ.get("NEWS_API_KEY"))

    response = requests.get(url)
    return response.json()

@mcp.tool
def pdf_parsing(file, file_name):
    """Parse a PDF file."""
    text = pypdf2.PdfReader(file)
    chunks = TfidfVectorizer().fit_transform([page.extract_text() for page in text.pages])
    return chunks

@mcp.tool
def retrieve_pdf():



@mcp.prompt
def analyze_data(user_input: str) -> str:
    output = f"Please analyze the following query: {user_input}"
    final_output.append(output)

if __name__ == "__main__":
    mcp.run()

# the flow - 

# MCP has all the tools listed and the implementation of each tool will be written just like i have the func for scrapping the news api. 

# there will be 5 tools as you have mentioned and the llm will have the access to them throgh its description and name tags etc, when the query will comw we will take each word out of it or apply a llm or nlp layer to get the user intent and based on that we will make a list of tools on runtime that which tools this query can access and we will hit those tools after the result of those tools. we will provide it to the llm through the prompt along with the user query and the tools desc. to check the results are relevant. 

# after the llm synthesis, we will have the instruction to provide the json response and using the json-to-pdf library we can convert the json to pdf and provide it to the user.

# here the challenges i faced is working with fastmcp, and manual coding where in i dont remember the exact and proper code but i know what things to use and how to use them, 

