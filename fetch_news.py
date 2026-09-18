# import json
# from dotenv import load_dotenv
# import mcp
# import requests
# import os
# load_dotenv()

# @mcp.tool(
#     name="scrape_news_articles",           # Custom tool name for the LLM
#     description="Scrape news articles from a given category.", # Custom description
#     tags={"news", "scraping"},      # Optional tags for organization/filtering
#     meta={"version": "1.0", "author": "news-team"}  # Custom metadata
# )
# def scrape_news_articles(category)->json:
#     """Scrape news articles from a given category using the NewsAPI."""
#     url = ('https://newsapi.org/v2/top-headlines?'
#            'country=us&'
#            'category=' + category + '&'
#            'apiKey=' + os.environ.get("NEWS_API_KEY"))

#     response = requests.get(url)
#     return response.json()