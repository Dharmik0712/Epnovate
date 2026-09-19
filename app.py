import os
from fastmcp import FastMCP, Client
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape
import json
import asyncio
import requests
from pydantic import BaseModel
from typing import List
load_dotenv()
import PyPDF2 as pypdf2
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
import numpy as np


user_query = input("Enter your query: ")

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

LLM_MODEL_NAME = "openai/gpt-oss-120b"

final_results = []
pdf_cache = {}

mcp_server = FastMCP("MyServer")

mcp_server = FastMCP(
    "news_search",
    instructions="Provides tools for analyzing numerical datasets. Start with get_summary() for an overview.",
)


# structured output schema (used by synthesize_insights)

class NewsItem(BaseModel):
    headline: str
    source: str
    takeaway: str

class MarketMetrics(BaseModel):
    enterprise_adoption_pct: float
    sentiment_score: float
    cost_index: float
    commentary: str

class RiskItem(BaseModel):
    risk: str
    severity: str
    mitigation: str

class Report(BaseModel):
    title: str
    executive_summary: str
    key_news: List[NewsItem]
    market_metrics: MarketMetrics
    compliance_risks: List[RiskItem]
    recommendations: List[str]


# tool 1 : web scraper 

@mcp_server.tool(
    name="scrape_news_articles",           # Custom tool name for the LLM
    description="Scrape news articles from a given category.", # Custom description
    tags={"news", "scraping"},      # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "news-team"}  # Custom metadata
)
def scrape_news_articles(category: str)->dict:
    """Scrape news articles from a given category using the NewsAPI."""
    news_api_url = ('https://newsapi.org/v2/top-headlines?'
                    'country=us&'
                    'category=' + category + '&'
                    'apiKey=' + os.environ.get("NEWS_API_KEY"))

    api_response = requests.get(news_api_url)
    return api_response.json()


#  tool 2 : internal data hub (MCP) 

TOPIC_BENCHMARKS = {
    "technology":    {"enterprise_adoption_pct": 72.5, "sentiment_score": 0.68, "cost_index": 118.0},
    "business":      {"enterprise_adoption_pct": 58.0, "sentiment_score": 0.41, "cost_index": 104.0},
    "health":        {"enterprise_adoption_pct": 46.3, "sentiment_score": 0.55, "cost_index": 131.0},
    "science":       {"enterprise_adoption_pct": 39.8, "sentiment_score": 0.62, "cost_index": 97.0},
    "entertainment": {"enterprise_adoption_pct": 51.2, "sentiment_score": 0.47, "cost_index": 90.0},
    "sports":        {"enterprise_adoption_pct": 33.4, "sentiment_score": 0.59, "cost_index": 85.0},
    "general":       {"enterprise_adoption_pct": 44.0, "sentiment_score": 0.50, "cost_index": 100.0},
}

@mcp_server.tool(
    name="get_market_benchmarks",
    description="Pull internal enterprise adoption metrics, sentiment and cost index for a topic.",
    tags={"internal", "metrics"},
)
def get_market_benchmarks(topic: str) -> dict:
    """Return internal company benchmarks for a topic."""
    topic_data = TOPIC_BENCHMARKS.get(topic.lower(), TOPIC_BENCHMARKS["general"])
    return {"topic": topic, **topic_data}


# tool 3 : compliance (MCP) 

COMPLIANCE_RULES = {
    "technology": [
        {"risk": "Data privacy (GDPR / DPDP Act)", "severity": "high", "mitigation": "Data minimisation, consent management and DPIAs."},
        {"risk": "AI transparency obligations (EU AI Act)", "severity": "medium", "mitigation": "Model documentation and human oversight."},
    ],
    "business": [
        {"risk": "Financial disclosure rules", "severity": "medium", "mitigation": "Legal review before publishing figures."},
        {"risk": "Anti-trust scrutiny", "severity": "low", "mitigation": "Track M&A and partnership filings."},
    ],
    "health": [
        {"risk": "Patient data protection (HIPAA)", "severity": "high", "mitigation": "Encrypt PHI and restrict access by role."},
        {"risk": "Medical device / claims regulation", "severity": "medium", "mitigation": "Avoid unapproved clinical claims."},
    ],
    "general": [
        {"risk": "General data protection requirements", "severity": "medium", "mitigation": "Follow internal data handling policy."},
    ],
}

@mcp_server.tool(
    name="check_compliance_risks",
    description="Validate regulatory and compliance risks associated with a topic.",
    tags={"internal", "compliance"},
)
def check_compliance_risks(topic: str) -> dict:
    """Return regulatory and compliance risks for a topic."""
    topic_risks = COMPLIANCE_RULES.get(topic.lower(), COMPLIANCE_RULES["general"])
    severity_rank = {"low": 1, "medium": 2, "high": 3}
    highest_severity = max(topic_risks, key=lambda item: severity_rank[item["severity"]])["severity"]
    return {"topic": topic, "overall_risk": highest_severity, "risks": topic_risks}


# tool 4 : LLM processing (Groq) 

@mcp_server.tool(
    name="synthesize_insights",
    description="Use structured LLM output to merge news data and MCP metrics into one report.",
    tags={"llm", "synthesis"},
)
def synthesize_insights(query: str, news: dict, benchmarks: dict, compliance: dict) -> dict:
    """Merge news + benchmarks + compliance into a validated JSON report using Groq."""
    top_articles = [
        {
            "title": article.get("title"),
            "source": (article.get("source") or {}).get("name"),
            "description": article.get("description"),
        }
        for article in (news or {}).get("articles", [])[:8]
    ]

    analyst_prompt = (
        "You are a business analyst. Merge the news articles, internal benchmarks and compliance risks "
        "into ONE report. Respond ONLY with a JSON object that follows this JSON schema exactly:\n"
        + json.dumps(Report.model_json_schema())
    )
    request_payload = json.dumps({
        "query": query,
        "news_articles": top_articles,
        "market_benchmarks": benchmarks,
        "compliance": compliance,
    })

    latest_error = None
    for attempt in range(2):
        llm_response = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": analyst_prompt},
                {"role": "user", "content": request_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        try:
            parsed_report = Report.model_validate_json(llm_response.choices[0].message.content)
            return parsed_report.model_dump()
        except Exception as err:
            latest_error = err
    raise ValueError(f"LLM did not return valid structured output: {latest_error}")


# tool 5 : PDF generator

@mcp_server.tool(
    name="render_pdf_report",
    description="Compile the structured summary into a formatted PDF document on disk.",
    tags={"pdf", "report"},
)
def render_pdf_report(summary: dict, output_path: str = "report.pdf") -> str:
    """Render the structured summary into a PDF file and return its path."""
    pdf_styles = getSampleStyleSheet()
    pdf_elements = []

    def add_heading(text):
        pdf_elements.append(Paragraph(escape(str(text)), pdf_styles["Heading2"]))

    def add_paragraph(text):
        pdf_elements.append(Paragraph(escape(str(text)), pdf_styles["BodyText"]))
        pdf_elements.append(Spacer(1, 6))

    pdf_elements.append(Paragraph(escape(summary["title"]), pdf_styles["Title"]))
    add_heading("Executive Summary")
    add_paragraph(summary["executive_summary"])

    add_heading("Key News")
    for news_item in summary["key_news"]:
        add_paragraph(f"{news_item['headline']} ({news_item['source']}) - {news_item['takeaway']}")

    metrics = summary["market_metrics"]
    add_heading("Market Metrics")
    add_paragraph(f"Enterprise adoption: {metrics['enterprise_adoption_pct']}%")
    add_paragraph(f"Sentiment score: {metrics['sentiment_score']}")
    add_paragraph(f"Cost index: {metrics['cost_index']}")
    add_paragraph(metrics["commentary"])

    add_heading("Compliance Risks")
    for risk_item in summary["compliance_risks"]:
        add_paragraph(f"[{risk_item['severity'].upper()}] {risk_item['risk']} - {risk_item['mitigation']}")

    add_heading("Recommendations")
    for recommendation in summary["recommendations"]:
        add_paragraph(f"- {recommendation}")

    SimpleDocTemplate(output_path).build(pdf_elements)
    return os.path.abspath(output_path)



# orchestrator

def detect_intent(query):
    """LLM layer: get user intent and decide which tools this query needs at runtime."""
    intent_response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "Return ONLY JSON like "
                '{"category": "...", "tools": ["..."]}. '
                "category must be one of: business, entertainment, general, health, science, sports, technology. "
                "tools must be a subset of: scrape_news_articles, get_market_benchmarks, check_compliance_risks."
            )},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    user_intent = json.loads(intent_response.choices[0].message.content)
    user_intent.setdefault("category", "general")
    user_intent["tools"] = user_intent.get("tools") or [
        "scrape_news_articles", "get_market_benchmarks", "check_compliance_risks"
    ]
    return user_intent

def unwrap(tool_result):
    """Get the plain python value out of an MCP tool result."""
    result_data = getattr(tool_result, "data", None)
    if result_data is not None:
        return result_data
    return json.loads(tool_result.content[0].text)

async def main():
    user_intent = detect_intent(user_query)
    selected_category = user_intent["category"]

    async with Client(mcp_server) as mcp_client:
        news_data, benchmark_data, compliance_data = {}, {}, {}

        if "scrape_news_articles" in user_intent["tools"]:
            news_data = unwrap(await mcp_client.call_tool("scrape_news_articles", {"category": selected_category}))
        if "get_market_benchmarks" in user_intent["tools"]:
            benchmark_data = unwrap(await mcp_client.call_tool("get_market_benchmarks", {"topic": selected_category}))
        if "check_compliance_risks" in user_intent["tools"]:
            compliance_data = unwrap(await mcp_client.call_tool("check_compliance_risks", {"topic": selected_category}))

        report_summary = unwrap(await mcp_client.call_tool("synthesize_insights", {
            "query": user_query,
            "news": news_data,
            "benchmarks": benchmark_data,
            "compliance": compliance_data,
        }))

        with open("final_output.json", "w") as json_file:
            json.dump(report_summary, json_file, indent=2)

        saved_pdf_path = unwrap(await mcp_client.call_tool("render_pdf_report", {
            "summary": report_summary,
            "output_path": "report.pdf",
        }))
        print("PDF saved at:", saved_pdf_path)

if __name__ == "__main__":
    asyncio.run(main())

