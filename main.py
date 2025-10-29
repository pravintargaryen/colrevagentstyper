import typer
from dotenv import load_dotenv
import dspy
import os
import requests
from rich.console import Console
from rich.panel import Panel


load_dotenv()
console = Console()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment!")

dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash"))


# ----------------------------
#  Tool: Crossref API Fetch
# ----------------------------
def crossref_search_json(query: str):
    """Fetch publication details based on the given search query."""
    url = f"https://api.crossref.org/works?query={query}&rows=5"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "Failed to fetch from Crossref."}
    data = response.json()
    items = data.get("message", {}).get("items", [])

    papers = []
    for item in items:
        papers.append({
            "doi": item.get("DOI"),
            "title": item.get("title", ["Untitled"])[0],
            "url": item.get("URL"),
            "publisher": item.get("publisher"),
        })
    return papers


# ----------------------------
# DSPy Agent Signatures
# ----------------------------
class ProblemFormulationAgent(dspy.Signature):
    """Formulate a precise research problem statement."""
    query: str = dspy.InputField()
    refined_query: str = dspy.OutputField(desc="A refined version of the research question.")

class QueryBuilderAgent(dspy.Signature):
    """Convert a refined research question into a structured Boolean search query."""
    refined_query: str = dspy.InputField()
    search_query: str = dspy.OutputField(desc="A Boolean-style search query suitable for Crossref, e.g. 'gene therapy' AND hemophilia.")
    

class DataRetrievalAgent(dspy.Signature):
    """Retrieve relevant papers from Crossref API."""
    refined_query: str = dspy.InputField()
    papers: list = dspy.OutputField(desc="List of retrieved papers.")


class DataPrescreenAgent(dspy.Signature):
    """Filter and select the most relevant papers."""
    papers: list = dspy.InputField()
    filtered_papers: list = dspy.OutputField(desc="Filtered subset of papers relevant to the topic.")


class PDFRetrievalAgent(dspy.Signature):
    """Retrieve or simulate retrieval of PDFs for the selected papers."""
    filtered_papers: list = dspy.InputField()
    pdf_links: list = dspy.OutputField(desc="List of simulated or retrieved PDF URLs.")


# ----------------------------
#  Synthesis Agent (Enhanced)
# ----------------------------
class DataSynthesisAgent(dspy.Signature):
    """Synthesize insights from retrieved papers."""
    pdf_links: list = dspy.InputField()
    filtered_papers: list = dspy.InputField()
    final_summary: str = dspy.OutputField(desc="Concise synthesized research summary.")


# ----------------------------
#  Create DSPy ReAct Agents
# ----------------------------
problem_agent = dspy.ReAct(ProblemFormulationAgent, tools=[])
query_builder_agent = dspy.ReAct(QueryBuilderAgent, tools=[])
retrieval_agent = dspy.ReAct(DataRetrievalAgent, tools=[crossref_search_json])
prescreen_agent = dspy.ReAct(DataPrescreenAgent, tools=[])
pdf_agent = dspy.ReAct(PDFRetrievalAgent, tools=[])
synthesis_agent = dspy.ChainOfThought(DataSynthesisAgent, tools=[])


# ----------------------------
#  Typer App
# ----------------------------
app = typer.Typer()

@app.command()
def run(query: str):
    console.rule("[bold green]Systematic Review Pipeline[/bold green]")
    console.print(Panel.fit(f"🔍 Input Query: {query}", style="bold cyan"))

    # 1️⃣ Problem formulation
    problem = problem_agent(query=query)
    refined_query = problem.get("refined_query", query)
    console.print(Panel.fit(f"🧩 Refined Query: {refined_query}", style="bold green"))

    # 1.5️⃣ Query builder — generate Crossref-compatible Boolean query
    query_built = query_builder_agent(refined_query=refined_query)
    search_query = query_built.get("search_query", refined_query)
    console.print(Panel.fit(f"🧱 Search Query: {search_query}", style="bold cyan"))

    # 2️⃣ Data retrieval
    retrieved = retrieval_agent(refined_query=search_query)
    papers = retrieved.get("papers", [])
    console.print(Panel.fit(f"📄 Retrieved {len(papers)} papers", style="bold yellow"))

    # 3️⃣ Prescreening
    prescreened = prescreen_agent(papers=papers)
    filtered_papers = prescreened.get("filtered_papers", [])
    console.print(Panel.fit(f"🧹 Filtered Papers: {filtered_papers}", style="bold magenta"))

    # 4️⃣ PDF retrieval
    pdf_stage = pdf_agent(filtered_papers=filtered_papers)
    pdf_links = pdf_stage.get("pdf_links", [])
    console.print(Panel.fit(f"📥 PDF Links: {pdf_links}", style="bold cyan"))

    # 5️⃣ Synthesis — now with Gemini reasoning
    synthesis = synthesis_agent(pdf_links=pdf_links, filtered_papers=filtered_papers)
    final_summary = synthesis.get("final_summary", "No papers were provided for analysis.")
    console.print(Panel.fit(f"🧠 Final Synthesized Summary:\n{final_summary}", style="bold green"))


if __name__ == "__main__":
    app()
