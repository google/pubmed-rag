# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Final synthesis: scored articles in, markdown literature review out."""

import logging
from typing import Any

from . import llm
from .links import scored_table

logger = logging.getLogger(__name__)

ANALYSIS_TEMPLATE = """You are a research analyst synthesizing findings from a comprehensive literature review. Your goal is to provide insights that are valuable for research purposes.

RESEARCH CONTEXT:
Original Query/Case: {case_description}

Primary Focus: {primary_focus}
Key Concepts Searched: {key_concepts}

ANALYZED ARTICLES:
{articles_content}

Based on the research context and analyzed articles above, please provide a comprehensive synthesis in markdown format with the following sections:

## Literature Analysis: {primary_focus}

### 1. Executive Summary
Provide a concise overview of the key findings from the literature review, highlighting:
- Main themes identified across the literature
- Most significant insights relevant to the research query
- Overall quality and quantity of available evidence
- Key takeaways for researchers in this field

### 2. Key Findings by Concept
| Concept | Articles Discussing | Key Findings | Evidence Quality |
|---------|-------------------|--------------|------------------|
[For each key concept searched, summarize what the literature reveals about it. In "Articles Discussing", list articles using their PMCID as clickable links, e.g., [PMC7654321](https://pmc.ncbi.nlm.nih.gov/articles/PMC7654321/)]

### 3. Methodological Landscape
| Research Method | Frequency | Notable Studies | Insights Generated |
|-----------------|-----------|-----------------|-------------------|
[Map the research methodologies used across the analyzed articles. Reference studies by PMCID]

### 4. Temporal Trends
| Time Period | Research Focus | Key Developments | Paradigm Shifts |
|-------------|----------------|------------------|-----------------|
[Analyze how research in this area has evolved over time. Cite articles using PMCID]

### 5. Cross-Study Patterns
| Pattern | Supporting Evidence | Implications | Confidence Level |
|---------|-------------------|--------------|------------------|
[Identify patterns that appear across multiple studies. List supporting evidence with PMCID references]

### 6. Controversies & Unresolved Questions
| Issue | Different Perspectives | Evidence For/Against | Current Consensus |
|-------|----------------------|---------------------|-------------------|
[Highlight areas of disagreement or ongoing debate in the literature. Cite specific articles by PMCID]

### 7. Knowledge Gaps & Future Research
| Gap Identified | Why It Matters | Potential Approaches | Expected Impact |
|----------------|----------------|---------------------|-----------------|
[Map areas where further research is needed based on the analyzed articles]

### 8. Practical Applications
Based on the synthesized literature, identify:
- How these findings can be applied in practice
- Recommendations for researchers entering this field
- Tools, methods, or frameworks that emerge from the literature
- Potential interdisciplinary connections

### 9. Quality & Reliability Assessment
Evaluate the overall body of literature:
- **Study Types**: Distribution of research designs (experimental, observational, reviews, etc.)
- **Sample Characteristics**: Common sample sizes, populations studied
- **Geographic Distribution**: Where research is being conducted
- **Publication Patterns**: Journal quality, publication years, citation patterns
- **Methodological Rigor**: Strengths and limitations observed

### 10. Synthesis & Conclusions
Provide an integrated narrative that:
- Connects findings across all analyzed articles
- Identifies the strongest evidence and most reliable findings
- Suggests how this research area is likely to develop
- Offers guidance for stakeholders interested in this topic

### 11. Bibliography
**Most Relevant Articles** (in order of relevance to the research query):
[For each article, format as follows:
- Title, Journal (Year). [PMCID: PMCxxxxxx](https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxx/) | [PMID: xxxxxxxx](https://pubmed.ncbi.nlm.nih.gov/xxxxxxxx/)]

---

IMPORTANT NOTES:
- When referencing articles throughout the analysis, ALWAYS use their PMCID or PMID identifiers, not generic labels like "Article 1"
- Format all article references as clickable links: [PMCxxxxxx](https://pmc.ncbi.nlm.nih.gov/articles/PMCxxxxxx/)
- Maintain objectivity and clearly distinguish between strong evidence and preliminary findings
- Use accessible language while preserving scientific accuracy
- All claims must be traceable to specific articles in the analysis
- When evidence is conflicting, present all viewpoints fairly
- Focus on research insights and knowledge synthesis rather than prescriptive recommendations
- Highlight both the strengths and limitations of the current literature
"""


def _articles_block(articles: list[dict[str, Any]]) -> str:
    parts = []
    for index, article in enumerate(articles, 1):
        parts.append(
            f"""
Article {index}:
Title: {article.get('title', 'Unknown')}
Journal: {article.get('journal_title', 'Unknown')} | Year: {article.get('year', 'N/A')}
Type: {article.get('paper_type', 'Unknown')}
Score: {article.get('score', 0)}
PMID: {article.get('pmid', 'N/A')} | PMCID: {article.get('pmc_id', 'N/A')}

Full Text excerpt:
{(article.get('content') or 'No content available')[:5000]}
"""
        )
    return "\n" + "=" * 80 + "\n".join(parts)


def build_report(
    case_notes: str,
    disease: str,
    concepts: list[str],
    scored_articles: list[dict[str, Any]],
    *,
    top_n: int = 10,
) -> str:
    """Render the scored-results table followed by the LLM synthesis."""
    if not scored_articles:
        return "No scored articles available to synthesize."

    top = scored_articles[:top_n]
    logger.info("building report over %d of %d articles", len(top), len(scored_articles))

    prompt = ANALYSIS_TEMPLATE.format(
        case_description=case_notes,
        primary_focus=disease,
        key_concepts=", ".join(concepts),
        articles_content=_articles_block(top),
    )
    synthesis = llm.generate_text(prompt, temperature=0.3, max_output_tokens=8192)

    header = f"### Scored Search Results (Top {len(top)})\n\n{scored_table(top)}\n\n---\n\n"
    return header + synthesis
