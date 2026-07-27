# 🏥 PubMed RAG: Medical Literature Analysis with BigQuery and Gemini

Build retrieval-augmented applications over medical literature using Google
Cloud BigQuery vector search and Vertex AI Gemini models.

This project converts the user experience from the
[Capricorn Medical Research Application](https://capricorn-medical-research.web.app/)
into notebooks and deployable agents, for clinicians, data scientists, and
engineers alike.

Everything here searches the public
[PubMed Central open-access corpus](https://console.cloud.google.com/marketplace/product/breast-cancer-research/pmc-open-access)
in BigQuery — 2.4 million articles with precomputed embeddings, free to read.

---

## Pick your starting point

### 🚀 For Clinicians (No Coding Required)

1. Open the [Clinician notebook](notebooks/clinician.ipynb)
2. Click **Runtime → Run all** (or press Ctrl/Cmd + F9)
3. Authenticate with your Google account
4. Use the interactive Gradio app to:
   - Paste your medical case notes
   - Extract disease and events automatically
   - Search and analyze PubMed literature
   - Generate comprehensive analysis reports

### 💻 For Data Scientists

1. Open the [Data Scientist notebook](notebooks/data_scientist.ipynb)
2. Configure your Google Cloud project
3. Customize the analysis pipeline:
   ```python
   # Define custom scoring criteria
   CUSTOM_CRITERIA = [
       {"name": "clinical_trial", "weight": 50},
       {"name": "pediatric_focus", "weight": 60},
       # Add your own criteria
   ]

   # Process medical case
   results = process_medical_case(
       case_text,
       default_articles=10,
       min_per_event=3
   )
   ```

### 🤖 For Engineers — deployable agents

Head to [`agents/`](agents/): a catalog of composable agents built on the
[Agent Development Kit](https://google.github.io/adk-docs/), with two
ready-made pipelines and a one-command local run.

```bash
export PROJECT_ID=your-gcp-project-id
pip install -r agents/requirements.txt
python agents/setup_bigquery.py
python agents/run_local.py --pipeline full_review
```

[`agents/README.md`](agents/README.md) covers the catalog, the composition
contract, how to build your own pipeline, and deployment to Vertex AI Agent
Engine.

---

## Architecture

![Medical Literature Analysis Architecture](https://github.com/google/pubmed-rag/blob/main/visuals/1.png?raw=true)

## Repository layout

| Path | Contents |
| :-- | :-- |
| [`notebooks/`](notebooks/) | Colab notebooks for clinicians and data scientists |
| [`agents/`](agents/) | Composable ADK agents, pipelines, and the shared engine |
| [`data/`](data/) | SCImago journal impact scores, loaded into BigQuery by the setup script |
| [`visuals/`](visuals/) | Diagrams |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.

This software is intended for research and educational use. It does not provide
medical advice and must not be used for clinical decision-making.
