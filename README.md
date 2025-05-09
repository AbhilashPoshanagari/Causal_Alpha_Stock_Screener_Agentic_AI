
# 📌 Causal Alpha: A Real-world Stock Investment Decision-enablement Solution

---

## 📖 Overview

Stock price prediction is a critical yet complex task due to the inherent volatility and non-linearity of financial markets. Traditional time-series models often fail to capture interdependencies between financial entities and macroeconomic influences.

This project aims to provide an interface for an actionable decision using a modular, agentic system powered by large language models (LLMs) and real-time web search.

## Technologies used
- Google Drive + JSON cache for storage
- Gradio UI for live demos and interaction
- FastAPI for future scalability and Angular UI

---

## ✨ Features

### List the key features or functionality:
### Agentic Capabilities : 
- ✅ Tools Use : Integrated web search capabilities to access real-time market data
- ✅ Multi-agent Architecture : Purpose-built orchestrator agent coordinating specialized analysis sub-agents
- ✅ Memory and Context Preservation : Maintains state between agent calls through context accumulation
- ✅ Reflection and Review : Dedicated review agent to evaluate and refine outputs from analysis agents
- ✅ Adaptive Weighting : Temporal and confidence-based adjustments to synthesize final recommendations
- ✅ Structured Output : Standardized JSON format for consistent interpretation and presentation
  
  Agent modules at  : notebooks/Team_2_Causal_Alpha_Stock_screener_Agentic_AI_v17.ipynb
  
## 🛠️  Installation

```bash
# Clone the repo
git clone https://github.com/AbhilashPoshanagari/Causal_Alpha_Stock_Screener_Agentic_AI.git
# In google colab, run the setup from the ngrok_setup.ipynb
# Run the server in the codeblock and grab the domain
!python serve_with_ngrok.py
# Connect to Angular UI at 
https://abhilashposhanagari.github.io/Causal_Alpha_Stock_Screener_Agentic_AI/stock_screener
# From the UI, connect to ngrok domain
# Reload UI and run analysis with the chosen ticker.

