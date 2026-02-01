Build a Domain-Specific Customer Support Chatbot for Harel Insurance

# 🚀 Overview

Participants will design and implement an end-to-end GenAI system that:

This capstone simulates a real-world, high-stakes AI systems challenge: building a **production-grade, domain-specific customer support chatbot** for Israel’s largest insurance provider.

- Ingests and structures real insurance policy data
- Answers customer questions across **eight insurance domains**
    
    *(Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment)*
    
- Grounds every answer in official documentation, with explicit citations
- **Outperforms a GPT-5 baseline** using retrieval and agentic system design

This is not a toy demo. The final deliverable should resemble a system that could realistically power an insurer’s first-line support chatbot.

### Why This Challenge Matters

Real-world AI systems fail not because models are weak, but because:

- Data is messy and unstructured
- Knowledge must be grounded and verifiable
- Evaluation is subtle and unforgiving

This challenge puts you in the shoes of an AI startup or platform team:

- Solve a **high-value customer support problem** for a real enterprise
- Build an **end-to-end system**: data → retrieval → agents → evaluation → deployment
- Compete against a **strong foundation-model baseline (GPT-5.2)** and prove when domain-specific RAG wins

### Timeline & Format

- **Dates:** February 1st – February 21st, 2026
- **Teams:** 3–4 participants
- **Mentorship:** 3 live Discord office hours with APEX mentors
- **Final Presentations:** February 22nd (last class)

# **🧠** The Challenge

The challenge unfolds in three conceptual stages:

### **Stage 1: Model Baseline & Evaluation**

**Goal:** Establish a strong baseline and learn how to measure progress.

You will:

- Run evaluation on **GPT-4o** and **GPT-5.2** across a provided development question set
- Test accuracy across different prompt strategies or constraints
- Measure:
    - Answer relevance
    - Hallucination rate
    - Citation failures *(expected to be high)*

Deliverable: Baseline Report

- Where GPT-5 succeeds
- Where it fails without domain grounding

Outcome

- A clear quantitative and qualitative baseline
- An evaluation harness reused in later stages

### **Stage 2: Retrieval Pipeline (RAG Core)**

**Goal:** Beat the baseline with retrieval + grounding.

**You will:**

**1. Prepare the data**

- Parse and clean ASPX + PDF files
- Preserve structure (sections, tables, clauses)
- Chunk documents intelligently

**2. Build retrieval**

- Embed content into a vector store
- Implement semantic search over the corpus
- Return top-k grounded passages with metadata

**3. Generate grounded answers**

- Answer strictly from retrieved context
- Attach citations to every factual claim
- Implement safe fallback when evidence is missing

Outcome

- A working RAG system
- Measurable improvement over GPT-5 on (using open-source model this time)
    - Answer relevance
    - Hallucination rate
    - Citation failures

### **Stage 3: Agentic Flow & Systemization**

**Goal:** Build a robust, production-style AI system.

**You will:**

- Design a single or multi-agent architecture:
- Handle ambiguity and cross-domain questions
- Package the system behind a clean **FastAPI** endpoint

**Optional bonus**

- Voice interface
- Simple UI

**Outcome**

- A realistic customer-support AI system
- Clear separation of concerns (retrieval, reasoning, synthesis)
- API-ready system suitable for real deployment

### 📘 Additional information and materialsAbout Harel Insurance

Harel Insurance is Israel’s largest insurance and financial services group, serving millions of customers across health, life, general insurance, and long-term savings.

In this capstone, Harel serves as a **realistic enterprise customer** with:

- Broad and fragmented product coverage
- Highly regulated, legally precise documentation
- Complex policy structures that vary by product, customer type, and conditions

Your task is to design an AI system capable of operating in this environment where **accuracy, grounding, and trust** are non-negotiable. In this exercise, we’ll focus on **8 key insurance domains**: Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment.

Questions may range from simple eligibility checks to nuanced policy conditions and exclusions.

### Data Provided

All domain knowledge used by your system must be derived **exclusively** from Harel’s official insurance content. You should scrape, ingest, and store the relevant data.

**Dataset characteristics**

- URLs: https://www.harel-group.co.il/insurance/<insurance_type>
- Source types: ASPX web pages and PDFs (do not parse any other document)
- Languages: Hebrew and English (source-dependent)
- Rich structure: tables, bullet lists, conditions, and legal clauses
- Scale: ~350 documents

Every answer must include **document + section/page citation**

To enable iterative development and tuning, you will receive reference questions for a subset of the domains: Travel, Health, Car, Apartment, Life, Business. The final evaluation will be conducted on a **hidden blind question set** covering **all insurance domains in scope**, including unseen domains (Dental, Mortgage, Life).

**What this tests**

- Domain generalization beyond seen examples
- Retrieval robustness across heterogeneous policy structures
- Agent routing and grounding without memorization

## **🏆 The Competition**

You are competing against:

- **GPT-5.2 (baseline)**
- **Other APEX teams**
- …and realistically, against how GenAI systems fail in production 🙂

Your final system will be tested on a **hidden blind question set**.

**Scoring**

- **Relevance (65%)** – Does it correctly answer the question?
- **Citation Accuracy (15%)** – Are sources correct and precise?
- **Efficiency (10%)** – Latency and cost profile
- **Conversational Quality (10%)** – Clarity, tone, flow

**Bonus**

- Voice support: +5%
- UI polish: +5%

## **🛠️ Recommended Open-Source Stack**

*(Not mandatory, but strongly encouraged)*

- **Document Processing:** Docling
- **Vector DB:** Milvus
- **Agent Framework:** LangChain
- **Evaluation:** RAGAS / Opik
- **API:** FastAPI

[Reference Questions.json](attachment:21c1b975-964b-4051-9118-8f1ac96ca5e8:Reference_Questions.json)