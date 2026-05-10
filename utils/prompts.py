def build_pdf_analysis_prompt(extracted_text):

    return f"""
You are an expert Indian legal analyst.

Analyze the following legal document and provide:

1. SUMMARY
2. FACTS OF THE CASE
3. KEY ARGUMENTS
4. SECTIONS INVOLVED
5. FINAL JUDGMENT
6. CHARGES
7. NEXT STEPS

Document:
{extracted_text[:5000]}
"""


def build_followup_prompt(context, question):

    return f"""
You are an AI Legal Assistant.

Context:
{context}

Question:
{question}

Answer clearly and professionally.
"""


def build_compare_prompt(textA, textB):

    return f"""
You are an AI Legal Assistant.

Compare the following documents.

Document A:
{textA}

Document B:
{textB}

Provide:

1. Similarities
2. Differences
3. Missing Clauses
4. Legal Analysis
"""