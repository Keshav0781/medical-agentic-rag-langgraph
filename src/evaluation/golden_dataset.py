"""
Golden dataset for RAGAS evaluation.
Questions and verified answers from Siemens Healthineers public documents.

At Siemens this dataset was created by domain experts reading source
documents and writing verified question-answer pairs.
For this portfolio project answers verified from pipeline output
and cross-checked against source document page numbers.

Dataset size: 10 questions — sufficient for meaningful RAGAS scores.
At Siemens golden dataset had 50-100 questions reviewed quarterly.
"""

GOLDEN_DATASET = [
    {
        "question": "What was Siemens Healthineers total revenue in fiscal year 2023?",
        "ground_truth": "Siemens Healthineers total revenue in fiscal year 2023 was 21,680 million euros.",
        "source": "Annual Report 2023, Page 53"
    },
    {
        "question": "What was the comparable revenue growth of Siemens Healthineers in fiscal year 2023?",
        "ground_truth": "The comparable revenue growth of Siemens Healthineers in fiscal year 2023 was 1.2%.",
        "source": "Annual Report 2023, Page 17"
    },
    {
        "question": "What was the comparable revenue growth of the Imaging segment in fiscal year 2023?",
        "ground_truth": "The comparable revenue growth of the Imaging segment in fiscal year 2023 was 10.9%.",
        "source": "Annual Report 2023, Page 26"
    },
    {
        "question": "What was the comparable revenue growth of Varian in fiscal year 2023?",
        "ground_truth": "The comparable revenue growth of Varian in fiscal year 2023 was 14.8%.",
        "source": "Annual Report 2023, Page 26"
    },
    {
        "question": "What was the comparable revenue growth of the Diagnostics segment in fiscal year 2023?",
        "ground_truth": "The comparable revenue growth of the Diagnostics segment in fiscal year 2023 was -24.2%.",
        "source": "Annual Report 2023, Page 26"
    },
    {
        "question": "What was Siemens Healthineers comparable revenue growth in fiscal year 2022?",
        "ground_truth": "Siemens Healthineers comparable revenue growth in fiscal year 2022 was 5.9%.",
        "source": "Annual Report 2022, Page 23"
    },
    {
        "question": "How many employees does Siemens Healthineers have?",
        "ground_truth": "Siemens Healthineers has approximately 71,000 employees.",
        "source": "Sustainability Report 2023, Page 9"
    },
    {
        "question": "In how many countries is Siemens Healthineers represented?",
        "ground_truth": "Siemens Healthineers is represented in 70 or more countries.",
        "source": "Sustainability Report 2023, Page 9"
    },
    {
        "question": "What was the net income of Siemens Healthineers in fiscal year 2023?",
        "ground_truth": "The net income of Siemens Healthineers in fiscal year 2023 was 1,525 million euros.",
        "source": "Sustainability Report 2023, Page 9"
    },
    {
        "question": "What were the research and development expenses of Siemens Healthineers in fiscal year 2023?",
        "ground_truth": "The research and development expenses of Siemens Healthineers in fiscal year 2023 were 1,866 million euros.",
        "source": "Annual Report 2023, Page 53"
    }
]