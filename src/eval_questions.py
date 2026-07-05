# eval_questions.py
# Manual test dataset for RAGAS evaluation
# Questions and ground truth answers based on Infosys annual reports
# Ground truth must be taken DIRECTLY from the PDF — no paraphrasing

EVAL_DATASET = [
    {
        "question": "What was Infosys revenue in FY2025 on a consolidated basis?",
        "ground_truth": "Infosys reported revenue of ₹1,62,990 crore in FY2025 on a consolidated basis."
    },
    {
        "question": "What is Infosys operating margin for FY2025?",
        "ground_truth": "Infosys operating margin for FY2025 was 21.1%."
    },
    {
        "question": "What was the return on equity (ROE) for Infosys in year 2024?",
        "ground_truth": "ROE was 32.1% in 2024."
    },
    {
        "question": "What was Infosys dividend per share in FY2023?",
        "ground_truth": "Infosys declared a dividend of ₹34 per share in FY2023."
    },
    {
        "question": "What percentage of Infosys revenue comes from exports?",
        "ground_truth": "Approximately 97.1% of Infosys consolidated revenue were export revenues."
    },
    {
        "question": "Which region contributes the highest revenue to Infosys in year 2024?",
        "ground_truth": "North America contributes the highest share of Infosys revenue, i.e., 60.1% in FY2024."
    },
    {
        "question": "Which business segment contributed the most to Infosys revenue in FY2025?",
        "ground_truth": "FS (Includes enterprises in Financial Services and Insurance) contributed 27.7 percent in FY2025."
    },
    {
        "question": "What is ESG Vision 2030 of Infosys?",
        "ground_truth": "As part of our ESG Vision 2030, we aim to achieve 45% female representation in our workforce by 2030. In fiscal 2026, women made up 39.5 % of the total workforce"
    },
    {
        "question": "What was the percentage of revenue generated from top 5 customers in FY2024?",
        "ground_truth": "Revenue from top 5 customers was 11.6% when the year ended on March 31, 2024."
    },
    {
        "question": "What was the 2024 R&D expenditure of Infosys?",
        "ground_truth": "Revenue expenditure on R&D standalone was ₹695 crore in FY2024."
    },
    {
        "question": "Compare Revenue distribution by geographical segments (in %) in India for year 2024 and 2025.",
        "ground_truth": "The revenue distribution in 2024 was 2.5% and in 2025 was 3.1%, thus showing an increase of 0.6%."
    },
    {
        "question": "What is the standalone growth in infosys revenue in fiscal 2025 from fiscal 2024?",
        "ground_truth": "The change in revenue from fiscal 2024 to fiscal 2025 was 5.9%. It went from 1,28,933 cr in 2024 to 1,36,592 cr in 2025."
    },
    {
        "question": "What is the headcount of Infosys in 2026? How many of them percent wise are AI Aware?",
        "ground_truth": "Global Employee headcount of Infosys in 2026 is 3,28,594. Out of which 84% are AI Aware."
    },
    {
        "question": "What was the income tax expense of Infosys in FY2025?",
        "ground_truth": "The income tax expense of Infosys in FY2025 was ₹9873 crore."
    },
    {
        "question": "What was Infosys net profit after tax on a consolidated basis in FY2026?",
        "ground_truth": "Infosys reported a net profit after tax of ₹29,474 crore on a consolidated basis in FY2026."
    }
    
]