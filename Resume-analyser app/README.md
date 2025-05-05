# Advanced Resume Analyzer Pro

A powerful AI-driven tool that provides comprehensive analysis of resumes against job descriptions. This tool uses advanced algorithms to deliver industry-specific insights, detailed scoring, and actionable recommendations to optimize your resume.

## Features

### Core Analysis
- Upload PDF resumes or paste resume text
- Analyze resume against job descriptions
- **Resume Score out of 100** with detailed scoring breakdown
- **ATS Compatibility Check** to ensure your resume passes automated screening
- **Keyword Density Analysis** to optimize for important terms
- **Resume Section Completeness** to identify missing sections

### Advanced Features
- **Industry Detection** - Automatically identifies your industry and provides relevant insights
- **Job Level Detection** - Detects job seniority level and customizes analysis accordingly
- **Resume Format Analysis** - Evaluates formatting, structure, and visual presentation
- **Buzzword & Cliché Detection** - Identifies overused phrases that weaken your resume
- **Readability Analysis** - Calculates how easy your resume is to read and understand
- **Employment Gap Detection** - Finds potential gaps in your work history
- **Industry Benchmark Comparison** - Shows how your resume compares to industry standards
- **Job Title Relevance** - Analyzes how well your job titles match the position
- **Resume Freshness** - Evaluates how recently your resume has been updated

### Visualizations
- Interactive score gauge with color-coded ranges
- Resume quality radar chart showing strengths and weaknesses
- Job fit radar chart comparing your profile to the job requirements
- Visual resume section completeness display
- Keyword density charts and visualizations

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Choose your preferred method to input your resume (PDF upload or text paste)
2. Paste the job description in the provided text area
3. Click "Analyze Resume" to get a detailed analysis
4. Review the results and suggestions for improvement
5. Check the different tabs for detailed information about:
   - Resume Content (skills, education, experience, keyword density)
   - Strengths & Gaps (strengths, weaknesses, section completeness)
   - ATS Compatibility (issues that might prevent automated systems from reading your resume)
   - Format Analysis (layout, structure, and presentation evaluation)
   - Advanced Insights (readability, buzzwords, employment gaps, industry benchmarks)
   - Improvement Tips (custom suggestions and general resume advice)

## Adaptive Scoring System

The resume analyzer uses an adaptive scoring system that adjusts based on job level:

### Entry Level
- Keyword matching (30%)
- Technical skills (25%) 
- Education (25%)
- Experience (10%)
- Certifications (10%)

### Mid Level
- Keyword matching (25%)
- Technical skills (30%)
- Education (15%)
- Experience (20%)
- Certifications (10%)

### Senior Level
- Keyword matching (20%)
- Technical skills (25%)
- Education (10%)
- Experience (35%)
- Certifications (10%)

### Executive
- Keyword matching (15%)
- Technical skills (20%)
- Education (10%)
- Experience (40%)
- Certifications (15%)

## Industry-Specific Analysis

The analyzer provides customized analysis for various industries including:
- Tech/IT
- Finance
- Healthcare
- Marketing
- Engineering
- Education
- Sales
- Design

Each industry has its own benchmarks for skills, experience, certifications, and resume format.

## Requirements

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)

## Implementation Details

This Resume Analyzer implements advanced text analysis techniques:
- Regular expressions for pattern matching
- Custom tokenization for text processing
- Statistical analysis of resume content
- Heuristic scoring algorithms
- Industry-specific benchmark comparisons
- Advanced readability formulas
- Format and structure evaluation

This project uses sophisticated rule-based analysis that functions completely offline with no external API dependencies, ensuring privacy and fast performance. 