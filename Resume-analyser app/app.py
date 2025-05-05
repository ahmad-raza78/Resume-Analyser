import streamlit as st

# Set page config (MUST be called before any other Streamlit commands)
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

import os
from PyPDF2 import PdfReader
import re
import random
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define stopwords for text processing
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
            'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'of', 
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'do', 'does', 'did', 'i', 'you', 'he', 'she', 'it', 
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that', 
            'these', 'those', 'would', 'should', 'could', 'will', 'shall', 
            'may', 'might', 'must', 'can'}

# Industry benchmarks (average ideal scores for different aspects)
INDUSTRY_BENCHMARKS = {
    "Tech/IT": {
        "skills_count": 12,
        "experience_years": 3,
        "education_level": "Bachelor's",
        "certifications_count": 2,
        "bullet_points_per_job": 5,
        "action_verbs_percentage": 25,
        "technical_skills_percentage": 60,
        "soft_skills_percentage": 40
    },
    "Finance": {
        "skills_count": 10,
        "experience_years": 4,
        "education_level": "Bachelor's",
        "certifications_count": 2,
        "bullet_points_per_job": 6,
        "action_verbs_percentage": 30,
        "technical_skills_percentage": 50,
        "soft_skills_percentage": 50
    },
    "Healthcare": {
        "skills_count": 11,
        "experience_years": 3,
        "education_level": "Associate",
        "certifications_count": 3,
        "bullet_points_per_job": 4,
        "action_verbs_percentage": 20,
        "technical_skills_percentage": 70,
        "soft_skills_percentage": 30
    },
    "Marketing": {
        "skills_count": 12,
        "experience_years": 2,
        "education_level": "Bachelor's",
        "certifications_count": 1,
        "bullet_points_per_job": 5,
        "action_verbs_percentage": 35,
        "technical_skills_percentage": 40,
        "soft_skills_percentage": 60
    },
    "Engineering": {
        "skills_count": 10,
        "experience_years": 3,
        "education_level": "Bachelor's",
        "certifications_count": 1,
        "bullet_points_per_job": 5,
        "action_verbs_percentage": 25,
        "technical_skills_percentage": 70,
        "soft_skills_percentage": 30
    },
    "Education": {
        "skills_count": 9,
        "experience_years": 3,
        "education_level": "Master's",
        "certifications_count": 2,
        "bullet_points_per_job": 4,
        "action_verbs_percentage": 30,
        "technical_skills_percentage": 30,
        "soft_skills_percentage": 70
    },
    "Sales": {
        "skills_count": 10,
        "experience_years": 2,
        "education_level": "Bachelor's",
        "certifications_count": 1,
        "bullet_points_per_job": 6,
        "action_verbs_percentage": 40,
        "technical_skills_percentage": 30,
        "soft_skills_percentage": 70
    },
    "Design": {
        "skills_count": 11,
        "experience_years": 2,
        "education_level": "Bachelor's",
        "certifications_count": 1,
        "bullet_points_per_job": 4,
        "action_verbs_percentage": 35,
        "technical_skills_percentage": 60,
        "soft_skills_percentage": 40
    },
    "General": {  # Default benchmark
        "skills_count": 10,
        "experience_years": 3,
        "education_level": "Bachelor's",
        "certifications_count": 1,
        "bullet_points_per_job": 5,
        "action_verbs_percentage": 30,
        "technical_skills_percentage": 50,
        "soft_skills_percentage": 50
    }
}

# Scoring weights by job level
JOB_LEVEL_WEIGHTS = {
    "Entry Level": {
        "Keyword Match": 0.3,
        "Technical Skills": 0.25,
        "Education": 0.25,
        "Experience": 0.1,
        "Certifications": 0.1
    },
    "Mid Level": {
        "Keyword Match": 0.25,
        "Technical Skills": 0.3,
        "Education": 0.15,
        "Experience": 0.2,
        "Certifications": 0.1
    },
    "Senior Level": {
        "Keyword Match": 0.2,
        "Technical Skills": 0.25,
        "Education": 0.1,
        "Experience": 0.35,
        "Certifications": 0.1
    },
    "Executive": {
        "Keyword Match": 0.15,
        "Technical Skills": 0.2,
        "Education": 0.1,
        "Experience": 0.4,
        "Certifications": 0.15
    }
}

# List of common resume clichés and buzzwords
RESUME_CLICHES = [
    "team player", "hard worker", "detail-oriented", "motivated", 
    "self-starter", "go-getter", "think outside the box", "results-driven", 
    "driven", "dynamic", "proactive", "synergy", "go-to person", 
    "track record", "bottom line", "value add", "results-oriented",
    "best of breed", "gave 110 percent", "win-win", "team player", 
    "fast-paced environment", "problem solver", "entrepreneurial"
]

# Define industry keywords for different sectors
INDUSTRY_KEYWORDS = {
    "Tech/IT": [
        "software", "developer", "engineer", "programming", "code", "java", "python", "javascript", 
        "react", "node", "database", "cloud", "aws", "azure", "devops", "agile", "scrum", "fullstack", 
        "frontend", "backend", "web", "mobile", "app", "development", "algorithm", "data structure",
        "microservices", "api", "rest", "soap", "git", "ci/cd", "testing", "qa", "debugging"
    ],
    "Finance": [
        "finance", "accounting", "investment", "banking", "portfolio", "financial analysis", "trading", 
        "stocks", "bonds", "securities", "assets", "liabilities", "capital", "equity", "balance sheet", 
        "income statement", "cash flow", "budget", "forecast", "audit", "tax", "compliance", "risk management",
        "wealth management", "financial planning", "loan", "credit", "debt", "interest rate"
    ],
    "Healthcare": [
        "medical", "healthcare", "patient", "clinical", "doctor", "nurse", "physician", "hospital", 
        "treatment", "diagnosis", "therapy", "pharmaceutical", "medicine", "surgery", "care", "health", 
        "insurance", "records", "billing", "coding", "telehealth", "ehr", "emr", "hipaa", "compliance",
        "regulatory", "patient care", "bedside", "vitals", "symptoms", "prognosis"
    ],
    "Marketing": [
        "marketing", "branding", "advertising", "campaign", "social media", "digital marketing", "seo", 
        "content", "market research", "analytics", "audience", "consumer", "customer", "engagement", 
        "conversion", "lead generation", "funnel", "brand awareness", "strategy", "copywriting",
        "creative", "media buying", "public relations", "pr", "influencer", "viral", "metrics"
    ],
    "Engineering": [
        "engineering", "mechanical", "electrical", "civil", "design", "manufacturing", "construction", 
        "project", "cad", "blueprint", "specification", "simulation", "modeling", "prototype", "testing", 
        "quality assurance", "safety", "compliance", "industrial", "materials", "structure", "process",
        "automation", "robotics", "instrument", "control system", "maintenance", "reliability"
    ],
    "Education": [
        "education", "teaching", "learning", "curriculum", "instruction", "classroom", "student", "school", 
        "assessment", "evaluation", "pedagogy", "e-learning", "training", "development", "educational technology", 
        "lesson plan", "course", "program", "academic", "research", "professor", "instructor", "faculty",
        "administration", "higher education", "k-12", "educational leadership"
    ],
    "Sales": [
        "sales", "business development", "account management", "client", "customer", "revenue", "pipeline", 
        "quota", "target", "negotiate", "close", "deal", "prospect", "lead", "opportunity", "territory", 
        "relationship", "crm", "solution selling", "consultative selling", "presentation", "proposal",
        "objection handling", "cold calling", "follow-up", "upsell", "cross-sell", "customer retention"
    ],
    "Design": [
        "design", "graphic", "ui", "ux", "user experience", "user interface", "visual", "creative", "artwork", 
        "illustration", "typography", "layout", "color theory", "branding", "logo", "identity", "wireframe", 
        "prototype", "usability", "accessibility", "responsive", "mobile", "web", "print", "packaging",
        "animation", "video", "motion graphics", "3d", "adobe", "sketch", "figma", "invision"
    ]
}

# Job titles for each industry
INDUSTRY_JOB_TITLES = {
    "Tech/IT": [
        "software engineer", "developer", "programmer", "data scientist", "web developer", 
        "devops engineer", "system administrator", "database administrator", "network engineer", 
        "security analyst", "it manager", "product manager", "scrum master", "qa engineer",
        "technical lead", "cto", "it director", "information security"
    ],
    "Finance": [
        "financial analyst", "accountant", "controller", "auditor", "tax specialist", 
        "investment banker", "financial advisor", "wealth manager", "risk analyst", 
        "credit analyst", "loan officer", "portfolio manager", "financial planner", "trader",
        "underwriter", "actuary", "compliance officer", "finance director", "cfo"
    ],
    "Healthcare": [
        "physician", "nurse", "medical assistant", "therapist", "healthcare administrator", 
        "medical director", "healthcare manager", "clinical coordinator", "lab technician", 
        "radiologist", "pharmacist", "dental hygienist", "physical therapist", "nutritionist",
        "medical coder", "billing specialist", "health information manager"
    ],
    "Marketing": [
        "marketing manager", "marketing coordinator", "digital marketer", "seo specialist", 
        "content writer", "social media manager", "brand manager", "market researcher", 
        "public relations", "communications specialist", "marketing director", "campaign manager",
        "marketing analyst", "growth hacker", "copywriter", "media planner", "cmo"
    ],
    "Engineering": [
        "mechanical engineer", "electrical engineer", "civil engineer", "chemical engineer", 
        "aerospace engineer", "industrial engineer", "process engineer", "quality engineer", 
        "safety engineer", "project engineer", "structural engineer", "design engineer",
        "manufacturing engineer", "automation engineer", "engineering manager", "chief engineer"
    ],
    "Education": [
        "teacher", "instructor", "professor", "principal", "dean", "educational director", 
        "curriculum developer", "academic advisor", "school counselor", "trainer", 
        "e-learning specialist", "education coordinator", "researcher", "superintendent",
        "academic coach", "tutor", "education consultant", "instructional designer"
    ],
    "Sales": [
        "sales representative", "account executive", "business development", "sales manager", 
        "sales director", "account manager", "client relationship manager", "territory manager", 
        "sales consultant", "inside sales", "outside sales", "regional sales manager",
        "sales coordinator", "sales analyst", "customer success manager", "vp of sales"
    ],
    "Design": [
        "graphic designer", "ui designer", "ux designer", "web designer", "product designer", 
        "art director", "creative director", "visual designer", "interactive designer", 
        "industrial designer", "fashion designer", "interior designer", "design manager",
        "brand designer", "motion designer", "3d artist", "illustrator", "photographer"
    ]
}

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Common skills keywords
SKILL_KEYWORDS = [
    # Programming Languages
    "python", "java", "javascript", "c\\+\\+", "c#", "ruby", "php", "swift", "typescript", "go", "rust", "kotlin", "perl", "scala", "r",
    # Web Development
    "html", "css", "react", "angular", "vue", "node", "express", "django", "flask", "spring", "asp\\.net", "laravel", "ruby on rails",
    # Data Science / ML
    "machine learning", "deep learning", "data science", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", 
    "data analysis", "statistics", "natural language processing", "computer vision", "neural networks", "big data",
    # DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd", "git", "terraform", "ansible", "chef", "puppet",
    # Database
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "nosql", "redis", "elasticsearch", "dynamodb",
    # Mobile
    "android", "ios", "flutter", "react native", "swift", "mobile development", "xamarin",
    # General Tech
    "agile", "scrum", "jira", "rest api", "graphql", "microservices", "testing", "debugging", "api", "oop", "functional programming",
    # Soft Skills
    "leadership", "communication", "teamwork", "problem solving", "critical thinking", "time management", "collaboration",
    # Professional
    "project management", "sales", "marketing", "customer service", "accounting", "finance", "consulting", "hr", "recruitment",
    # Creative
    "photoshop", "illustrator", "indesign", "ui/ux", "design", "video editing", "animation", "3d modeling", "figma",
    # Business
    "microsoft office", "excel", "powerpoint", "word", "google docs", "analytics", "seo", "social media", "content writing"
]

# Common education keywords
EDUCATION_KEYWORDS = [
    "bachelor", "master", "phd", "doctorate", "mba", "bs", "ms", "ba", "ma", "b.tech", "m.tech", "bsc", "msc",
    "associate degree", "diploma", "certificate", "high school", "graduate", "undergraduate", "post-graduate"
]

# Common certifications
CERTIFICATION_KEYWORDS = [
    "certified", "certification", "certificate", "aws certified", "microsoft certified", "cisco certified", "comptia",
    "pmp", "cfa", "cpa", "ccna", "ccnp", "mcsa", "mcse", "itil", "prince2", "six sigma", "capm", "scrum", "agile", "lean",
    "professional", "specialist", "expert", "associate", "foundation", "advanced", "practitioner"
]

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_skills(text, job_description):
    """Extract skills from resume text"""
    skills = []
    text_lower = text.lower()
    
    # First find common skills from our predefined list
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + skill + r'\b', text_lower):
            skills.append(skill.title())
    
    # Get skills mentioned in the job description to prioritize
    job_skills = []
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + skill + r'\b', job_description.lower()):
            job_skills.append(skill.title())
    
    # Add any skills from job description that were found in resume but not in our keyword list
    job_desc_words = set(job_description.lower().split())
    resume_words = set(text_lower.split())
    common_words = job_desc_words.intersection(resume_words)
    
    for word in common_words:
        if len(word) > 3 and word not in ['and', 'the', 'for', 'with', 'that', 'this', 'have', 'from']:
            if word.title() not in skills and word.title() not in job_skills:
                skills.append(word.title())
    
    # Combine and prioritize skills mentioned in job description
    all_skills = job_skills + [s for s in skills if s not in job_skills]
    
    # Return top skills (limit to 15)
    return all_skills[:15]

def extract_education(text):
    """Extract education details from resume text"""
    education = []
    text_lower = text.lower()
    
    # Find sentences with education keywords
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        for keyword in EDUCATION_KEYWORDS:
            if keyword in sentence.lower():
                # Clean the sentence
                clean_sentence = sentence.strip()
                if clean_sentence and len(clean_sentence) > 10:
                    education.append(clean_sentence)
                break
    
    # Remove duplicates and limit
    education = list(set(education))
    return education[:5]

def extract_experience(text):
    """Extract work experience from resume text"""
    experience = []
    
    # Common job title keywords
    job_titles = ["engineer", "developer", "manager", "director", "specialist", 
                 "analyst", "assistant", "associate", "consultant", "coordinator",
                 "designer", "administrator", "supervisor", "lead", "head"]
    
    # Find paragraphs with job titles
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        for title in job_titles:
            if title.lower() in paragraph.lower():
                # Clean the paragraph
                clean_paragraph = paragraph.strip()
                if clean_paragraph and len(clean_paragraph) > 15:
                    # Limit length of very long paragraphs
                    if len(clean_paragraph) > 100:
                        clean_paragraph = clean_paragraph[:100] + "..."
                    experience.append(clean_paragraph)
                break
    
    # If no experience found with this method, try to find date patterns
    if not experience:
        date_pattern = r'(20\d{2}|19\d{2})\s*-\s*(20\d{2}|19\d{2}|present|current|now)'
        date_matches = re.finditer(date_pattern, text, re.IGNORECASE)
        
        for match in date_matches:
            # Get the surrounding text (50 chars before, 100 after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()
            
            if len(context) > 15:
                experience.append(context)
    
    # Remove duplicates and limit
    experience = list(set(experience))
    return experience[:5]

def extract_certifications(text):
    """Extract certifications from resume text"""
    certifications = []
    text_lower = text.lower()
    
    # Find sentences with certification keywords
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        for keyword in CERTIFICATION_KEYWORDS:
            if keyword in sentence.lower():
                # Clean the sentence
                clean_sentence = sentence.strip()
                if clean_sentence and len(clean_sentence) > 10:
                    certifications.append(clean_sentence)
                break
    
    # Remove duplicates and limit
    certifications = list(set(certifications))
    return certifications[:5]

def analyze_strengths_weaknesses(skills, job_description):
    """Analyze strengths and weaknesses based on job description"""
    strengths = []
    weaknesses = []
    
    # Extract skills from job description
    job_skills = []
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + skill + r'\b', job_description.lower()):
            job_skills.append(skill.title())
    
    # Find matching skills (strengths)
    for skill in skills:
        if skill.lower() in job_description.lower() or any(s.lower() == skill.lower() for s in job_skills):
            strengths.append(f"Proficiency in {skill}")
    
    # Find skills in job description that are not in resume (weaknesses)
    for skill in job_skills:
        if skill.title() not in skills and skill.title() not in [s.title() for s in skills]:
            weaknesses.append(f"No mention of {skill} which is required/preferred")
    
    # If we don't have enough strengths, add generic ones
    generic_strengths = [
        "Strong communication skills",
        "Attention to detail",
        "Problem-solving abilities",
        "Team collaboration experience",
        "Project management skills"
    ]
    
    random.shuffle(generic_strengths)
    strengths.extend(generic_strengths[:max(0, 3 - len(strengths))])
    
    # If we don't have enough weaknesses, add generic ones
    generic_weaknesses = [
        "Resume could highlight more specific achievements",
        "Could include more quantifiable results",
        "More detail about project involvement would be beneficial",
        "Limited demonstration of leadership experience",
        "Could better align experience with job requirements"
    ]
    
    random.shuffle(generic_weaknesses)
    weaknesses.extend(generic_weaknesses[:max(0, 3 - len(weaknesses))])
    
    return strengths[:5], weaknesses[:5]

def tokenize_text(text):
    """Simple tokenization function without NLTK"""
    # Split by spaces and clean up words
    words = text.lower().split()
    # Remove punctuation and filter out stopwords
    words = [word.strip('.,!?()[]{}:;"\'') for word in words]
    words = [word for word in words if word.isalpha() and word not in STOPWORDS and len(word) > 2]
    return words

def calculate_matching_score(resume_text, job_description, job_level="Mid Level"):
    """Calculate detailed matching score between resume and job description (out of 100)"""
    # Get weights based on job level
    weights = JOB_LEVEL_WEIGHTS.get(job_level, JOB_LEVEL_WEIGHTS["Mid Level"])
    
    # Break down scoring into separate components
    
    # Component 1: Keyword matching
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())
    
    # Remove common words
    jd_words = jd_words - STOPWORDS
    
    # Calculate keyword matches
    matches = sum(1 for word in jd_words if word in resume_words)
    keyword_score = min(100, (matches / max(1, len(jd_words))) * 100)
    
    # Component 2: Technical skills match
    skill_matches = 0
    total_skills_in_jd = 0
    
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + skill + r'\b', job_description.lower()):
            total_skills_in_jd += 1
            if re.search(r'\b' + skill + r'\b', resume_text.lower()):
                skill_matches += 1

    skills_score = 0
    if total_skills_in_jd > 0:
        skills_score = min(100, (skill_matches / total_skills_in_jd) * 100)
    
    # Component 3: Education keywords
    education_score = 0
    education_matches = 0
    for keyword in EDUCATION_KEYWORDS:
        if keyword in job_description.lower() and keyword in resume_text.lower():
            education_matches += 1
    
    education_score = min(100, education_matches * 25)
    
    # Component 4: Experience relevance
    # Extract potential years of experience from job description
    experience_required = re.search(r'(\d+)[+]?\s+years', job_description.lower())
    experience_found = re.search(r'(\d+)[+]?\s+years', resume_text.lower())
    
    experience_score = 33  # Base score
    
    # Adjust based on years if found
    if experience_required and experience_found:
        req_years = int(experience_required.group(1))
        found_years = int(experience_found.group(1))
        
        if found_years >= req_years:
            experience_score = 100
        else:
            experience_score = 33 + min(67, (found_years / req_years) * 67)
    
    # Component 5: Certification matches
    certification_score = 0
    certification_matches = 0
    for cert in CERTIFICATION_KEYWORDS:
        if cert in job_description.lower() and cert in resume_text.lower():
            certification_matches += 1
    
    certification_score = min(100, certification_matches * 20)
    
    # Calculate weighted score
    total_score = (
        keyword_score * weights["Keyword Match"] +
        skills_score * weights["Technical Skills"] +
        education_score * weights["Education"] +
        experience_score * weights["Experience"] +
        certification_score * weights["Certifications"]
    )
    
    # Store detailed scores for visualization
    score_breakdown = {
        "Keyword Match": keyword_score,
        "Technical Skills": skills_score,
        "Education": education_score,
        "Experience": experience_score,
        "Certifications": certification_score
    }
    
    return round(total_score), score_breakdown

def analyze_keyword_density(resume_text):
    """Analyze keyword density in the resume without NLTK"""
    # Simple tokenization
    words = tokenize_text(resume_text)
    
    # Count word frequency
    word_counts = Counter(words)
    
    # Get top keywords
    top_keywords = word_counts.most_common(10)
    
    return top_keywords

def check_ats_compatibility(resume_text):
    """Check resume for ATS compatibility issues"""
    issues = []
    
    # Check for tables (might be inaccurate in text extraction)
    if re.search(r'\|', resume_text) or re.search(r'\+[-]+\+', resume_text):
        issues.append("Potential tables detected - consider using plain text formatting")
    
    # Check for special characters
    if re.search(r'[^\w\s.,;:?!()\-\'\"/$%&]', resume_text):
        issues.append("Special characters detected - may cause issues with some ATS systems")
    
    # Check for headers/footers (simplified check)
    lines = resume_text.split('\n')
    if len(lines) > 5:
        first_line = lines[0].strip().lower()
        last_line = lines[-1].strip().lower()
        
        if re.search(r'page \d+|^\d+$', first_line) or re.search(r'page \d+|^\d+$', last_line):
            issues.append("Page numbers detected - may interfere with ATS parsing")
    
    # Check for common file format issues
    if len(resume_text) < 100:
        issues.append("Very short text detected - possible PDF extraction issues")
    
    # If no issues found
    if not issues:
        issues.append("No obvious ATS compatibility issues detected")
    
    return issues

def analyze_section_completeness(resume_text):
    """Analyze completeness of resume sections"""
    sections = {
        "Contact Information": 0,
        "Professional Summary": 0,
        "Work Experience": 0,
        "Education": 0,
        "Skills": 0,
        "Projects": 0,
        "Certifications": 0,
        "References": 0
    }
    
    # Check for contact information
    if re.search(r'email|phone|address|linkedin', resume_text.lower()):
        sections["Contact Information"] = 1
    
    # Check for summary
    if re.search(r'summary|objective|profile', resume_text.lower()):
        sections["Professional Summary"] = 1
    
    # Check for work experience section
    if re.search(r'experience|work|employment|job', resume_text.lower()):
        sections["Work Experience"] = 1
    
    # Check for education
    if re.search(r'education|degree|university|college|school', resume_text.lower()):
        sections["Education"] = 1
    
    # Check for skills
    if re.search(r'skill|proficiency|competenc', resume_text.lower()):
        sections["Skills"] = 1
    
    # Check for projects
    if re.search(r'project|portfolio', resume_text.lower()):
        sections["Projects"] = 0.5  # Optional section
    
    # Check for certifications
    if re.search(r'certification|certificate|certified', resume_text.lower()):
        sections["Certifications"] = 0.5  # Optional section
    
    # Check for references
    if re.search(r'reference', resume_text.lower()):
        sections["References"] = 0.5  # Optional section
    
    # Calculate overall completeness (excluding optional sections that are worth 0.5)
    required_sections = sum(1 for k, v in sections.items() if v == 1)
    total_required = 5  # Number of required sections
    
    completeness_score = (required_sections / total_required) * 100
    
    return sections, completeness_score

def generate_suggestions(strengths, weaknesses, skills, job_description):
    """Generate suggestions for resume improvement"""
    suggestions = []
    
    # Add suggestions based on weaknesses
    for weakness in weaknesses:
        if "no mention of" in weakness.lower():
            skill = weakness.lower().replace("no mention of", "").strip().rstrip('.')
            if skill:
                suggestions.append(f"Add {skill} to your skills section if you have experience with it")
    
    # Check for skills mentioned in job description but not in resume
    jd_skills = []
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + skill + r'\b', job_description.lower()) and not any(s.lower() == skill for s in skills):
            jd_skills.append(skill)
    
    if jd_skills:
        random.shuffle(jd_skills)
        for skill in jd_skills[:2]:
            suggestions.append(f"Highlight {skill.title()} skills if you have them")
    
    # Generic suggestions
    generic_suggestions = [
        "Quantify your achievements with specific metrics and results",
        "Tailor your resume to match the specific job description",
        "Use action verbs to describe your accomplishments",
        "Include relevant keywords from the job description",
        "Remove outdated or irrelevant experience",
        "Keep your resume concise and focused",
        "Highlight leadership experiences and initiative",
        "Make sure your contact information is up-to-date",
        "Consider adding a professional summary section",
        "Ensure consistent formatting throughout your resume"
    ]
    
    random.shuffle(generic_suggestions)
    suggestions.extend(generic_suggestions[:max(0, 5 - len(suggestions))])
    
    return suggestions[:5]

def detect_industry(text, job_description):
    """Detect the likely industry from the resume and job description"""
    # Combine resume text and job description for better detection
    combined_text = text.lower() + " " + job_description.lower()
    
    # Count matches for each industry
    industry_scores = {}
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in combined_text:
                score += 1
        
        # Check for job titles specific to this industry
        for title in INDUSTRY_JOB_TITLES.get(industry, []):
            if title.lower() in combined_text:
                score += 2  # Give extra weight to job titles
        
        # Calculate percentage match
        match_percentage = (score / (len(keywords) + len(INDUSTRY_JOB_TITLES.get(industry, [])))) * 100
        industry_scores[industry] = min(100, match_percentage)
    
    # Get the industry with the highest score
    if industry_scores:
        primary_industry = max(industry_scores.items(), key=lambda x: x[1])
        
        # Get secondary industries (with at least 25% match)
        secondary_industries = [(i, s) for i, s in industry_scores.items() 
                            if s >= 25 and i != primary_industry[0]]
        secondary_industries.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "primary": primary_industry,
            "secondary": secondary_industries[:2]  # Top 2 secondary industries
        }
    
    return {"primary": ("General", 0), "secondary": []}

def get_industry_specific_recommendations(industry, resume_text, job_description):
    """Generate industry-specific recommendations"""
    recommendations = []
    
    if industry == "Tech/IT":
        # Check for technical skills
        if not re.search(r'programming|language|framework|technology|stack', resume_text.lower()):
            recommendations.append("Add a dedicated technical skills section listing specific programming languages and technologies")
        
        # Check for projects
        if not re.search(r'project|develop|implement|create', resume_text.lower()):
            recommendations.append("Include specific projects with technical details and your role in implementation")
        
        # Check for GitHub or portfolio
        if not re.search(r'github|portfolio|repository|code', resume_text.lower()):
            recommendations.append("Add a link to your GitHub profile or code portfolio")
    
    elif industry == "Finance":
        # Check for specific financial certifications
        if not re.search(r'cpa|cfa|series 7|series 63|certified', resume_text.lower()):
            recommendations.append("Include relevant financial certifications or licenses")
        
        # Check for quantitative achievements
        if not re.search(r'\$|percent|%|million|billion|increased|decreased|improved', resume_text.lower()):
            recommendations.append("Quantify your financial achievements with specific numbers and percentages")
    
    elif industry == "Healthcare":
        # Check for certifications/licenses
        if not re.search(r'license|certified|certification|registered', resume_text.lower()):
            recommendations.append("Ensure your medical licenses and certifications are clearly listed with expiration dates")
        
        # Check for specific medical skills
        if not re.search(r'patient|care|treatment|diagnosis|procedure', resume_text.lower()):
            recommendations.append("Detail specific patient care skills and medical procedures you're proficient in")
    
    elif industry == "Marketing":
        # Check for campaign metrics
        if not re.search(r'roi|conversion|growth|engagement|click|impression|lead', resume_text.lower()):
            recommendations.append("Include specific marketing metrics and KPIs from your campaigns (ROI, conversion rates, etc.)")
        
        # Check for tools
        if not re.search(r'analytics|social media|seo|sem|content|crm|marketing automation', resume_text.lower()):
            recommendations.append("List marketing tools and platforms you're proficient with (Google Analytics, HubSpot, etc.)")
    
    # If no industry-specific recommendations were generated
    if not recommendations:
        recommendations.append(f"Highlight specific {industry} achievements with measurable results")
        recommendations.append(f"Add industry-specific keywords relevant to {industry} roles")
    
    return recommendations[:3]  # Return top 3 recommendations

def calculate_resume_freshness(resume_text):
    """Calculate how recently the resume has been updated"""
    # Look for recent dates (current year or previous year)
    current_year = datetime.now().year
    
    # Check for current year
    if re.search(f'{current_year}', resume_text):
        return 100  # Very fresh
    
    # Check for previous year
    if re.search(f'{current_year-1}', resume_text):
        return 80  # Fairly fresh
    
    # Check for last 2-3 years
    if re.search(f'{current_year-2}|{current_year-3}', resume_text):
        return 60  # Moderately fresh
    
    # Check for 4-5 years ago
    if re.search(f'{current_year-4}|{current_year-5}', resume_text):
        return 40  # Getting outdated
    
    # If we find dates but they're all older than 5 years
    date_pattern = r'20\d{2}|19\d{2}'
    if re.search(date_pattern, resume_text):
        return 20  # Outdated
    
    # If no dates are found at all
    return 0  # Cannot determine

def extract_job_title(text):
    """Extract potential job titles from text"""
    job_titles = []
    
    # Flatten all industry job titles into one list
    all_job_titles = [title for titles in INDUSTRY_JOB_TITLES.values() for title in titles]
    
    # Look for matches of job titles in the text
    for title in all_job_titles:
        if re.search(r'\b' + re.escape(title) + r'\b', text.lower()):
            job_titles.append(title.title())
    
    # If we found job titles, return them
    if job_titles:
        return job_titles[:3]  # Return top 3 found
    
    # Otherwise, look for patterns that might indicate job titles
    lines = text.split('\n')
    for line in lines:
        # Look for standalone short lines that might be titles
        if 3 <= len(line.strip().split()) <= 5 and any(word[0].isupper() for word in line.strip().split()):
            if not re.search(r'@|http|www|email|phone|address', line.lower()):
                job_titles.append(line.strip())
                if len(job_titles) >= 3:
                    break
    
    return job_titles

def calculate_job_title_match(resume_titles, jd_titles):
    """Calculate match between resume job titles and job description titles"""
    if not resume_titles or not jd_titles:
        return 0
    
    # Create sets of words from titles for fuzzy matching
    resume_words = set()
    for title in resume_titles:
        resume_words.update(title.lower().split())
    
    jd_words = set()
    for title in jd_titles:
        jd_words.update(title.lower().split())
    
    # Find matching words
    matching_words = resume_words.intersection(jd_words)
    
    # Calculate score based on percentage of matching words
    if not jd_words:
        return 0
    
    score = (len(matching_words) / len(jd_words)) * 100
    return min(100, score)

def analyze_resume_format(resume_text):
    """Analyze resume format quality without NLTK"""
    format_issues = []
    suggestions = []
    
    # Check for bullet points
    bullet_count = len(re.findall(r'•|▪|▫|◦|⦿|⚫|⚬|-|\*|\+|o', resume_text))
    if bullet_count < 5:
        format_issues.append("Limited use of bullet points")
        suggestions.append("Use bullet points to highlight achievements and responsibilities")
    
    # Check for section headings (all caps or title case followed by colon or newline)
    heading_pattern = r'\n[A-Z][A-Za-z\s]+:|\n[A-Z]{2,}[\s\n]'
    headings = re.findall(heading_pattern, resume_text)
    if len(headings) < 3:
        format_issues.append("Few clear section headings")
        suggestions.append("Use clear section headings (e.g., EXPERIENCE, EDUCATION, SKILLS)")
    
    # Check for consistent date formatting
    date_formats = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b', resume_text)
    if date_formats:
        date_format_types = set()
        for date in date_formats:
            if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', date):
                date_format_types.add("MM/DD/YYYY")
            elif re.match(r'\d{1,2}-\d{1,2}-\d{2,4}', date):
                date_format_types.add("MM-DD-YYYY")
            else:
                date_format_types.add("Month YYYY")
                
        if len(date_format_types) > 1:
            format_issues.append("Inconsistent date formatting")
            suggestions.append("Use consistent date format throughout your resume")
    
    # Check for very long paragraphs
    paragraphs = re.split(r'\n\s*\n', resume_text)
    long_paragraphs = [p for p in paragraphs if len(p.split()) > 40]
    if long_paragraphs:
        format_issues.append("Contains lengthy paragraphs")
        suggestions.append("Break down long paragraphs into bullet points for better readability")
    
    # Check for consistent tense usage in experience descriptions
    present_tense_verbs = re.findall(r'\b(?:manage|develop|lead|create|implement|design|coordinate|oversee|maintain|analyze)\s', resume_text.lower())
    past_tense_verbs = re.findall(r'\b(?:managed|developed|led|created|implemented|designed|coordinated|oversaw|maintained|analyzed)\s', resume_text.lower())
    
    if present_tense_verbs and past_tense_verbs:
        # Check if they're used inconsistently in the same sections
        experiences = extract_experience(resume_text)
        mixed_tense = False
        for exp in experiences:
            present_count = sum(1 for v in present_tense_verbs if v in exp.lower())
            past_count = sum(1 for v in past_tense_verbs if v in exp.lower())
            if present_count > 0 and past_count > 0:
                mixed_tense = True
                break
                
        if mixed_tense:
            format_issues.append("Mixed verb tenses within same experience")
            suggestions.append("Use past tense for previous roles and present tense for current roles")
    
    # Check for contact information
    contact_info = re.search(r'(?:\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)|(?:\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)|(?:linkedin\.com/in/[a-zA-Z0-9_-]+)', resume_text)
    if not contact_info:
        format_issues.append("Missing or hard-to-find contact information")
        suggestions.append("Add clear contact information at the top of your resume")
    
    # Calculate format quality score
    max_issues = 6  # Maximum number of issues we check for
    format_score = 100 - (len(format_issues) / max_issues * 100)
    
    return {
        "format_issues": format_issues,
        "format_suggestions": suggestions,
        "format_score": format_score
    }

def detect_job_level(job_description):
    """Detect the job level based on the job description"""
    # Look for specific job level indicators
    entry_level_keywords = ["entry", "junior", "associate", "intern", "trainee", "assistant", "beginner", "0-2 years", "1-2 years"]
    mid_level_keywords = ["mid", "intermediate", "experienced", "3-5 years", "4-6 years"]
    senior_level_keywords = ["senior", "lead", "manager", "head", "principal", "6+ years", "7+ years", "8+ years"]
    executive_keywords = ["director", "executive", "chief", "vp", "president", "ceo", "cto", "cfo", "coo", "10+ years"]
    
    # Convert to lowercase for matching
    job_desc_lower = job_description.lower()
    
    # Count matches for each level
    entry_count = sum(1 for keyword in entry_level_keywords if re.search(r'\b' + re.escape(keyword) + r'\b', job_desc_lower))
    mid_count = sum(1 for keyword in mid_level_keywords if re.search(r'\b' + re.escape(keyword) + r'\b', job_desc_lower))
    senior_count = sum(1 for keyword in senior_level_keywords if re.search(r'\b' + re.escape(keyword) + r'\b', job_desc_lower))
    executive_count = sum(1 for keyword in executive_keywords if re.search(r'\b' + re.escape(keyword) + r'\b', job_desc_lower))
    
    # Determine the most likely job level
    levels = [
        ("Entry Level", entry_count),
        ("Mid Level", mid_count),
        ("Senior Level", senior_count),
        ("Executive", executive_count)
    ]
    
    # Sort by count in descending order
    levels.sort(key=lambda x: x[1], reverse=True)
    
    # If no clear indicators, default to Mid Level
    if levels[0][1] == 0:
        return "Mid Level"
    
    return levels[0][0]

def detect_cliches(resume_text):
    """Detect clichés and overused buzzwords in resume"""
    found_cliches = []
    
    for cliche in RESUME_CLICHES:
        if re.search(r'\b' + re.escape(cliche) + r'\b', resume_text.lower()):
            found_cliches.append(cliche)
    
    return found_cliches

def calculate_readability(text):
    """Calculate readability metrics for text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences and words
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    
    # Calculate number of syllables (simplified estimation)
    syllable_count = 0
    for word in words:
        word = word.lower()
        if len(word) <= 3:
            syllable_count += 1
            continue
            
        # Count vowel groups as syllables
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
            
        # Ensure at least one syllable per word
        syllable_count += max(1, count)
    
    # Calculate metrics
    try:
        # Flesch Reading Ease
        if len(sentences) > 0 and len(words) > 0:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllable_count / len(words)
            reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Adjust to 0-100 scale
            reading_ease = max(0, min(100, reading_ease))
            
            # Interpret score
            if reading_ease >= 90:
                interpretation = "Very Easy"
            elif reading_ease >= 80:
                interpretation = "Easy"
            elif reading_ease >= 70:
                interpretation = "Fairly Easy"
            elif reading_ease >= 60:
                interpretation = "Standard"
            elif reading_ease >= 50:
                interpretation = "Fairly Difficult"
            elif reading_ease >= 30:
                interpretation = "Difficult"
            else:
                interpretation = "Very Difficult"
                
            return {
                "reading_ease": reading_ease,
                "interpretation": interpretation,
                "avg_words_per_sentence": avg_sentence_length,
                "avg_syllables_per_word": avg_syllables_per_word
            }
    except:
        pass
        
    # Return default if calculation fails
    return {
        "reading_ease": 0,
        "interpretation": "Unknown",
        "avg_words_per_sentence": 0,
        "avg_syllables_per_word": 0
    }

def detect_employment_gaps(resume_text):
    """Detect gaps in employment history"""
    # Extract potential dates in format: YYYY or MM/YYYY or Month YYYY
    date_pattern = r'\b((?:19|20)\d{2})\b|(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:19|20)\d{2}'
    
    # Extract all dates
    dates = re.findall(date_pattern, resume_text)
    
    # If we have fewer than 2 dates, we can't detect gaps
    if len(dates) < 2:
        return []
    
    # Convert all dates to years for simpler comparison
    years = []
    for date in dates:
        # If it's just a year
        if re.match(r'^(?:19|20)\d{2}$', date):
            years.append(int(date))
        # Otherwise, extract the year part
        else:
            year_match = re.search(r'((?:19|20)\d{2})', date)
            if year_match:
                years.append(int(year_match.group(1)))
    
    # Sort years and remove duplicates
    years = sorted(list(set(years)))
    
    # Find gaps of more than 1 year
    gaps = []
    for i in range(len(years) - 1):
        gap = years[i+1] - years[i]
        if gap > 1:
            gaps.append((years[i], years[i+1], gap))
    
    return gaps

def compare_to_benchmarks(analysis, industry):
    """Compare resume to industry benchmarks"""
    benchmark = INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["General"])
    
    comparisons = []
    
    # Compare skill count
    skills_count = len(analysis["skills"])
    if skills_count < benchmark["skills_count"]:
        comparisons.append(f"Your resume has {skills_count} skills listed, while the industry average is {benchmark['skills_count']}.")
    
    # Extract years of experience from resume
    experience_years = 0
    experience_match = re.search(r'(\d+)[+]?\s+years?', " ".join(analysis["experience"]).lower())
    if experience_match:
        experience_years = int(experience_match.group(1))
        
        if experience_years < benchmark["experience_years"]:
            comparisons.append(f"Your resume shows {experience_years} years of experience, while the industry average is {benchmark['experience_years']}.")
    
    # Calculate bullet points per job
    bullet_count = 0
    for exp in analysis["experience"]:
        bullet_count += len(re.findall(r'•|▪|▫|◦|⦿|⚫|⚬|-|\*|\+|o', exp))
    
    avg_bullets = bullet_count / max(1, len(analysis["experience"]))
    if avg_bullets < benchmark["bullet_points_per_job"]:
        comparisons.append(f"Your resume has approximately {avg_bullets:.1f} bullet points per job, while the industry standard is {benchmark['bullet_points_per_job']}.")
    
    # Compare certifications count
    cert_count = len(analysis["certifications"])
    if cert_count < benchmark["certifications_count"]:
        comparisons.append(f"Your resume has {cert_count} certifications, while the industry average is {benchmark['certifications_count']}.")
    
    return comparisons

def analyze_resume(resume_text, job_description):
    """Analyze resume without any external API"""
    try:
        # Detect job level from job description
        job_level = detect_job_level(job_description)
        
        # Extract information from resume
        skills = extract_skills(resume_text, job_description)
        education = extract_education(resume_text)
        experience = extract_experience(resume_text)
        certifications = extract_certifications(resume_text)
        
        # Analyze strengths and weaknesses
        strengths, weaknesses = analyze_strengths_weaknesses(skills, job_description)
        
        # Calculate matching score with detailed breakdown
        matching_score, score_breakdown = calculate_matching_score(resume_text, job_description, job_level)
        
        # Generate improvement suggestions
        suggestions = generate_suggestions(strengths, weaknesses, skills, job_description)
        
        # New features
        keyword_density = analyze_keyword_density(resume_text)
        ats_issues = check_ats_compatibility(resume_text)
        sections, completeness_score = analyze_section_completeness(resume_text)
        
        # Advanced features
        detected_industry = detect_industry(resume_text, job_description)
        industry_specific_recommendations = []
        if detected_industry["primary"][0] != "General" and detected_industry["primary"][1] > 30:
            industry_specific_recommendations = get_industry_specific_recommendations(
                detected_industry["primary"][0], resume_text, job_description
            )
        
        resume_freshness = calculate_resume_freshness(resume_text)
        resume_job_titles = extract_job_title(resume_text)
        jd_job_titles = extract_job_title(job_description)
        job_title_match = calculate_job_title_match(resume_job_titles, jd_job_titles)
        
        # Format analysis
        format_analysis = analyze_resume_format(resume_text)
        
        # More advanced features
        cliches = detect_cliches(resume_text)
        readability = calculate_readability(resume_text)
        employment_gaps = detect_employment_gaps(resume_text)
        
        # Compare to industry benchmarks
        industry_comparisons = []
        if detected_industry["primary"][0] != "General" and detected_industry["primary"][1] > 30:
            industry_comparisons = compare_to_benchmarks(
                {"skills": skills, "experience": experience, "certifications": certifications},
                detected_industry["primary"][0]
            )
        
        # Create analysis dictionary
        analysis = {
            "skills": skills,
            "education": education,
            "experience": experience,
            "certifications": certifications,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "matching_score": matching_score,
            "score_breakdown": score_breakdown,
            "suggestions": suggestions,
            "keyword_density": keyword_density,
            "ats_issues": ats_issues,
            "section_completeness": sections,
            "completeness_score": completeness_score,
            # Advanced analysis
            "detected_industry": detected_industry,
            "industry_specific_recommendations": industry_specific_recommendations,
            "resume_freshness": resume_freshness,
            "resume_job_titles": resume_job_titles,
            "jd_job_titles": jd_job_titles,
            "job_title_match": job_title_match,
            # Format analysis
            "format_issues": format_analysis["format_issues"],
            "format_suggestions": format_analysis["format_suggestions"],
            "format_score": format_analysis["format_score"],
            # Super advanced features
            "cliches": cliches,
            "readability": readability,
            "employment_gaps": employment_gaps,
            "job_level": job_level,
            "industry_comparisons": industry_comparisons
        }
        
        return analysis
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def display_score_gauge(score):
    """Display a gauge chart for the resume score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Resume Match Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_resume_quality_radar(analysis):
    """Create a radar chart showing overall resume quality across dimensions"""
    # Calculate various quality dimensions
    dimensions = {
        "ATS Compatibility": 100 if "No obvious ATS compatibility issues detected" in analysis["ats_issues"] else max(0, 100 - 25 * len(analysis["ats_issues"])),
        "Content Completeness": analysis["completeness_score"],
        "Skills Match": analysis["score_breakdown"]["Technical Skills"] * 4,  # Scale to 100
        "Experience Match": analysis["score_breakdown"]["Experience"] * (100/15),  # Scale to 100
        "Keyword Optimization": analysis["score_breakdown"]["Keyword Match"] * (100/40),  # Scale to 100
        "Freshness": analysis["resume_freshness"]
    }
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(dimensions.values()),
        theta=list(dimensions.keys()),
        fill='toself',
        name='Resume Quality'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Resume Quality Assessment"
    )
    
    return fig

def create_job_fit_radar(analysis):
    """Create a radar chart showing job fit across dimensions"""
    # Gather the relevant dimensions
    dimensions = {
        "Skills Match": analysis["score_breakdown"]["Technical Skills"] * 4,  # Scale to 100
        "Education": analysis["score_breakdown"]["Education"] * 10,  # Scale to 100
        "Experience": analysis["score_breakdown"]["Experience"] * (100/15),  # Scale to 100
        "Certifications": analysis["score_breakdown"]["Certifications"] * 10,  # Scale to 100
        "Job Title Relevance": analysis["job_title_match"],
        "Industry Relevance": analysis["detected_industry"]["primary"][1]
    }
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(dimensions.values()),
        theta=list(dimensions.keys()),
        fill='toself',
        name='Job Fit'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Job Fit Analysis"
    )
    
    return fig

def main():
    st.title("📄 Advanced Resume Analyzer")
    st.markdown("""
    This tool helps you analyze your resume against a job description.
    Upload your resume or paste the text, and provide the job description to get a detailed analysis.
    """)
    
    # Input methods for resume
    resume_input_method = st.radio(
        "Choose how to input your resume:",
        ["Upload PDF", "Paste Text"]
    )
    
    resume_text = ""
    if resume_input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
        if uploaded_file:
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                st.success("PDF successfully processed!")
            except Exception as e:
                st.error(f"Error extracting text from PDF: {str(e)}")
    else:
        resume_text = st.text_area("Paste your resume text here:", height=200)
    
    # Job description input
    job_description = st.text_area("Paste the job description here:", height=200)
    
    # Analyze button
    if st.button("Analyze Resume"):
        if not resume_text or not job_description:
            st.warning("Please provide both resume and job description.")
            return
        
        with st.spinner("Analyzing your resume..."):
            analysis = analyze_resume(resume_text, job_description)
            
            if analysis:
                # Job level detection
                st.subheader("📊 Job Level & Scoring")
                st.info(f"Detected Job Level: **{analysis['job_level']}**")
                st.caption("Scoring weights are customized based on the detected job level.")
                
                # Display score with gauge
                score_gauge = display_score_gauge(analysis['matching_score'])
                st.plotly_chart(score_gauge)
                
                # Display quality radar charts
                col1, col2 = st.columns(2)
                
                with col1:
                    resume_quality_radar = create_resume_quality_radar(analysis)
                    st.plotly_chart(resume_quality_radar)
                
                with col2:
                    job_fit_radar = create_job_fit_radar(analysis)
                    st.plotly_chart(job_fit_radar)
                
                # Industry detection
                if analysis['detected_industry']["primary"][1] > 0:
                    st.subheader("🏢 Industry Analysis")
                    primary_industry, primary_score = analysis['detected_industry']["primary"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Primary Industry", primary_industry, f"{primary_score:.1f}% match")
                        
                        if analysis['resume_job_titles']:
                            st.subheader("📋 Detected Job Titles")
                            for title in analysis['resume_job_titles']:
                                st.markdown(f"- {title}")
                    
                    with col2:
                        if analysis['detected_industry']["secondary"]:
                            st.subheader("Related Industries")
                            for industry, score in analysis['detected_industry']["secondary"]:
                                st.metric(industry, f"{score:.1f}% match")
                        
                        if analysis['job_title_match'] > 0:
                            st.metric("Job Title Relevance", f"{analysis['job_title_match']:.1f}%")
                            
                    # Industry-specific recommendations
                    if analysis['industry_specific_recommendations']:
                        st.subheader(f"💡 {primary_industry}-Specific Recommendations")
                        for rec in analysis['industry_specific_recommendations']:
                            st.markdown(f"- {rec}")
                
                # Resume freshness
                if analysis['resume_freshness'] > 0:
                    freshness = analysis['resume_freshness']
                    freshness_color = "normal"  # Default to normal
                    if freshness >= 80:
                        freshness_status = "Up-to-date"
                        freshness_color = "normal"  # Green equivalent
                    elif freshness >= 60:
                        freshness_status = "Moderately current"
                        freshness_color = "normal"  # Blue equivalent
                    elif freshness >= 40:
                        freshness_status = "Needs updating"
                        freshness_color = "inverse"  # Orange equivalent
                    else:
                        freshness_status = "Outdated"
                        freshness_color = "inverse"  # Red equivalent
                    
                    st.metric("Resume Recency", freshness_status, delta=f"{freshness}%", delta_color=freshness_color)
                
                # Organize analysis in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Resume Content", "Strengths & Gaps", 
                    "ATS Compatibility", "Format Analysis", 
                    "Advanced Insights"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Skills")
                        for skill in analysis['skills']:
                            st.markdown(f"- {skill}")
                        
                        st.subheader("🎓 Education")
                        for edu in analysis['education']:
                            st.markdown(f"- {edu}")
                    
                    with col2:                        
                        st.subheader("💼 Work Experience")
                        for exp in analysis['experience']:
                            st.markdown(f"- {exp}")
                            
                        st.subheader("📜 Certifications")
                        for cert in analysis['certifications']:
                            st.markdown(f"- {cert}")
                    
                    # Keyword density visualization
                    st.subheader("Keyword Density")
                    if analysis['keyword_density']:
                        keywords, counts = zip(*analysis['keyword_density'])
                        keyword_df = pd.DataFrame({
                            'Keyword': keywords,
                            'Count': counts
                        })
                        fig = px.bar(keyword_df, x='Keyword', y='Count',
                                    title='Top 10 Keywords in Your Resume')
                        st.plotly_chart(fig)
                    else:
                        st.info("No significant keywords found")
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✨ Strengths")
                        for strength in analysis['strengths']:
                            st.markdown(f"- {strength}")
                    
                    with col2:
                        st.subheader("⚠️ Gaps to Address")
                        for weakness in analysis['weaknesses']:
                            st.markdown(f"- {weakness}")
                    
                    # Section completeness visualization
                    st.subheader("Resume Section Completeness")
                    sections = analysis['section_completeness']
                    
                    # Convert to dataframe for visualization
                    section_df = pd.DataFrame({
                        'Section': list(sections.keys()),
                        'Status': ["Present" if val >= 0.5 else "Missing" for val in sections.values()]
                    })
                    
                    fig = px.bar(section_df, x='Section', y=[1]*len(section_df), color='Status',
                                color_discrete_map={'Present': 'green', 'Missing': 'red'},
                                title=f'Section Completeness: {round(analysis["completeness_score"])}%')
                    fig.update_layout(yaxis_title="")
                    fig.update_yaxes(showticklabels=False)
                    st.plotly_chart(fig)
                
                with tab3:
                    st.subheader("ATS Compatibility Check")
                    for issue in analysis['ats_issues']:
                        icon = "❌" if "No" not in issue else "✅"
                        st.markdown(f"{icon} {issue}")
                    
                    # Add information about ATS
                    st.info("""
                    **What is ATS?** Applicant Tracking Systems (ATS) are software used by employers to scan resumes 
                    for keywords and filter candidates. Making your resume ATS-friendly increases its chances of being seen by human recruiters.
                    """)
                
                with tab4:
                    st.subheader("📝 Resume Format Analysis")
                    
                    format_score = analysis['format_score']
                    st.metric("Format Quality Score", f"{format_score:.1f}%")
                    
                    if analysis['format_issues']:
                        st.subheader("Format Issues")
                        for issue in analysis['format_issues']:
                            st.markdown(f"❌ {issue}")
                    else:
                        st.success("No format issues detected. Your resume has a good structure!")
                    
                    if analysis['format_suggestions']:
                        st.subheader("Format Improvement Suggestions")
                        for suggestion in analysis['format_suggestions']:
                            st.markdown(f"✏️ {suggestion}")
                
                with tab5:
                    st.subheader("🔍 Advanced Resume Insights")
                    
                    # Readability metrics
                    if analysis['readability']["reading_ease"] > 0:
                        st.subheader("Readability Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            readability_score = analysis['readability']["reading_ease"]
                            readability_interpretation = analysis['readability']["interpretation"]
                            st.metric("Readability Score", f"{readability_score:.1f}/100")
                            st.info(f"Interpretation: **{readability_interpretation}**")
                            
                        with col2:
                            st.metric("Avg. Words Per Sentence", f"{analysis['readability']['avg_words_per_sentence']:.1f}")
                            st.metric("Avg. Syllables Per Word", f"{analysis['readability']['avg_syllables_per_word']:.2f}")
                    
                    # Cliché detection
                    if analysis['cliches']:
                        st.subheader("Detected Clichés and Buzzwords")
                        st.warning("These overused phrases may weaken your resume. Consider replacing them with specific achievements.")
                        for cliche in analysis['cliches']:
                            st.markdown(f"- \"{cliche}\"")
                    
                    # Employment gaps
                    if analysis['employment_gaps']:
                        st.subheader("Potential Employment Gaps")
                        st.info("Consider addressing these gaps in your resume or cover letter.")
                        for start_year, end_year, gap_years in analysis['employment_gaps']:
                            st.markdown(f"- **{gap_years} year gap** detected between {start_year} and {end_year}")
                    
                    # Industry benchmark comparison
                    if analysis['industry_comparisons']:
                        st.subheader("Industry Benchmark Comparison")
                        st.info(f"How your resume compares to others in the {analysis['detected_industry']['primary'][0]} industry:")
                        for comparison in analysis['industry_comparisons']:
                            st.markdown(f"- {comparison}")

if __name__ == "__main__":
    main() 