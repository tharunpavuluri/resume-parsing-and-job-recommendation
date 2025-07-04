import spacy
from spacy.matcher import PhraseMatcher
import docx
from PyPDF2 import PdfReader
from io import BytesIO

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Predefined normalized skill set (all lowercase)
SKILLS = {
    # Programming Languages
    "c", "c++", "python", "java", "javascript", "typescript", "go", "ruby", "swift", "kotlin",
    "rust", "php", "perl", "r", "dart", "matlab", "scala", "haskell", "lua", "shell", "bash",
    "powershell", "groovy", "elixir", "f#", "c#", "objective-c",

    # Web Development
    "html", "css", "sass", "less", "bootstrap", "tailwind", "react", "angular", "vue", "svelte",
    "next.js", "nuxt.js", "express.js", "django", "flask", "fastapi", "ruby on rails", "laravel",
    "spring boot", "asp.net", "graphql", "websocket", "redux",

    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra", "couchdb", "dynamodb",
    "neo4j", "elasticsearch", "firebase", "oracle", "mariadb", "snowflake",

    # Cloud & DevOps
    "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "travis ci", "circleci", "gitlab ci/cd", "github actions", "helm", "prometheus", "grafana",
    "splunk", "datadog", "new relic", "cloudformation",

    # Machine Learning & Data Science
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "nlp", "computer vision", "opencv", "transformers", "hugging face", "deep learning",
    "reinforcement learning", "xgboost", "lightgbm", "catboost", "bigquery", "data engineering",

    # Mobile Development
    "android", "ios", "react native", "flutter", "swiftui", "jetpack compose", "xamarin",
    "cordova", "ionic",

    # Other Technical Skills
    "git", "linux", "unix", "bash scripting", "vim", "emacs", "networking", "cybersecurity", "ethical hacking",
    "penetration testing", "blockchain", "smart contracts", "solidity", "web3.js", "hardhat", "truffle",
    "game development", "unity", "unreal engine", "godot", "microservices", "event-driven architecture",
    "serverless", "message queues", "rabbitmq", "kafka"
}


def parse_resume(file):
    """
    Entry point to parse a resume file (.pdf or .docx).
    Returns: List of extracted skills.
    """
    file_extension = file.filename.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file)
        else:
            raise ValueError("Unsupported file type")

        return extract_skills(text)

    except Exception as e:
        print(f"Error parsing file: {e}")
        return []


def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    try:
        file_stream = BytesIO(file.read())
        pdf_reader = PdfReader(file_stream)
        return ''.join(page.extract_text() or '' for page in pdf_reader.pages)
    except Exception as e:
        print(f"Failed to extract PDF: {e}")
        return ""


def extract_text_from_docx(file):
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        doc = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        print(f"Failed to extract DOCX: {e}")
        return ""


def extract_skills(text):
    """
    Extracts matching skills from text using spaCy PhraseMatcher.
    """
    text = text.lower()
    doc = nlp(text)

    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in SKILLS]
    matcher.add("SKILLS", patterns)

    matches = matcher(doc)
    found_skills = {doc[start:end].text for _, start, end in matches}

    return sorted(found_skills)


# Example usage (Flask or Django handler might use something like):
# uploaded_file = request.files['resume']
# skills = parse_resume(uploaded_file)
