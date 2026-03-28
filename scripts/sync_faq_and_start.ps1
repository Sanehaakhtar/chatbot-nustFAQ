param(
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"

& "d:/chatbot/.venv/Scripts/python.exe" ingest.py --url "https://nust.edu.pk/faqs/" --url "https://nust.edu.pk/faq-category/ug-admission/" --url "https://nust.edu.pk/faq-category/mbbs-admissions-faqs/" --url "https://nust.edu.pk/faq-category/bshnd-admissions-faqs/"

if ($LASTEXITCODE -ne 0) {
    throw "Ingestion failed with exit code $LASTEXITCODE"
}

& "d:/chatbot/scripts/start_server.ps1" -Port $Port
