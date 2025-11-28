from pathlib import Path

# Project root = folder where this script lives
project_root = Path(__file__).resolve().parent

# 1) Create .streamlit folder if it doesn't exist
secrets_dir = project_root / ".streamlit"
secrets_dir.mkdir(exist_ok=True)

# 2) Create secrets.toml file with the API key
secrets_file = secrets_dir / "secrets.toml"
content = 'NEWSAPI_KEY = "4e97accc9e0c407382695a94d94dfe52"\n'
secrets_file.write_text(content, encoding="utf-8")

print(f"Created {secrets_file}")
