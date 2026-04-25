@echo off
SET "PROJECT_DIR=C:\Users\rohit\Documents\Research Papers\ResearchPaper-2_JudgeSense\Project\judgeSense"
SET "BASH_PATH=C:\Program Files\Git\bin\bash.exe"

REM Open 12 Git Bash windows in parallel

start "Tab 1" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task factuality --runs 1; read -p 'Press enter...'"

start "Tab 2" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task factuality --runs 1; read -p 'Press enter...'"

start "Tab 3" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task factuality --runs 1; read -p 'Press enter...'"

start "Tab 4" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task coherence --runs 1; read -p 'Press enter...'"

start "Tab 5" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task coherence --runs 1; read -p 'Press enter...'"

start "Tab 6" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task coherence --runs 1; read -p 'Press enter...'"

start "Tab 7" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task relevance --runs 1; read -p 'Press enter...'"

start "Tab 8" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task relevance --runs 1; read -p 'Press enter...'"

start "Tab 9" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task relevance --runs 1; read -p 'Press enter...'"

start "Tab 10" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task preference --runs 1; read -p 'Press enter...'"

start "Tab 11" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task preference --runs 1; read -p 'Press enter...'"

start "Tab 12" "%BASH_PATH%" -c "cd \"$PROJECT_DIR\" && python src/evaluate.py --model gemini-flash --task preference --runs 1; read -p 'Press enter...'"