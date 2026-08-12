# Final GitHub, Streamlit, and Moodle handoff

These are intentionally human steps: they require the student's authenticated
GitHub, Streamlit and Moodle accounts.

## 1. Review and local check

1. Unzip `z5646230_projectB.zip` and open that folder.
2. Read `docs/STUDENT_REVIEW_GUIDE.md` and approve/rewrite the report in your own
   voice. Do not hand in a statement you cannot explain.
3. If you want to reproduce from raw data, create a virtual environment, install
   `requirements.txt` plus `requirements-dev.txt`, then run
   `python scripts/run_part_b.py`.
4. Run `python -m pytest -q`, launch `streamlit run streamlit_app.py`, and run
   `python scripts/check_handin.py`.

## 2. Create the separate GitHub repository

Create a new empty GitHub repository named `z5646230_projectB`. Keep it private
while checking it. Do not add a GitHub README or `.gitignore`, because both are
already in this folder. From this folder run:

```text
git init
git branch -M main
git add .
git commit -m "Complete FINS5545 Project B"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/z5646230_projectB.git
git push -u origin main
```

On GitHub, confirm that `streamlit_app.py`, `requirements.txt`, the full
`results/` folder, `report/report.pdf`, and the `ai/` folder are visible. Confirm
that `.venv`, raw Parquet/CSV data outside `results/`, and secrets are absent.

## 3. Deploy Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns the
   repository.
2. Choose **Create app** / **Deploy a public app from GitHub**.
3. Select repository `z5646230_projectB`, branch `main`, and main file
   `streamlit_app.py`. Use the available Python 3.13 runtime.
4. Deploy. The app needs no secrets and performs no NLP or optimisation at
   runtime; it reads committed files from `results/`.
5. Test all five tabs. In particular, change fund, holdings, allocation sliders,
   sector and cost scenario.

## 4. Make public and verify logged out

At hand-in, change the GitHub repository visibility to **Public** and ensure the
Streamlit app is public. Open both URLs in an incognito/private browser window
while logged out. A marker must be able to open the repository and app without a
request-access screen.

## 5. Submit

Upload `z5646230_projectB.zip` and `report/report.pdf` where required in Moodle,
then paste the public GitHub repository URL and live Streamlit URL into the
submission fields. Reopen the Moodle confirmation page and verify all four
items are recorded.
