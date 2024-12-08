# QA Issues Analysis Tool

A tool for analyzing and managing QA testing issues for chatbot responses. Uses LLM to help spot patterns and suggest merges for related issues.

## Overview

Designed to accelerate the interpretation of QA results, making it faster and more efficient to implement effective changes. The tool analyzes test data using GPT-4o to find patterns, keeps humans in the loop in determining whether to fully or selectively combine suggested issue groups, and generates insights into cause and suggested improvements of detected issues.

### Main Features

- Automated issue analysis using LLM
- Smart merging of related issues
- PDF reports with insights and recommendations
- Merge history management

## Setup

### Requirements

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repo:
```bash
git clone https://github.com/your-username/SUDCare-issue-analysis.git
cd SUDCare-issue-analysis
```

2. Set up Python environment:
```bash
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # Unix/MacOS
```

3. Install packages:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key in the terminal using the `set` command:
```bash
set OPENAI_API_KEY=your_api_key_here
```

## Using the Tool

1. Start it up:
```bash
streamlit run app.py
```

2. Open `http://localhost:8501` in your browser

### Basic Workflow

1. **Upload Data**
   - Upload your QA issues CSV in the format that exists in the QA workbook.
   - Make sure it has all required columns

2. **Analyze Issues**
   - Go to "Merge Analysis"
   - Click "Analyze Issues"
   - Review suggested merges

3. **Merge Issues**
   - Check merge suggestions and confidence scores
   - Apply the ones that make sense
   - Download the updated CSV

4. **Generate Report**
   - Get a PDF report with analysis and recommendations

### Required CSV Columns

- Issue ID
- Result ID
- Test Case IDs
- Input Prompt
- Ground Truth
- Generated Response
- Linked Theme
- Linked Standard
- Session IDs
- Version Tested
- Run Date
- Failure Rationale
- Final Weighted Score (1-3)

## Features

### Overview Tab
- See total issues
- Track active vs merged issues
- View distribution charts
- Check standards analysis

### Merge Analysis Tab
- Get merge suggestions from GPT-4o
- See confidence scores
- Preview merges
- Pick which issues to merge
- Manage merge history

### Merge History Tab
- View all past merges
- Check merge details
- Track when merges happened

## Cache Management

We keep track of merge history to make things faster:
- Turn cache on/off as needed
- Clear it when you want
- Auto-updates as you work

## Report Details

Our PDF reports include:
- Quick summary
- Pattern analysis
- What to fix first
- Charts and graphs
- Step-by-step improvement plan

We focus on active issues to keep things relevant:
- Look at merged groups (related issues)
- Check individual unmerged issues
- Skip already-merged individual issues to avoid duplicates

## Recent Fixes

### Bug Fixes
1. **Unmerged Issues Counter**
   - Fixed issue counting
   - Added live updates
   - Added validation

2. **Active Issues Counter**
   - Fixed the math
   - Better status checking
   - Added tests
