# QA Issues Analysis Tool

A streamlined tool for analyzing and managing QA testing issues for chatbot responses, powered by GPT-4o for intelligent issue analysis and merging.

## Overview

The QA Issues Analysis Tool is designed to automate and enhance the quality assurance process for chatbot responses. It processes QA testing data through advanced LLM analysis to identify patterns, merge related issues, and generate actionable insights for development teams.

### Key Features

- 📊 **Automated Issue Analysis**: Uses GPT-4o to analyze QA issues and identify patterns
- 🔄 **Smart Issue Merging**: Automatically suggests and handles merging of related issues
- 📈 **Comprehensive Reporting**: Generates detailed PDF reports with insights and recommendations
- 💾 **Cache Management**: Efficient handling of merge history with options to use or clear cache
- 📱 **Modern UI**: Clean, intuitive interface built with Streamlit

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/SUDCare-issue-analysis.git
cd SUDCare-issue-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On Unix or MacOS
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

### Workflow

1. **Data Input**
   - Upload your QA issues CSV file through the web interface
   - File must include required columns (Issue ID, Result ID, etc.)

2. **Issue Analysis**
   - Navigate to the "Merge Analysis" tab
   - Click "Analyze Issues for Merging"
   - Review suggested merge groups based on LLM analysis

3. **Issue Merging**
   - Review each merge suggestion with confidence scores
   - Apply merges as appropriate
   - Download updated CSV with merged issues

4. **Report Generation**
   - Generate comprehensive PDF reports
   - View analysis of patterns and trends
   - Get actionable recommendations

### Input Data Structure

Required CSV columns:
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

### 1. Overview Tab
- Total issues count
- Active vs. merged issues statistics
- Distribution visualizations
- Standards analysis

### 2. Merge Analysis Tab
- LLM-powered merge suggestions
- Confidence scoring
- Detailed merge previews
- Cache management options

### 3. Merge History Tab
- Complete merge audit trail
- Detailed merge records
- Timestamp tracking

## Cache Management

The tool maintains a cache of merge history for efficiency:
- Toggle cache usage with checkbox
- Clear cache with dedicated button
- Automatic cache updates

## Report Generation

Generated PDF reports include:
- Executive summary
- Issue patterns analysis
- Priority recommendations
- Visual data representations
- Improvement roadmap

## Development

### Project Structure
```
SUDCare-issue-analysis/
├── app.py              # Main Streamlit application
├── llm_utils.py        # LLM integration utilities
├── merge_utils.py      # Merge operation handling
├── analysis_utils.py   # Data analysis functions
├── report_utils.py     # Report generation utilities
├── requirements.txt    # Project dependencies
└── .env               # Environment variables
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenAI GPT-4o](https://openai.com/gpt-4)
- Developed by the Dimagi team
