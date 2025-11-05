# Contributing to Reddit Sentiment Dashboard

Thank you for considering contributing to the Reddit Sentiment Dashboard! 🎉

This document provides guidelines for contributing to this project. Following these guidelines helps maintain code quality and makes the review process smoother.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

---

## 🤝 Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- **Be respectful** and considerate in your communication
- **Be collaborative** and help others learn
- **Be patient** with maintainers and contributors
- **Focus on** what is best for the community
- **Show empathy** towards other community members

Unacceptable behavior includes harassment, trolling, insulting comments, and personal attacks.

---

## 🎯 How Can I Contribute?

### 1. Reporting Bugs 🐛

Before submitting a bug report:
- Check the [existing issues](https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard/issues)
- Try the latest version of the code
- Collect information about the bug

When submitting a bug report, include:
- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details**:
  - OS (Windows/Linux/Mac)
  - Python version
  - Browser (if relevant)
  - Package versions

**Bug Report Template**:
```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.5
- Browser: Chrome 120

## Screenshots
[If applicable]

## Additional Context
Any other relevant information
```

---

### 2. Suggesting Enhancements 💡

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement:

- **Use a clear title** describing the enhancement
- **Provide detailed description** of the suggested enhancement
- **Explain why** this enhancement would be useful
- **List alternatives** you've considered
- **Include mockups** or examples if applicable

**Enhancement Template**:
```markdown
## Feature Description
Clear description of the feature

## Problem it Solves
What problem does this solve?

## Proposed Solution
How should this work?

## Alternatives Considered
What other solutions did you consider?

## Additional Context
Mockups, examples, related issues
```

---

### 3. Code Contributions 💻

We welcome code contributions! Here's how to get started:

#### Types of Contributions

- **Bug fixes**: Fix existing bugs
- **New features**: Implement new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve README, guides, or code comments
- **Tests**: Add or improve test coverage
- **Refactoring**: Improve code structure without changing behavior

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)
- Docker (optional, for containerized development)

### Local Setup

1. **Fork the repository** on GitHub

2. **Clone your fork**:
```bash
git clone https://github.com/YOUR_USERNAME/Reddit-Sentiment-Dashboard.git
cd Reddit-Sentiment-Dashboard
```

3. **Add upstream remote**:
```bash
git remote add upstream https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard.git
```

4. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

5. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

6. **Set up .env file**:
```bash
cp .env.example .env
# Add your Reddit API credentials
```

7. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

### Development Workflow

1. **Keep your fork synced**:
```bash
git fetch upstream
git merge upstream/main
```

2. **Make your changes**
   - Write clean, readable code
   - Follow coding standards (see below)
   - Add tests for new features
   - Update documentation

3. **Test your changes**:
```bash
# Run the application
streamlit run main_code.py

# Run tests (if available)
pytest

# Run linting
flake8 .
black --check .
```

4. **Commit your changes**:
```bash
git add .
git commit -m "feat: add amazing new feature"
```

5. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request** on GitHub

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with some modifications:

- **Line length**: 120 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Code Formatting

We use **Black** for code formatting:
```bash
# Format all files
black .

# Check formatting without changing files
black --check .
```

### Linting

We use **Flake8** for linting:
```bash
# Run linting
flake8 . --max-line-length=120
```

### Type Hints

Use type hints for function parameters and return values:
```python
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text."""
    return {"sentiment": "positive", "confidence": 0.95}
```

### Documentation

#### Docstrings

Use Google-style docstrings:
```python
def fetch_reddit_posts(subreddit: str, limit: int = 10) -> list:
    """
    Fetch posts from a subreddit.

    Args:
        subreddit: Name of the subreddit
        limit: Maximum number of posts to fetch

    Returns:
        List of post dictionaries

    Raises:
        ValueError: If subreddit is invalid
        ConnectionError: If Reddit API is unreachable

    Example:
        >>> posts = fetch_reddit_posts("python", limit=5)
        >>> len(posts)
        5
    """
    pass
```

#### Comments

- Write clear, concise comments
- Explain **why**, not **what**
- Update comments when code changes
- Avoid obvious comments

```python
# Good
# Use exponential backoff to avoid rate limiting
time.sleep(2 ** retry_count)

# Bad
# Sleep for 2 seconds
time.sleep(2)
```

---

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings

### PR Title Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Test additions or changes
- `chore:` Maintenance tasks

**Examples**:
- `feat: add dark mode toggle`
- `fix: resolve rate limiting issue`
- `docs: update installation instructions`
- `refactor: simplify sentiment analysis logic`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Fixes #123

## Changes Made
- Added feature X
- Updated component Y
- Refactored function Z

## Testing
- [ ] Tested locally
- [ ] Added unit tests
- [ ] Tested on different browsers/OS

## Screenshots
[If applicable]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Tests pass
```

### Review Process

1. **Automated checks** run (CI/CD pipeline)
2. **Maintainer review** (usually within 2-3 days)
3. **Address feedback** if requested
4. **Approval** from maintainer
5. **Merge** into main branch

### After Your PR is Merged

- Delete your feature branch
- Update your local main branch
- Celebrate! 🎉

---

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_sentiment.py
import pytest
from app.multi_model_sentiment import analyze_sentiment_multi

class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""

    def test_positive_sentiment(self):
        """Test detection of positive sentiment."""
        text = "I love this product!"
        result = analyze_sentiment_multi(text)
        assert result['cardiff']['label'] == 'positive'

    def test_empty_input(self):
        """Test handling of empty input."""
        result = analyze_sentiment_multi("")
        assert 'error' in result

    @pytest.mark.parametrize("text,expected", [
        ("Great!", "positive"),
        ("Terrible!", "negative"),
        ("It's okay", "neutral"),
    ])
    def test_multiple_cases(self, text, expected):
        """Test multiple sentiment cases."""
        result = analyze_sentiment_multi(text)
        assert result['cardiff']['label'] == expected
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_sentiment.py

# Run specific test
pytest tests/test_sentiment.py::TestSentimentAnalysis::test_positive_sentiment
```

---

## 🏗️ Project Structure

Understanding the project structure helps in contributing:

```
Reddit-Sentiment-Dashboard/
├── app/                          # Application modules
│   ├── __init__.py
│   ├── multi_model_sentiment.py  # Sentiment analysis
│   ├── emotion_detector.py       # Emotion detection
│   └── sentiment_analyzer.py     # Legacy analyzer
├── tests/                        # Test files
│   ├── __init__.py
│   ├── test_sentiment.py
│   └── test_emotion.py
├── .github/                      # GitHub configuration
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
├── main_code.py                 # Main application
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose
├── README.md                    # Project documentation
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # This file
├── DEPLOYMENT.md                # Deployment guide
├── ROADMAP.md                   # Future plans
└── LICENSE                      # MIT License
```

---

## 📚 Resources

### Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PRAW Documentation](https://praw.readthedocs.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Python Best Practices](https://docs.python-guide.org/)

### Tools

- [Visual Studio Code](https://code.visualstudio.com/) - Recommended IDE
- [Black](https://black.readthedocs.io/) - Code formatter
- [Flake8](https://flake8.pycqa.org/) - Linter
- [Pytest](https://docs.pytest.org/) - Testing framework

---

## 💬 Communication

### Where to Ask Questions

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code-specific discussions

### Response Times

- Bug reports: 1-3 days
- Feature requests: 3-7 days
- Pull requests: 2-5 days

Please be patient! Maintainers are often volunteers with limited time.

---

## 🏆 Recognition

Contributors are recognized in:
- README.md (Contributors section)
- CHANGELOG.md (per release)
- GitHub contributors page

Your contributions, no matter how small, are valued and appreciated! 🙏

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## ❓ Questions?

If you have questions not covered in this guide, feel free to:
- Open a [GitHub Discussion](https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard/discussions)
- Create an [Issue](https://github.com/DharmpratapSingh/Reddit-Sentiment-Dashboard/issues)

**Happy Contributing! 🚀**
