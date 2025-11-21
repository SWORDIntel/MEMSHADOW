"""
MEMSHADOW Python SDK Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="memshadow-sdk",
    version="0.1.0",
    author="MEMSHADOW Team",
    author_email="contact@memshadow.dev",
    description="Python SDK for MEMSHADOW memory persistence with LLM wrappers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SWORDIntel/MEMSHADOW",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.7.0"],
        "all": ["openai>=1.0.0", "anthropic>=0.7.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="memory llm ai chatgpt claude openai anthropic semantic-search embeddings",
    project_urls={
        "Bug Reports": "https://github.com/SWORDIntel/MEMSHADOW/issues",
        "Source": "https://github.com/SWORDIntel/MEMSHADOW",
        "Documentation": "https://github.com/SWORDIntel/MEMSHADOW/blob/main/sdk/README.md",
    },
)
