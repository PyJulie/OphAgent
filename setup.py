from setuptools import setup, find_packages

setup(
    name="ophagent",
    version="0.1.0",
    description="OphAgent: An LLM-driven ophthalmic AI agent system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.40.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.40.0",
        "faiss-cpu>=1.8.0",
        "fastapi>=0.111.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "PyYAML>=6.0.1",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=8.2.0", "pytest-asyncio>=0.23.0"],
    },
)
