.PHONY: help fix check check-manual test test-unit test-integration test-cov clean install

help:
	@echo "Available commands:"
	@echo "  make fix              - Auto-fix linting issues (black, isort)"
	@echo "  make check            - Check code quality without fixing (flake8, black, isort)"
	@echo "  make check-manual     - Show issues that require manual fixing (unused imports, etc.)"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make clean            - Remove build artifacts and cache files"
	@echo "  make install          - Install package in editable mode"

fix:
	@echo "Running black formatter..."
	black ssa/ tests/
	@echo "Running isort import sorter..."
	isort ssa/ tests/
	@echo "✓ Code formatting complete"
	@echo ""
	@echo "Note: 'make fix' only handles code formatting (black, isort)"
	@echo "To check for issues requiring manual fixes, run: make check-manual"

check:
	@echo "Checking code with flake8..."
	flake8 ssa/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ssa/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Checking code with black..."
	black --check ssa/ tests/
	@echo "Checking imports with isort..."
	isort --check-only ssa/ tests/
	@echo "✓ Code quality checks complete"

check-manual:
	@echo "Checking for issues requiring manual fixes..."
	@echo ""
	@echo "=== Unused Imports (F401) and Variables (F841) ==="
	@flake8 ssa/ tests/ --select=F401,F841 --show-source || echo "✓ No unused imports or variables found"
	@echo ""
	@echo "=== Undefined Names (F821) ==="
	@flake8 ssa/ tests/ --select=F821 --show-source || echo "✓ No undefined names found"
	@echo ""
	@echo "Note: Complexity warnings (C901) are informational and indicate"
	@echo "      functions that may benefit from refactoring, but do not break builds."

test:
	@echo "Running all tests..."
	python3 -m pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	python3 -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	python3 -m pytest tests/integration/ -v

test-cov:
	@echo "Running tests with coverage..."
	python3 -m pytest tests/ -v --cov=ssa --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

install:
	@echo "Installing package in editable mode..."
	pip install -e .
	@echo "Installing development dependencies..."
	pip install black isort flake8
	@echo "✓ Installation complete"
