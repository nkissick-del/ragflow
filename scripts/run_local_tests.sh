#!/bin/bash
# =============================================================================
# Local Unit Test Runner for Mac
# =============================================================================
#
# This script sets up a minimal Python environment and runs unit tests
# without requiring Docker or heavy dependencies like infinity/opendal.
#
# WHAT WORKS LOCALLY:
#   - Semantic template tests (test_semantic.py)
#   - Standardized document tests (test_standardized_document.py)
#
# WHAT REQUIRES DOCKER:
#   - Orchestrator tests (deep dependency chain)
#   - Integration tests (needs services)
#
# Usage:
#   ./scripts/run_local_tests.sh              # Run semantic tests (default)
#   ./scripts/run_local_tests.sh semantic     # Run semantic tests
#   ./scripts/run_local_tests.sh setup        # Just set up the venv
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv-test"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RAGFlow Local Test Runner ===${NC}"

# Check for python3 and validate version >= 3.10
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
     echo "Error: Python 3.10+ is required. Found $(python3 --version 2>&1)."
     exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install dependencies if needed
REQ_FILE="$PROJECT_ROOT/requirements-test.txt"
MARKER_FILE="$VENV_DIR/.deps_installed"

if [ ! -f "$REQ_FILE" ]; then
    echo "Error: $REQ_FILE not found."
    exit 1
fi

if [ ! -f "$MARKER_FILE" ] || [ "$REQ_FILE" -nt "$MARKER_FILE" ]; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install --upgrade pip -q
    pip install -r "$REQ_FILE" -q
    touch "$MARKER_FILE"
    echo -e "${GREEN}Dependencies installed.${NC}"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Handle commands
case "${1:-semantic}" in
    setup)
        echo -e "${GREEN}Setup complete. Virtual environment ready at: $VENV_DIR${NC}"
        echo "To activate: source $VENV_DIR/bin/activate"
        ;;
    semantic)
        echo -e "${YELLOW}Running semantic chunking tests...${NC}"
        pytest "$PROJECT_ROOT/test/unit_test/rag/app/test_standardized_document.py" \
               "$PROJECT_ROOT/test/unit_test/rag/app/templates/test_semantic.py" \
               -v --tb=short
        ;;
    all)
        # Run all unit tests that don't require Docker dependencies
        # Excludes: test_orchestrator.py, test_format_parsers.py (require heavy deps/imports)
        # Includes: standardized_document, semantic template, and other standalone tests
        echo -e "${YELLOW}Running all locally-compatible unit tests...${NC}"
        pytest "$PROJECT_ROOT/test/unit_test/rag/app" \
               --ignore="$PROJECT_ROOT/test/unit_test/rag/app/test_orchestrator.py" \
               --ignore="$PROJECT_ROOT/test/unit_test/rag/app/test_format_parsers.py" \
               -v --tb=short
        ;;
    *)
        # Run specific test file
        if [ -f "$1" ]; then
            echo -e "${YELLOW}Running: $1${NC}"
            pytest "$1" -v --tb=short
        else
            echo "Usage: $0 [setup|semantic|all|<test_file>]"
            echo ""
            echo "Local tests (no Docker needed):"
            echo "  semantic    - Semantic template tests (default)"
            echo "  all         - All locally-compatible tests"
            echo ""
            echo "Docker-only tests (run in container):"
            echo "  docker exec -it ragflow pytest test/unit_test/rag/app/test_orchestrator.py -v"
            exit 1
        fi
        ;;
esac

echo -e "${GREEN}Done.${NC}"
