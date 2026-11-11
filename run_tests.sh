#!/usr/bin/env bash
# ==============================================================================
# run_tests.sh — Unified 4-Tier Automated E2E Test Suite Runner
# ==============================================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Determine Python binary
if command -v /usr/bin/python3 >/dev/null 2>&1; then
    PYTHON_BIN="/usr/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Configure NVML compat library path
COMPAT_DIR="/data/shangguanrz/仪表盘/compat"
if [[ -d "$COMPAT_DIR" ]]; then
    export LD_LIBRARY_PATH="$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
elif [[ -d "$BASE_DIR/compat" ]]; then
    export LD_LIBRARY_PATH="$BASE_DIR/compat:${LD_LIBRARY_PATH:-}"
fi

echo -e "${BLUE}==========================================================================${NC}"
echo -e "${BOLD}${CYAN}   GPU & RAM Monitoring Dashboard — Automated 4-Tier Test Runner${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo -e " Target Base Directory : ${BOLD}$BASE_DIR${NC}"
echo -e " Python Executable     : ${BOLD}$PYTHON_BIN${NC} ($($PYTHON_BIN --version 2>&1))"
echo -e " LD_LIBRARY_PATH       : ${BOLD}${LD_LIBRARY_PATH:-None}${NC}"
echo -e "${BLUE}--------------------------------------------------------------------------${NC}"

TOTAL_PASSED=0
TOTAL_FAILED=0
START_TIME=$(date +%s)

# Helper function to run a specific tier
run_tier() {
    local tier_num="$1"
    local tier_name="$2"
    local test_file="$3"

    echo ""
    echo -e "${YELLOW}>> Running Tier $tier_num: $tier_name ($test_file)...${NC}"
    echo -e "${BLUE}--------------------------------------------------------------------------${NC}"

    if "$PYTHON_BIN" -m unittest "$test_file" -v; then
        echo -e "${GREEN}✓ [PASS] Tier $tier_num: $tier_name completed successfully.${NC}"
        return 0
    else
        echo -e "${RED}✗ [FAIL] Tier $tier_num: $tier_name encountered test failures or errors.${NC}"
        return 1
    fi
}

TIER1_FAILED=0
TIER2_FAILED=0
TIER3_FAILED=0
TIER4_FAILED=0
TIER5_FAILED=0

# Run Tier 1
run_tier "1" "Feature & Precision Coverage" "tests.test_tier1_feature" || TIER1_FAILED=1

# Run Tier 2
run_tier "2" "Boundary & Fault Tolerance" "tests.test_tier2_boundary" || TIER2_FAILED=1

# Run Tier 3
run_tier "3" "Process Lifecycle Integration" "tests.test_tier3_lifecycle" || TIER3_FAILED=1

# Run Tier 4
run_tier "4" "Stress, Latency SLA & Air-Gapped Assets" "tests.test_tier4_stress" || TIER4_FAILED=1

# Run Tier 5 (Memory Deep Dashboard v2.0)
run_tier "5" "Memory Deep Analytics & History API" "tests.test_memory_dashboard" || TIER5_FAILED=1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

TOTAL_FAILED=$((TIER1_FAILED + TIER2_FAILED + TIER3_FAILED + TIER4_FAILED + TIER5_FAILED))

echo ""
echo -e "${BLUE}==========================================================================${NC}"
echo -e "${BOLD}${CYAN}                      TEST SUITE EXECUTION SUMMARY                        ${NC}"
echo -e "${BLUE}==========================================================================${NC}"
echo -e " Execution Duration : ${ELAPSED} seconds"
echo -e " Tier 1 (Features)  : $([[ $TIER1_FAILED -eq 0 ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e " Tier 2 (Boundary)  : $([[ $TIER2_FAILED -eq 0 ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e " Tier 3 (Lifecycle) : $([[ $TIER3_FAILED -eq 0 ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e " Tier 4 (Stress/SLA): $([[ $TIER4_FAILED -eq 0 ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e " Tier 5 (Mem Deep)  : $([[ $TIER5_FAILED -eq 0 ]] && echo -e "${GREEN}PASSED${NC}" || echo -e "${RED}FAILED${NC}")"
echo -e "${BLUE}--------------------------------------------------------------------------${NC}"

# Restore production daemon on default port if stopped during testing
if [[ -f "$BASE_DIR/scripts/start.sh" ]]; then
    bash "$BASE_DIR/scripts/start.sh" >/dev/null 2>&1 || true
fi

if [[ $TOTAL_FAILED -eq 0 ]]; then
    echo -e "${BOLD}${GREEN} [ALL PASSED] All 5 Tiers of Automated E2E Tests Succeeded Cleanly!${NC}"
    echo -e "${BLUE}==========================================================================${NC}"
    exit 0
else
    echo -e "${BOLD}${RED} [FAILED] $TOTAL_FAILED tier(s) failed. Check details above.${NC}"
    echo -e "${BLUE}==========================================================================${NC}"
    exit 1
fi
