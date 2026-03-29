#!/bin/bash

# Ralph - Autonomous AI Agent Loop for TurboQuant & RaBitQ Research & Implementation
# Based on snarktank/ralph: https://github.com/snarktank/ralph

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_ITERATIONS=${1:-10}
TOOL="claude"  # Using Claude Code

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Ralph - TurboQuant & RaBitQ Research & Implementation        ║"
echo "║  Max iterations: $MAX_ITERATIONS                                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Create directories if they don't exist
mkdir -p "$SCRIPT_DIR/research"
mkdir -p "$SCRIPT_DIR/src"
mkdir -p "$SCRIPT_DIR/tests"
mkdir -p "$SCRIPT_DIR/integrations/glm5"
mkdir -p "$SCRIPT_DIR/integrations/deepseek"
mkdir -p "$SCRIPT_DIR/benchmarks"
mkdir -p "$SCRIPT_DIR/docs"
mkdir -p "$SCRIPT_DIR/examples"

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: TurboQuant & RaBitQ project setup"
fi

# Check if branch exists, create if not
BRANCH_NAME=$(jq -r '.branchName' "$SCRIPT_DIR/prd.json")
CURRENT_BRANCH=$(git branch --show-current)

if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
    if git show-ref --quiet refs/heads/"$BRANCH_NAME"; then
        echo "Switching to branch: $BRANCH_NAME"
        git checkout "$BRANCH_NAME"
    else
        echo "Creating branch: $BRANCH_NAME"
        git checkout -b "$BRANCH_NAME"
    fi
fi

# Main loop
for ((i=1; i<=MAX_ITERATIONS; i++)); do
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Iteration $i of $MAX_ITERATIONS"
    echo "═══════════════════════════════════════════════════════════════"
    
    # Check if all stories are complete
    INCOMPLETE=$(jq '[.userStories[] | select(.passes == false)] | length' "$SCRIPT_DIR/prd.json")
    if [ "$INCOMPLETE" -eq 0 ]; then
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║  🎉 ALL STORIES COMPLETE!                                    ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        exit 0
    fi
    
    # Show current status
    echo ""
    echo "📋 Current Status:"
    jq -r '.userStories[] | "\(.passes | if . then "✅" else "⬜" end) \(.id): \(.title)"' "$SCRIPT_DIR/prd.json"
    echo ""
    
    # Get next story to work on
    NEXT_STORY=$(jq -r '[.userStories[] | select(.passes == false)] | sort_by(.priority) | .[0].id' "$SCRIPT_DIR/prd.json")
    NEXT_TITLE=$(jq -r ".userStories[] | select(.id == \"$NEXT_STORY\") | .title" "$SCRIPT_DIR/prd.json")
    
    echo "🎯 Next Story: $NEXT_STORY - $NEXT_TITLE"
    echo ""
    
    # Run Claude Code with the CLAUDE.md prompt
    echo "🤖 Starting Claude Code iteration..."
    OUTPUT=$(claude --dangerously-skip-permissions --print < "$SCRIPT_DIR/CLAUDE.md" 2>&1 | tee /dev/stderr) || true
    
    # Check for completion signal
    if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
        echo ""
        echo "╔═══════════════════════════════════════════════════════════════╗"
        echo "║  🎉 ALL STORIES COMPLETE!                                    ║"
        echo "╚═══════════════════════════════════════════════════════════════╝"
        exit 0
    fi
    
    echo ""
    echo "⏸️  Iteration $i complete. Pausing before next iteration..."
    sleep 2
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ⚠️  Max iterations reached ($MAX_ITERATIONS)                              ║"
echo "║  Check prd.json for remaining incomplete stories              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Final status
echo ""
echo "📊 Final Status:"
jq -r '.userStories[] | "\(.passes | if . then "✅" else "⬜" end) \(.id): \(.title)"' "$SCRIPT_DIR/prd.json"

COMPLETE=$(jq '[.userStories[] | select(.passes == true)] | length' "$SCRIPT_DIR/prd.json")
TOTAL=$(jq '.userStories | length' "$SCRIPT_DIR/prd.json")
echo ""
echo "Progress: $COMPLETE/$TOTAL stories complete"
