#!/bin/bash
#
# Test script for SGLang Server
# Verifies that the SGLang server is running and responding correctly
#

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SERVER_URL="http://localhost:8001"

echo "Testing SGLang Server at $SERVER_URL"
echo ""

# Test 1: Check if server is reachable
echo -n "1. Checking if server is running... "
if curl -s -f "$SERVER_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Server is not reachable. Make sure it's running with ./start-sglang.sh"
    exit 1
fi

# Test 2: Get server info
echo -n "2. Getting server info... "
response=$(curl -s "$SERVER_URL/get_model_info")
if [ -n "$response" ]; then
    echo -e "${GREEN}OK${NC}"
    echo "   Model info: $response"
else
    echo -e "${YELLOW}WARNING${NC} - Could not get model info"
fi

# Test 3: Simple completion test
echo -n "3. Testing completion API... "
response=$(curl -s "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.0
  }')

if echo "$response" | grep -q "Paris"; then
    echo -e "${GREEN}OK${NC}"
    echo "   Response: $(echo $response | jq -r '.choices[0].text' 2>/dev/null || echo $response)"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "   Response: $response"
fi

# Test 4: Chat completion test
echo -n "4. Testing chat completion API... "
response=$(curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Say hello in one word"}
    ],
    "max_tokens": 10,
    "temperature": 0.0
  }')

if echo "$response" | grep -q "choices"; then
    echo -e "${GREEN}OK${NC}"
    echo "   Response: $(echo $response | jq -r '.choices[0].message.content' 2>/dev/null || echo $response)"
else
    echo -e "${YELLOW}WARNING${NC}"
    echo "   Response: $response"
fi

echo ""
echo -e "${GREEN}All tests completed!${NC}"
echo ""
echo "Server is ready to use. Example usage:"
echo ""
echo "curl http://localhost:8001/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"model\": \"default\","
echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
echo "    \"max_tokens\": 100"
echo "  }'"
