#!/bin/bash

# --- Configuration ---
BASE_URL="http://127.0.0.1:5001"
HEADER_JSON="Content-Type: application/json"
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Function ---
print_header() {
  echo ""
  echo -e "${BLUE}======================================================${NC}"
  echo -e "${BLUE}  $1 ${NC}"
  echo -e "${BLUE}======================================================${NC}"
}

# --- Test Functions ---
test_home() {
  print_header "Testing Home Endpoint (GET /)"
  curl -s ${BASE_URL}/ | jq
}

test_health() {
  print_header "Testing Health Check (GET /health)"
  curl -s ${BASE_URL}/health | jq
}

test_predict_valid() {
  print_header "Making a Valid Prediction (POST /predict)"
  curl -s -X POST -H "${HEADER_JSON}" -d '{"features": [5.1, 3.5, 1.4, 0.2]}' ${BASE_URL}/predict | jq
}

test_predict_invalid() {
  print_header "Making an Invalid Prediction (POST /predict)"
  curl -s -X POST -H "${HEADER_JSON}" -d '{"features": [-1.0, 3.5, 1.4, 0.2]}' ${BASE_URL}/predict | jq
}

test_add_data() {
  print_header "Adding New Training Data (POST /add_training_data)"
  curl -s -X POST -H "${HEADER_JSON}" -d '{"features": [4.9, 3.0, 1.4, 0.2], "target": 0}' ${BASE_URL}/add_training_data | jq
}

test_trigger_retrain() {
  print_header "Triggering a Forced Retrain (POST /trigger_retrain)"
  curl -s -X POST -H "${HEADER_JSON}" -d '{"force_retrain": true}' ${BASE_URL}/trigger_retrain | jq
}

test_system_metrics() {
  print_header "Getting System Metrics (GET /system_metrics)"
  curl -s ${BASE_URL}/system_metrics | jq
}

test_history() {
  print_header "Getting Prediction History (GET /predictions/history)"
  curl -s "${BASE_URL}/predictions/history?limit=5" | jq
}

test_prometheus() {
  print_header "Getting Prometheus Metrics (GET /metrics)"
  curl -s ${BASE_URL}/metrics | head -n 10 # Displaying first 10 lines
}

test_dashboard() {
  print_header "Getting Dashboard HTML (GET /dashboard)"
  curl -s ${BASE_URL}/dashboard | grep "ML Model Monitoring Dashboard"
}

run_all_tests() {
  test_home
  test_health
  test_predict_valid
  test_predict_invalid
  test_add_data
  test_trigger_retrain
  test_system_metrics
  test_history
  test_prometheus
  test_dashboard
}

# --- Main Menu Loop ---
while true; do
  echo ""
  echo -e "${BLUE}--- Iris API Test Menu ---${NC}"
  echo " 1) Test Home Endpoint"
  echo " 2) Test Health Check"
  echo " 3) Test Predict (Valid)"
  echo " 4) Test Predict (Invalid)"
  echo " 5) Test Add Training Data"
  echo " 6) Test Trigger Retrain"
  echo " 7) Test System Metrics"
  echo " 8) Test Prediction History"
  echo " 9) Test Prometheus Metrics"
  echo "10) Test Dashboard HTML"
  echo ""
  echo "11) Run ALL Tests"
  echo " 0) Exit"
  echo ""
  read -p "Enter your choice [0-11]: " choice

  case $choice in
    1) test_home ;;
    2) test_health ;;
    3) test_predict_valid ;;
    4) test_predict_invalid ;;
    5) test_add_data ;;
    6) test_trigger_retrain ;;
    7) test_system_metrics ;;
    8) test_history ;;
    9) test_prometheus ;;
    10) test_dashboard ;;
    11) run_all_tests ;;
    0)
      echo "Exiting."
      break
      ;;
    *)
      echo "Invalid option. Please try again."
      ;;
  esac
done