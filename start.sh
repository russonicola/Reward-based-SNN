#!/bin/bash

# Define a cleanup function
cleanup() {
  echo "Cleaning up..."
  # Kill all child processes of this script
  pkill -P $$
}

# Trap SIGINT and SIGTERM signals
trap cleanup SIGINT SIGTERM

# Start the timer to stop the script after 1 hour (3600 seconds) + init and cooling + extra 1 min
# (sleep 4260 && kill -SIGTERM $$) &

# Start mcu server
echo "Start mcu server"
echo "$output" | python3 virtual_mcu.py &
MCU_SERVER_PID=$!
sleep 10

if [ -f /.dockerenv ]; then
    echo "Running inside a Docker container."
else
    # Activate environment
    echo "Activate environment"
    source .env/bin/activate
    echo "Not running inside a Docker container."
fi



echo "Initialise shared memory"
python shared_memory.py &
SHARED_CREATE_LIGHT_PID=$!
sleep 5
echo "Run rimulation"
echo "$output" | python decision_unit.py &
SNN_REPLICA_PID=$!
echo "$output" | python control_unit.py &
BOKEH_LITE_PID=$!

# execute for 120 minutes
sleep 7200

echo "$output" | curl -X GET http://127.0.0.1:5001/accuracy

kill $MCU_SERVER_PID
kill $BOKEH_LITE_PID
kill $SHARED_CREATE_LIGHT_PID


wait

# Remove the temporary file
#rm mem_config.json


echo "Main script exited cleanly."
