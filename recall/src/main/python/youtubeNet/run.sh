# time
time=$(date +"%m%d%H%M")

# run
mkdir -p train_log
python3 main.py -e=3 -r > "train_log/$time" 2>&1
