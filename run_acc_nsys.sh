#!/bin/bash

source params.sh


# Check if profiling is enabled
PROFILE=false
for arg in "$@"; do
    if [[ $arg == "--profile" ]]; then
        PROFILE=true
    fi
done

# Create necessary directories
if [ ! -d $CPP_OUT_FOLDER ]; then
    mkdir $CPP_OUT_FOLDER
fi
if [ ! -d $MEASURES_FOLDER ]; then
    mkdir $MEASURES_FOLDER
fi

# Create a directory for profiling data
PROFILE_FOLDER="$MEASURES_FOLDER/profiles"
if [ ! -d $PROFILE_FOLDER ]; then
    mkdir $PROFILE_FOLDER
fi

# Check if model and dataset files are available
if [ ! -f $MODEL_PATH ]; then
    echo Error: missing model $MODEL_PATH!
    exit 1
fi

if [ ! -d $DTASET_FOLDER ]; then
    echo Error: missing dataset $DTASET_FOLDER!
    exit 1
fi

# Check if executable is available
if [ ! -f "acc_bin/vit.exe" ]; then
    echo Error: missing acc_bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi

# Create the CSV file if it doesn't exist
if [ ! -f $MEASURES_FOLDER/cpp.csv ]; then
    touch $MEASURES_FOLDER/cpp.csv
fi

# Initialize CSV with headers
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >$MEASURES_FOLDER/cpp.csv

# Run the profiling loop if --profile flag is set
if $PROFILE; then
    echo "Running with nsys profiling..."
    
    # Create a unique timestamp for each run
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    
    # Loop over dataset
    for ((i=0; i<$DTASET_DIM; i++)); do
        # Run the executable with nsys profiling, store the output in the PROFILE_FOLDER with a unique name
        nsys profile --output=$PROFILE_FOLDER/profile_${TIMESTAMP}_$i.nsys-rep ./acc_bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $CPP_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/cpp.csv
    done
else
    # Without profiling, just run the normal execution
    for ((i=0; i<$DTASET_DIM; i++)); do
        ./acc_bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $CPP_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/cpp.csv
    done
fi

echo vit executed

# Run Python script to analyze the time measures
python3 scripts/analyze_time_measures.py $MEASURES_FOLDER/cpp.csv $MEASURES_FOLDER/cpp_summary.txt
