TIMESTEP3=data/json/timestep3_dev.json
OUTPUT_DATA=data/pointer/pointer_dev.json
python -m src.pointer.get_pointer_data -t $TIMESTEP3 -o $OUTPUT_DATA

MODEL=output/pointer/20-0.01.hdf5
DATA=data/pointer/pointer_dev.json
GPU=0
OUTPUT=output/pointer/dev_epoch20.json
python -m src.pointer.pointer -m $MODEL -d $DATA -g $GPU -o $OUTPUT