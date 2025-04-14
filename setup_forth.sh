# setup env for forth
export HDF5_USE_FILE_LOCKING=FALSE # turn off file locking for HDF5 error. Reduces errors.
export AUSRAIN=~stett2/software/aus_rain_analysis
export AUSRAIN_HIST_DIR=$AUSRAIN/histories/ # where history info goes
conda activate $AUSRAIN/venv # activate python env
# config files
export AUSRAIN_CONFIG_DIR=$AUSRAIN/config_files/forth_config/
# paths and stuff
export PYTHONPATH=$PYTHONPATH:~stett2/software/commonLib:$AUSRAIN
export PATH=$AUSRAIN:$AUSRAIN/process_submit:$PATH
# give some info to user.
echo "setup complete."
module list
echo  "Virtual env is $VIRTUAL_ENV"
