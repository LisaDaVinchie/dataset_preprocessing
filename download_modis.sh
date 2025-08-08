#!/bin/bash
#SBATCH --job-name=modis_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --output=logs/modis_dl_%j.out
#SBATCH --error=logs/modis_dl_%j.err

source bot_codes.txt
bot_id=${bot_id}
chat_id=${chat_id}

set -e

notify_telegram() {
    local status="$1"
    curl -s -X POST "https://api.telegram.org/bot${bot_id}/sendMessage" \
    -d chat_id=${chat_id} \
    -d text="Your dataset_preprocessing download job (Job ID: $SLURM_JOB_ID) has completed with status: $status"
}

trap 'notify_telegram "FAILED (job terminated or timed out)"' TERM EXIT

source ./venv_download/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

START_YEAR=2021
END_YEAR=2021
START_MONTH=1
END_MONTH=12

START_DATE=$(printf "%04d-%02d-01T00:00:00Z" $START_YEAR $START_MONTH)
if [ "$END_MONTH" -eq 12 ]; then
    END_DATE=$(printf "%04d-01-01T00:00:00Z" $(($END_YEAR + 1)))
else
    END_DATE=$(printf "%04d-%02d-01T00:00:00Z" $END_YEAR $(($END_MONTH + 1)))
fi

echo "Start Date: $START_DATE"
echo "End Date:   $END_DATE"


DESTINATION_DIR="./data/modis/raw/"


podaac-data-downloader -c MODIS_TERRA_L3_SST_THERMAL_DAILY_4KM_NIGHTTIME_V2019.0 -d $DESTINATION_DIR --start-date $START_DATE --end-date $END_DATE -e ""
deactivate

rm -rf $DESTINATION_DIR/*.NRT.nc

notify_telegram "SUCCESS"

# Clear the EXIT trap to avoid double notifications
trap - EXIT