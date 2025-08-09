#!/bin/bash
#SBATCH --job-name=modis_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=4GB
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source bot_codes.txt
bot_id=${bot_id}
chat_id=${chat_id}

set -euo pipefail

notify_telegram() {
    local status="$1"
    curl -s -X POST "https://api.telegram.org/bot${bot_id}/sendMessage" \
        -d chat_id=${chat_id} \
        -d text="Your MODIS download job (Job ID: $SLURM_JOB_ID) completed with status: $status"
}

trap 'notify_telegram "FAILED (job terminated or timed out)"' TERM EXIT

source ./venv_download/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }


YEARS="2000 2001 2002 2003 2008 2016 2017 2022"

START_MONTH=1
END_MONTH=12
DESTINATION_DIR="./data/modis/raw/"

download_and_zip_year() {
    local year=$1
    local start_date
    local end_date

    start_date=$(printf "%04d-%02d-01T00:00:00Z" "$year" "$START_MONTH")
    end_date=$(printf "%04d-01-01T00:00:00Z" $((year + 1)))

    echo "[Year $year] Downloading from $start_date to $end_date"

    podaac-data-downloader \
        -c MODIS_TERRA_L3_SST_THERMAL_DAILY_4KM_NIGHTTIME_V2019.0 \
        -d "$DESTINATION_DIR" \
        --start-date "$start_date" \
        --end-date "$end_date" \
        -e ".nc"

    # Remove NRT files
    rm -f "$DESTINATION_DIR"/*.NRT.nc

    # Zip all .nc for this year
    local zip_name="${DESTINATION_DIR}/${year}.zip"
    find "$DESTINATION_DIR" -type f -name "TERRA_MODIS.${year}*.L3m.DAY.NSST.sst.4km.nc" \
        | zip -j "$zip_name" -@

    # Delete unzipped files for this year
    rm -f "$DESTINATION_DIR"/TERRA_MODIS.${year}*.L3m.DAY.NSST.sst.4km.nc

    echo "[Year $year] Done."
}

export -f download_and_zip_year
export DESTINATION_DIR START_MONTH END_MONTH

echo "Using $SLURM_CPUS_PER_TASK jobs in parallel"
# Run specified years in parallel
parallel -j $SLURM_CPUS_PER_TASK download_and_zip_year ::: $YEARS

deactivate

notify_telegram "SUCCESS"
trap - EXIT
