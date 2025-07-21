PARAMS_FILE="./src/params.json"

START_YEAR=$(jq -r '.dataset.year_range[0]' $PARAMS_FILE)
END_YEAR=$(jq -r '.dataset.year_range[1]' $PARAMS_FILE)
START_MONTH=$(jq -r '.dataset.month_range[0]' $PARAMS_FILE)
END_MONTH=$(jq -r '.dataset.month_range[1]' $PARAMS_FILE)

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

rm -rf $DESTINATION_DIR/*.NRT.nc